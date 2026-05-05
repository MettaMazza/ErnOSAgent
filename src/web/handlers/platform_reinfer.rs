//! Re-inference dispatch for platform tool chains.
//!
//! After a tool executes, re-runs inference to determine the next action:
//! continue with another tool, reply to the user, or escalate to ReAct.

use crate::web::state::AppState;
use super::platform_ingest::{ToolEvent, AuditSummary, audit_and_capture};
use super::platform_context::enforce_context_budget;

/// Result of re-inference after tool execution.
pub(crate) enum LoopAction {
    Reply(String, AuditSummary),
    NextTool(crate::tools::schema::ToolCall),
    MultiTool(Vec<crate::tools::schema::ToolCall>),
    Escalate(String, Vec<ToolEvent>, Option<AuditSummary>),
    Error(String),
}

/// Re-infer after tool execution and classify the result.
pub(crate) async fn reinfer_and_dispatch(
    state: &AppState,
    provider: &dyn crate::provider::Provider,
    messages: &mut Vec<crate::provider::Message>,
    tools: &serde_json::Value,
    user_query: &str,
    session_id: &str,
) -> LoopAction {
    // thinking=false: the model already reasoned during initial inference.
    // Re-enabling thinking for tool result processing causes Gemma 4 to emit
    // stop with no content (verified via llama-server timings: cache_n=48933,
    // prompt_n=88, elapsed=327ms — immediate stop after processing tool result).
    let rx = match provider.chat(messages, Some(tools), false).await {
        Ok(rx) => rx,
        Err(e) => {
            tracing::error!(error = %e, "Platform L1 re-inference failed");
            return LoopAction::Error(format!("Tool execution error: {}", e));
        }
    };

    use crate::inference::stream_consumer::{self, ConsumeResult, NullSink};
    let mut sink = NullSink;
    let result = stream_consumer::consume_stream(rx, &mut sink).await;

    match result {
        ConsumeResult::Spiral { .. } => {
            handle_spiral(state, provider, messages, tools, user_query, session_id).await
        }
        ConsumeResult::Reply { text, .. } if text.trim().is_empty() => {
            handle_empty_response(state, provider, messages, tools, user_query, session_id).await
        }
        ConsumeResult::Reply { text, .. } => {
            let (audited, audit) = audit_and_capture(
                state, provider, messages, tools, user_query, &text, session_id,
            ).await;
            LoopAction::Reply(audited, audit)
        }
        ConsumeResult::ToolCall { id, name, arguments } => {
            LoopAction::NextTool(crate::tools::schema::ToolCall { id, name, arguments })
        }
        ConsumeResult::ToolCalls(calls) => {
            let tcs: Vec<crate::tools::schema::ToolCall> = calls.into_iter()
                .map(|(id, name, arguments)| crate::tools::schema::ToolCall { id, name, arguments })
                .collect();
            LoopAction::MultiTool(tcs)
        }
        ConsumeResult::Escalate { objective, plan, .. } => {
            let (reply, events, audit) = super::platform_exec::run_platform_react(
                state, provider, messages.clone(), &objective, plan.as_deref(),
                user_query, session_id, None,
            ).await;
            LoopAction::Escalate(reply, events, audit)
        }
        ConsumeResult::Error(e) => {
            tracing::error!(error = %e, "Platform L1 stream error");
            LoopAction::Error(format!("Error: {}", e))
        }
        _ => LoopAction::Error("Unexpected stream result".to_string()),
    }
}

/// Handle spiral recovery during re-inference.
async fn handle_spiral(
    state: &AppState,
    provider: &dyn crate::provider::Provider,
    messages: &mut Vec<crate::provider::Message>,
    tools: &serde_json::Value,
    user_query: &str,
    session_id: &str,
) -> LoopAction {
    use crate::inference::stream_consumer::{self, ConsumeResult, NullSink};
    let mut sink = NullSink;
    let recovered = stream_consumer::reprompt_after_spiral(
        provider, messages, Some(tools), &mut sink,
    ).await;
    match recovered {
        ConsumeResult::Reply { text, .. } => {
            let (audited, audit) = audit_and_capture(
                state, provider, messages, tools, user_query, &text, session_id,
            ).await;
            LoopAction::Reply(audited, audit)
        }
        ConsumeResult::ToolCall { id, name, arguments } => {
            LoopAction::NextTool(crate::tools::schema::ToolCall { id, name, arguments })
        }
        ConsumeResult::Error(e) => LoopAction::Error(format!("Spiral recovery: {}", e)),
        _ => LoopAction::Error("Spiral recovery: unexpected result".to_string()),
    }
}

/// Handle empty response (context overflow) — trim and retry once.
async fn handle_empty_response(
    state: &AppState,
    provider: &dyn crate::provider::Provider,
    messages: &mut Vec<crate::provider::Message>,
    tools: &serde_json::Value,
    user_query: &str,
    session_id: &str,
) -> LoopAction {
    use crate::inference::stream_consumer::{self, ConsumeResult, NullSink};

    let total_chars: usize = messages.iter().map(|m| m.text_content().len()).sum();
    tracing::error!(
        total_chars,
        msg_count = messages.len(),
        estimated_tokens = total_chars / 3 + 2000,
        context_length = state.model_spec.context_length,
        "Model returned empty response after tool execution — trimming and retrying"
    );
    enforce_context_budget(messages, state.model_spec.context_length / 2);

    // Inject recovery prompt to escape KV cache collision.
    // Without this, the retry hits the same cached state and produces
    // the same empty response (observed: prompt_n=1 on retry = full cache hit).
    messages.push(crate::provider::Message::text("system",
        "[RECOVERY] Your previous response was empty. The tool results above \
         are valid and available. Process them and respond to the user's query."
    ));

    let retry_rx = match provider.chat(messages, Some(tools), false).await {
        Ok(rx) => rx,
        Err(e) => return LoopAction::Error(format!("Retry after empty response failed: {}", e)),
    };
    let mut retry_sink = NullSink;
    match stream_consumer::consume_stream(retry_rx, &mut retry_sink).await {
        ConsumeResult::Reply { text, .. } if !text.trim().is_empty() => {
            let (audited, audit) = audit_and_capture(
                state, provider, messages, tools, user_query, &text, session_id,
            ).await;
            LoopAction::Reply(audited, audit)
        }
        ConsumeResult::ToolCall { id, name, arguments } => {
            LoopAction::NextTool(crate::tools::schema::ToolCall { id, name, arguments })
        }
        _ => {
            let total_chars: usize = messages.iter().map(|m| m.text_content().len()).sum();
            let est_tokens = total_chars / 3 + 2000;
            tracing::error!(
                total_chars, est_tokens,
                context_length = state.model_spec.context_length,
                "Context exhausted — both initial and retry inferences returned empty"
            );
            let reply = format!(
                "Context window exhausted after tool execution ({} chars ≈ {} tokens vs {} token limit). \
                 The tool results have been preserved. Please continue in a new message.",
                total_chars, est_tokens, state.model_spec.context_length,
            );
            LoopAction::Error(reply)
        }
    }
}

/// Inject ReAct system instruction into messages.
pub(crate) fn inject_react_instruction(
    messages: &mut Vec<crate::provider::Message>,
    objective: &str,
    plan: Option<&str>,
) {
    let instruction = format!(
        "You are now in ReAct (Reason-Act-Observe) mode.\n\
         Objective: {}\n{}\n\
         Use tools to accomplish the objective. When done, call `reply_request` with your final response.",
        objective,
        plan.map(|p| format!("Plan:\n{}", p)).unwrap_or_default(),
    );
    messages.push(crate::provider::Message::text("system", &instruction));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inject_react_instruction() {
        let mut messages = Vec::new();
        inject_react_instruction(&mut messages, "Test objective", Some("Step 1"));
        assert_eq!(messages.len(), 1);
        let content = messages[0].text_content();
        assert!(content.contains("Test objective"));
        assert!(content.contains("Step 1"));
    }

    #[test]
    fn test_inject_react_no_plan() {
        let mut messages = Vec::new();
        inject_react_instruction(&mut messages, "Simple task", None);
        assert_eq!(messages.len(), 1);
        assert!(messages[0].text_content().contains("Simple task"));
    }
}
