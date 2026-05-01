//! Platform execution engines — L1 tool chain and L2 ReAct loop for platforms.
//!
//! These run the full tool execution pipeline for platform-ingested messages,
//! collecting ToolEvent metadata for thinking thread observability.
//!
//! Split into three modules per governance §1.1:
//! - `platform_exec` (this file): orchestrators
//! - `platform_context`: context budget management
//! - `platform_reinfer`: re-inference dispatch

use crate::web::state::AppState;
use super::platform_ingest::{ToolEvent, AuditSummary, audit_and_capture};
use super::platform_reinfer::{LoopAction, reinfer_and_dispatch, inject_react_instruction};

// Re-export public items so existing callers don't need to change import paths.
pub use super::platform_context::{
    append_tool_messages, enforce_context_budget, build_tool_context,
};

/// Execute a tool and capture a ToolEvent with timing.
/// For `file_read` calls, auto-stitches paginated results transparently.
pub async fn execute_and_capture(
    state: &AppState,
    tc: &crate::tools::schema::ToolCall,
) -> (crate::tools::schema::ToolResult, ToolEvent) {
    let start = std::time::Instant::now();
    let mut result = crate::web::tool_dispatch::execute_tool_with_state(state, tc).await;

    // Auto-stitch: if this is a file_read with a [BOOKMARK], transparently
    // fetch remaining pages without burning an inference round.
    if tc.name == "file_read" && result.success {
        if let Ok(args) = serde_json::from_str::<serde_json::Value>(&tc.arguments) {
            if crate::tools::file_read::parse_bookmark(&result.output).is_some() {
                let stitched = crate::tools::file_read::auto_stitch(
                    &result.output, &args, state.model_spec.context_length,
                ).await;
                result.output = stitched;
            }
        }
    }

    let elapsed = start.elapsed().as_millis() as u64;

    let preview = if result.output.len() > 300 {
        let boundary = result.output.char_indices()
            .take_while(|(i, _)| *i <= 300)
            .last()
            .map(|(i, _)| i)
            .unwrap_or(0);
        format!("{}…", &result.output[..boundary])
    } else {
        result.output.clone()
    };

    let event = ToolEvent {
        name: tc.name.clone(),
        success: result.success,
        elapsed_ms: elapsed,
        output_preview: preview,
    };

    tracing::info!(
        tool = %tc.name,
        success = result.success,
        elapsed_ms = elapsed,
        "Platform tool executed"
    );

    (result, event)
}

/// Run L1 tool chain. Returns (reply, tool_events, audit_summary).
/// If `sse_tx` is provided, tool events are emitted live to keep the SSE stream alive.
pub async fn run_platform_tool_chain(
    state: &AppState,
    provider: &dyn crate::provider::Provider,
    messages: &mut Vec<crate::provider::Message>,
    tools: &serde_json::Value,
    user_query: &str,
    session_id: &str,
    first_tc: crate::tools::schema::ToolCall,
    sse_tx: Option<&tokio::sync::mpsc::Sender<Result<axum::response::sse::Event, std::convert::Infallible>>>,
) -> (String, Vec<ToolEvent>, Option<AuditSummary>) {
    let max_iterations = 50;
    let mut current_tc = first_tc;
    let mut tool_events: Vec<ToolEvent> = Vec::new();

    for iteration in 0..max_iterations {
        tracing::info!(tool = %current_tc.name, iteration, "Platform L1 tool execution");

        // Emit tool_start live
        if let Some(tx) = sse_tx {
            let _ = tx.send(Ok(axum::response::sse::Event::default()
                .event("tool_start")
                .data(serde_json::json!({"name": &current_tc.name}).to_string())
            )).await;
        }

        let (result, event) = execute_and_capture(state, &current_tc).await;

        // Emit tool_result live
        if let Some(tx) = sse_tx {
            let _ = tx.send(Ok(axum::response::sse::Event::default()
                .event("tool_result")
                .data(serde_json::json!({
                    "name": event.name, "success": event.success,
                    "elapsed_ms": event.elapsed_ms, "preview": event.output_preview,
                }).to_string())
            )).await;
        }

        tool_events.push(event);
        append_tool_messages(messages, &current_tc, &result);

        let total_chars: usize = messages.iter().map(|m| m.text_content().len()).sum();
        let tool_msg_count = messages.iter().filter(|m| m.role == "tool").count();
        tracing::info!(
            iteration,
            tool = %current_tc.name,
            msg_count = messages.len(),
            tool_msgs = tool_msg_count,
            total_chars,
            estimated_tokens = total_chars / 2 + 2000,
            context_length = state.model_spec.context_length,
            "Tool chain: context state before budget enforcement"
        );
        enforce_context_budget(messages, state.model_spec.context_length);

        // Pre-infer budget check
        let post_trim_chars: usize = messages.iter().map(|m| m.text_content().len()).sum();
        let post_trim_tokens = post_trim_chars / 2 + 2000;
        let budget_ratio = post_trim_tokens as f64 / state.model_spec.context_length as f64;
        if budget_ratio > 0.90 {
            tracing::warn!(
                post_trim_tokens,
                context_length = state.model_spec.context_length,
                budget_ratio = format!("{:.1}%", budget_ratio * 100.0),
                "Context at {:.0}% of window after trimming — stopping tool chain",
                budget_ratio * 100.0,
            );
            let reply = format!(
                "Context is at {:.0}% capacity ({} tokens used of {} available) after {} tool calls. \
                 Please continue in a new message — bookmarks are preserved for continuation.",
                budget_ratio * 100.0,
                post_trim_tokens,
                state.model_spec.context_length,
                tool_events.len(),
            );
            return (reply, tool_events, None);
        }

        match reinfer_and_dispatch(state, provider, messages, tools, user_query, session_id).await {
            LoopAction::Reply(text, audit) => return (text, tool_events, Some(audit)),
            LoopAction::NextTool(tc) => current_tc = tc,
            LoopAction::MultiTool(tcs) => {
                let mut remaining = tcs;
                let last_tc = remaining.pop();
                for tc in remaining {
                    tracing::info!(tool = %tc.name, iteration, "Platform L1 parallel tool execution");
                    if let Some(tx) = sse_tx {
                        let _ = tx.send(Ok(axum::response::sse::Event::default()
                            .event("tool_start")
                            .data(serde_json::json!({"name": &tc.name}).to_string())
                        )).await;
                    }
                    let (result, event) = execute_and_capture(state, &tc).await;
                    if let Some(tx) = sse_tx {
                        let _ = tx.send(Ok(axum::response::sse::Event::default()
                            .event("tool_result")
                            .data(serde_json::json!({
                                "name": event.name, "success": event.success,
                                "elapsed_ms": event.elapsed_ms, "preview": event.output_preview,
                            }).to_string())
                        )).await;
                    }
                    tool_events.push(event);
                    append_tool_messages(messages, &tc, &result);
                }
                match last_tc {
                    Some(tc) => current_tc = tc,
                    None => return ("No tool calls to execute.".to_string(), tool_events, None),
                }
            }
            LoopAction::Escalate(reply, events, audit) => {
                tool_events.extend(events);
                return (reply, tool_events, audit);
            }
            LoopAction::Error(msg) => return (msg, tool_events, None),
        }
    }

    ("Tool chain reached maximum iterations.".to_string(), tool_events, None)
}

/// Run L2 ReAct loop. Returns (reply, tool_events, audit_summary).
pub async fn run_platform_react(
    state: &AppState,
    provider: &dyn crate::provider::Provider,
    mut messages: Vec<crate::provider::Message>,
    objective: &str,
    plan: Option<&str>,
    user_query: &str,
    session_id: &str,
    sse_tx: Option<&tokio::sync::mpsc::Sender<Result<axum::response::sse::Event, std::convert::Infallible>>>,
) -> (String, Vec<ToolEvent>, Option<AuditSummary>) {
    let tools = crate::tools::schema::layer2_tools();
    let mut tool_events: Vec<ToolEvent> = Vec::new();
    inject_react_instruction(&mut messages, objective, plan);

    for turn in 0..50 {
        let rx = match provider.chat(&messages, Some(&tools), true).await {
            Ok(rx) => rx,
            Err(e) => return (format!("ReAct error: {}", e), tool_events, None),
        };
        use crate::inference::stream_consumer::{self, ConsumeResult, NullSink};
        let mut sink = NullSink;
        let result = stream_consumer::consume_stream(rx, &mut sink).await;
        match result {
            ConsumeResult::Spiral { .. } => {
                let recovered = stream_consumer::reprompt_after_spiral(
                    provider, &mut messages, Some(&tools), &mut sink,
                ).await;
                if let ConsumeResult::Reply { text, .. } = recovered {
                    let (audited, audit) = audit_and_capture(
                        state, provider, &mut messages, &tools, user_query, &text, session_id,
                    ).await;
                    return (audited, tool_events, Some(audit));
                }
            }
            ConsumeResult::Reply { text, .. } => {
                let (audited, audit) = audit_and_capture(
                    state, provider, &mut messages, &tools, user_query, &text, session_id,
                ).await;
                return (audited, tool_events, Some(audit));
            }
            ConsumeResult::ToolCall { id, name, arguments } => {
                let tc = crate::tools::schema::ToolCall { id, name, arguments };
                if let Some(result) = handle_react_tool(
                    state, &mut messages, &mut tool_events, &tc, provider, &tools, user_query, session_id, turn, sse_tx,
                ).await {
                    return result;
                }
            }
            ConsumeResult::Escalate { objective: nested, .. } => {
                tracing::info!(nested = %nested, turn, "Platform ReAct nested escalation");
                messages.push(crate::provider::Message::text("system", &format!("Additional objective: {}", nested)));
            }
            ConsumeResult::ToolCalls(calls) => {
                for (id, name, arguments) in calls {
                    let tc = crate::tools::schema::ToolCall { id, name, arguments };
                    if let Some(result) = handle_react_tool(
                        state, &mut messages, &mut tool_events, &tc, provider, &tools, user_query, session_id, turn, sse_tx,
                    ).await {
                        return result;
                    }
                }
            }
            ConsumeResult::Error(e) => return (format!("ReAct error: {}", e), tool_events, None),
            _ => {}
        }
    }
    ("ReAct loop reached maximum turns.".to_string(), tool_events, None)
}

/// Handle a tool call within the ReAct loop.
/// Returns Some if the loop should exit, None to continue.
async fn handle_react_tool(
    state: &AppState,
    messages: &mut Vec<crate::provider::Message>,
    tool_events: &mut Vec<ToolEvent>,
    tc: &crate::tools::schema::ToolCall,
    provider: &dyn crate::provider::Provider,
    tools: &serde_json::Value,
    user_query: &str,
    session_id: &str,
    turn: usize,
    sse_tx: Option<&tokio::sync::mpsc::Sender<Result<axum::response::sse::Event, std::convert::Infallible>>>,
) -> Option<(String, Vec<ToolEvent>, Option<AuditSummary>)> {
    // Check for reply_request (loop terminator)
    if let Some(reply_text) = crate::tools::schema::extract_reply_text(tc) {
        let (audited, audit) = audit_and_capture(
            state, provider, messages, tools, user_query, &reply_text, session_id,
        ).await;
        crate::web::ws_learning::ingest_assistant_turn(state, &audited, session_id).await;
        return Some((audited, tool_events.clone(), Some(audit)));
    }

    // Emit tool_start live to SSE if available
    if let Some(tx) = sse_tx {
        let _ = tx.send(Ok(axum::response::sse::Event::default()
            .event("tool_start")
            .data(serde_json::json!({"name": &tc.name}).to_string())
        )).await;
    }

    tracing::info!(tool = %tc.name, turn, "Platform ReAct tool execution");
    let (result, event) = execute_and_capture(state, tc).await;

    // Emit tool_result live to SSE if available
    if let Some(tx) = sse_tx {
        let _ = tx.send(Ok(axum::response::sse::Event::default()
            .event("tool_result")
            .data(serde_json::json!({
                "name": event.name, "success": event.success,
                "elapsed_ms": event.elapsed_ms, "preview": event.output_preview,
            }).to_string())
        )).await;
    }

    tool_events.push(event);
    append_tool_messages(messages, tc, &result);
    enforce_context_budget(messages, state.model_spec.context_length);
    None
}
