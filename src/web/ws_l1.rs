// Ern-OS — Layer 1 tool chain handler (extracted from ws.rs for governance compliance).

use crate::provider::Message;
use crate::tools::schema;
use crate::web::state::AppState;
use crate::web::ws_learning::PendingToolChain;
use crate::web::ws_stream::{audit_and_retry, send_ws};
use crate::inference::stream_consumer::ConsumeResult;
use crate::web::ws_learning::{ingest_assistant_turn, spawn_insight_extraction};
use crate::web::ws_react::run_react_loop;
use axum::extract::ws::{Message as WsMessage, WebSocket};

/// Deliver a reply: audit, sanitize, stream, ingest, extract insights.
pub async fn deliver_reply(
    state: &AppState,
    provider: &dyn crate::provider::Provider,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
    messages: &mut Vec<Message>,
    tools: &serde_json::Value,
    user_query: &str,
    session_id: &str,
    text: &str,
    thinking: &Option<String>,
) {
    tracing::info!(text_len = text.len(), has_thinking = thinking.is_some(), "L1 result: Reply");
    if let Some(ref t) = thinking {
        tracing::debug!(len = t.len(), "Thinking captured for audit");
    }
    let approved_text = audit_and_retry(state, provider, sender, messages, tools, user_query, text, session_id).await;

    // Post-audit sanitization — last-resort scrubber for leaked tool output
    let scrub = crate::web::output_sanitizer::scrub_tool_leaks(&approved_text);
    let final_text = if crate::web::output_sanitizer::needs_reinference(&scrub) {
        // Leaked content was ALL there was — attempt silent re-inference
        tracing::warn!("Sanitizer stripped entire response — triggering re-inference");
        messages.push(Message::text("system", "[SYSTEM: Your previous response leaked raw tool output. Synthesize the tool results into a natural language response for the user.]"));
        match provider.chat_sync(messages, Some(tools)).await {
            Ok(retry) => {
                let retry_scrub = crate::web::output_sanitizer::scrub_tool_leaks(&retry);
                // DPO capture: leaked original → rejection, clean retry → chosen
                capture_leak_dpo(state, user_query, &retry_scrub.text, &approved_text).await;
                retry_scrub.text
            }
            Err(e) => {
                // Re-inference failed — deliver original output, never a canned message
                tracing::error!(error = %e, "Re-inference failed — delivering original output");
                capture_leak_dpo(state, user_query, &approved_text, &approved_text).await;
                approved_text.clone()
            }
        }
    } else if scrub.had_leak {
        // Partial leak was stripped — capture for DPO
        capture_leak_dpo(state, user_query, &scrub.text, &approved_text).await;
        scrub.text
    } else {
        scrub.text
    };

    send_ws(sender, "text_delta", &serde_json::json!({"content": &final_text})).await;
    ingest_assistant_turn(state, &final_text, session_id).await;
    spawn_insight_extraction(state, user_query, &final_text);
    send_ws(sender, "done", &serde_json::json!({})).await;
}

/// Run the L1 tool chain loop — handles chained tool calls.
/// The model decides when to stop (by emitting a reply or escalating to ReAct).
/// Safety cap of 50 prevents true runaway — matches the ReAct loop safety_cap.
pub async fn run_l1_tool_chain(
    state: &AppState,
    provider: &dyn crate::provider::Provider,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
    messages: &mut Vec<Message>,
    tools: &serde_json::Value,
    content: &str,
    session_id: &str,
    first_tc: schema::ToolCall,
    pending_chain: &mut Option<PendingToolChain>,
    stop_flag: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    let max_tool_iterations = 50;
    let mut current_tc = first_tc;
    let mut chain_tools: Vec<(String, String)> = Vec::new();
    let mut chain_reply = String::new();

    for tool_iter in 0..max_tool_iterations {
        tracing::info!(tool = %current_tc.name, id = %current_tc.id, iteration = tool_iter, "L1 tool call iteration");
        send_ws(sender, "tool_executing", &serde_json::json!({"name": &current_tc.name, "id": &current_tc.id})).await;
        chain_tools.push((current_tc.name.clone(), current_tc.arguments.clone()));

        let mut result = crate::web::tool_dispatch::execute_tool_with_state(state, &current_tc).await;

        // Auto-stitch: if file_read produced a [BOOKMARK], fetch remaining pages
        if current_tc.name == "file_read" && result.success {
            if let Ok(args) = serde_json::from_str::<serde_json::Value>(&current_tc.arguments) {
                if crate::tools::file_read::parse_bookmark(&result.output).is_some() {
                    result.output = crate::tools::file_read::auto_stitch(
                        &result.output, &args, state.model_spec.context_length,
                    ).await;
                }
            }
        }

        tracing::info!(tool = %current_tc.name, success = result.success, output_len = result.output.len(), "L1 tool: execution complete");

        send_ws(sender, "tool_completed", &serde_json::json!({
            "id": &current_tc.id, "name": &current_tc.name,
            "result": &result.output, "success": result.success,
        })).await;

        messages.push(Message::assistant_tool_call(&current_tc.id, &current_tc.name, &current_tc.arguments));
        if result.images.is_empty() {
            messages.push(Message::tool_result(&current_tc.id, &result.output));
        } else {
            messages.push(Message::tool_result_multipart(&current_tc.id, &result.output, result.images));
        }
        crate::web::handlers::platform_context::enforce_context_budget(
            provider, messages, Some(tools), state.model_spec.context_length, false,
        ).await;

        // thinking=false: the model already reasoned during initial inference.
        // Re-enabling thinking for tool result processing causes Gemma 4 to emit
        // stop with no content (same bug fixed in platform_reinfer.rs, commit 1a1e3f3).
        let rx_next = match provider.chat(messages, Some(tools), false).await {
            Ok(rx) => rx,
            Err(e) => { tracing::error!(error = %e, "L1 re-inference FAILED"); break; }
        };

        // Use thinking-only sink: text is accumulated silently, thinking
        // deltas stream live. Text only reaches the WebUI after observer
        // audit approves it via deliver_reply → audit_and_retry.
        use crate::inference::stream_consumer::{self, ThinkingOnlySink};
        let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let mut sink = ThinkingOnlySink { sender, cancel };
        let result = stream_consumer::consume_stream(rx_next, &mut sink).await;

        // Handle spiral inline
        let result = match result {
            ConsumeResult::Spiral { .. } => {
                stream_consumer::reprompt_after_spiral(
                    provider, messages, Some(tools), &mut sink,
                ).await
            }
            other => other,
        };

        match result {
            ConsumeResult::Reply { text, thinking } => {
                // §2.7: Empty reply (model leaked tool syntax into thinking) — never deliver blank
                if text.trim().is_empty() {
                    tracing::warn!(iteration = tool_iter + 1, "L1 chain: empty reply after tool calls — re-prompting");
                    messages.push(Message::text("system",
                        "[SYSTEM: Your response was empty. Synthesize all tool results above into a response for the user.]"
                    ));
                    match provider.chat_sync(messages, Some(tools)).await {
                        Ok(retry_text) if !retry_text.trim().is_empty() => {
                            deliver_reply(state, provider, sink.sender, messages, tools, content, session_id, &retry_text, &None).await;
                            chain_reply = retry_text;
                        }
                        _ => {
                            let error_msg = "I processed your request but was unable to generate a response. Please try again.";
                            tracing::error!("L1 chain: re-prompt also failed — delivering error");
                            deliver_reply(state, provider, sink.sender, messages, tools, content, session_id, error_msg, &None).await;
                            chain_reply = error_msg.to_string();
                        }
                    }
                    stash_chain(pending_chain, chain_tools, content, &chain_reply, session_id);
                    return;
                }
                deliver_reply(state, provider, sink.sender, messages, tools, content, session_id, &text, &thinking).await;
                chain_reply = text;
                stash_chain(pending_chain, chain_tools, content, &chain_reply, session_id);
                return;
            }
            ConsumeResult::ToolCall { id, name, arguments } => {
                tracing::info!(tool = %name, id = %id, iteration = tool_iter + 1, "L1 chain → another ToolCall");
                current_tc = schema::ToolCall { id, name, arguments };
            }
            ConsumeResult::ToolCalls(calls) => {
                // Model emitted multiple tool calls in this chain iteration.
                // Dispatch all but the last inline (with UI events and
                // message appends), then continue the chain using the last
                // call as the next current_tc. The next loop iteration will
                // execute that last call and re-prompt the model with all
                // accumulated tool results.
                let count = calls.len();
                tracing::info!(count, iteration = tool_iter + 1, "L1 chain → ToolCalls (multi-call dispatch)");
                // Track leading tools in chain_tools so stash_chain captures the full set.
                chain_tools.extend(
                    calls.iter().take(count.saturating_sub(1))
                        .map(|(_, name, args)| (name.clone(), args.clone()))
                );
                match dispatch_leading_tool_calls(state, sink.sender, messages, calls).await {
                    Some(last_tc) => current_tc = last_tc,
                    None => {
                        // ToolCalls with empty Vec — stream_consumer invariant violation.
                        // Surface and break the chain rather than spin silently.
                        tracing::error!(iteration = tool_iter + 1, "L1 chain: ToolCalls(empty) — stream_consumer invariant violated");
                        send_ws(sink.sender, "error", &serde_json::json!({"message": "Internal error: empty tool call set in chain"})).await;
                        break;
                    }
                }
            }
            ConsumeResult::Escalate { objective, plan, planned_turns } => {
                tracing::info!(objective = %objective, planned_turns, "L1 chain → Escalate");
                stop_flag.store(false, std::sync::atomic::Ordering::Relaxed);
                send_ws(sink.sender, "status", &serde_json::json!({"message": format!("ReAct loop activated ({} turns planned)", planned_turns)})).await;
                run_react_loop(state, provider, messages.clone(), &objective, plan.as_deref(), planned_turns, content, session_id, sink.sender, stop_flag).await;
                return;
            }
            ConsumeResult::PlanProposal { title, plan_markdown, estimated_turns } => {
                tracing::info!(title = %title, turns = estimated_turns, "L1 chain → PlanProposal");
                let plan = crate::web::ws_plans::save_pending_plan(session_id, &title, &plan_markdown, estimated_turns);
                send_ws(sink.sender, "plan_proposal", &serde_json::json!({
                    "title": plan.title,
                    "plan_markdown": plan.plan_markdown,
                    "estimated_turns": plan.estimated_turns,
                    "revision": plan.revision,
                    "session_id": session_id,
                })).await;
                send_ws(sink.sender, "done", &serde_json::json!({})).await;
                return;
            }
            ConsumeResult::Error(e) => {
                tracing::error!(error = %e, "L1 chain → Error");
                send_ws(sink.sender, "error", &serde_json::json!({"message": e})).await;
                break;
            }
            _ => {}
        }
    }

    stash_chain(pending_chain, chain_tools, content, &chain_reply, session_id);
    send_ws(sender, "done", &serde_json::json!({})).await;
}

/// Execute leading tool calls inline and return the last call for the caller
/// to drive the L1 chain. Used when the model emits multiple tool calls in
/// a single inference turn — the leading calls are dispatched concretely
/// (sending tool_executing/tool_completed UI events and appending assistant
/// tool_call + tool_result messages), then the caller starts the L1 chain
/// using the last call as its entry point. The chain's re-prompt then sees
/// the full set of leading results in the message history.
///
/// Returns `None` only if `calls` is empty, which violates the
/// `stream_consumer::ConsumeResult::ToolCalls` invariant (the variant is
/// constructed only for `len() > 1`). Callers must surface this case
/// explicitly rather than silently dropping the turn.
pub async fn dispatch_leading_tool_calls(
    state: &AppState,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
    messages: &mut Vec<Message>,
    calls: Vec<(String, String, String)>,
) -> Option<schema::ToolCall> {
    let mut tcs: Vec<schema::ToolCall> = calls.into_iter()
        .map(|(id, name, arguments)| schema::ToolCall { id, name, arguments })
        .collect();
    let last_tc = tcs.pop()?;
    for tc in &tcs {
        send_ws(sender, "tool_executing", &serde_json::json!({"name": &tc.name, "id": &tc.id})).await;
        let result = crate::web::tool_dispatch::execute_tool_with_state(state, tc).await;
        tracing::info!(tool = %tc.name, success = result.success, output_len = result.output.len(), "L1 leading tool: execution complete");
        send_ws(sender, "tool_completed", &serde_json::json!({
            "id": &tc.id, "name": &tc.name,
            "result": &result.output, "success": result.success,
        })).await;
        messages.push(Message::assistant_tool_call(&tc.id, &tc.name, &tc.arguments));
        if result.images.is_empty() {
            messages.push(Message::tool_result(&tc.id, &result.output));
        } else {
            messages.push(Message::tool_result_multipart(&tc.id, &result.output, result.images));
        }
    }
    // Budget enforcement is intentionally deferred to the chain: the
    // caller passes `last_tc` to `run_l1_tool_chain`, which calls
    // `enforce_context_budget` after each tool result iteration
    // (see ws_l1.rs `run_l1_tool_chain` ~ L116). Mirrors the canonical
    // pattern in `platform_ingest.rs:192-215` which also does not
    // enforce budget during leading-call dispatch.
    Some(last_tc)
}

/// Stash a completed tool chain for delayed reinforcement on the next turn.
pub fn stash_chain(
    pending_chain: &mut Option<PendingToolChain>,
    chain_tools: Vec<(String, String)>,
    user_query: &str,
    chain_reply: &str,
    session_id: &str,
) {
    if !chain_tools.is_empty() && !chain_reply.is_empty() {
        *pending_chain = Some(PendingToolChain {
            tools: chain_tools,
            user_query: user_query.to_string(),
            reply: chain_reply.to_string(),
            _session_id: session_id.to_string(),
        });
    }
}

/// Capture a sanitizer-caught leak as a DPO rejection pair.
/// The leaked text becomes the rejection, the clean text becomes the chosen.
async fn capture_leak_dpo(state: &AppState, user_query: &str, chosen: &str, rejected: &str) {
    if chosen.is_empty() || rejected.is_empty() {
        return;
    }
    let mut buf = state.rejection_buffer.write().await;
    if let Err(e) = buf.add_pair(
        user_query, chosen, rejected, "output_sanitizer: tool output leak",
    ) {
        tracing::error!(error = %e, "Failed to capture leak DPO pair");
    } else {
        tracing::info!("DPO pair captured: tool output leak → rejection buffer");
    }
}
