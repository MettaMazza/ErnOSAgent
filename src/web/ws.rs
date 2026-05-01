// Ern-OS — WebSocket handler — real-time bidirectional communication.
// Fully wired: WebSocket → Provider → Stream → Observer → Deliver.

use crate::web::ws_learning::{self, PendingToolChain};
use crate::tools::schema;
use crate::web::state::AppState;
use axum::{
    extract::{State, WebSocketUpgrade, ws::{Message as WsMessage, WebSocket}},
    response::IntoResponse,
};
use futures_util::{StreamExt, SinkExt};

/// WebSocket upgrade handler.
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

/// Handle a single WebSocket connection.
async fn handle_socket(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();
    tracing::info!("WebSocket client connected");

    if send_welcome(&state, &mut sender).await.is_err() { return; }
    deliver_resume_if_pending(&state, &mut sender).await;

    let mut pending_chain: Option<PendingToolChain> = None;
    let stop_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));

    while let Some(msg_result) = receiver.next().await {
        let msg = match msg_result {
            Ok(m) => m,
            Err(e) => { tracing::warn!(error = %e, "WebSocket receive error"); break; }
        };
        match msg {
            WsMessage::Text(text) => {
                handle_text_message(&text, &state, &mut sender, &mut pending_chain, &stop_flag).await;
            }
            WsMessage::Close(_) => { tracing::info!("WebSocket client disconnected"); break; }
            _ => {}
        }
    }
}

/// Send the initial welcome message with model info.
async fn send_welcome(
    state: &AppState,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
) -> Result<(), ()> {
    let welcome = serde_json::json!({
        "type": "connected",
        "model": state.model_spec.name,
        "version": env!("CARGO_PKG_VERSION"),
    });
    sender.send(WsMessage::Text(welcome.to_string().into())).await.map_err(|e| {
        tracing::error!(error = %e, "Failed to send welcome");
    })
}

/// Deliver post-recompile resume message if one is pending for the web platform.
/// Loads the prior session context and runs real inference for a contextual wakeup.
/// Discord/Telegram resumes are handled by their own platform adapters.
async fn deliver_resume_if_pending(
    state: &AppState,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
) {
    // Only consume if platform is "web" — leave Discord/Telegram resumes for their adapters
    let resume = {
        let guard = state.resume_message.read().await;
        match guard.as_ref() {
            Some((_, _, platform)) if platform == "web" => true,
            _ => false,
        }
    };
    if !resume {
        return;
    }
    let (msg, session_id, _platform) = match state.resume_message.write().await.take() {
        Some(r) => r,
        None => return,
    };

    tracing::info!(msg_len = msg.len(), session_id = %session_id, "Delivering post-recompile resume (web)");

    // Tell the client to switch to the correct session BEFORE streaming text
    if !session_id.is_empty() {
        let switch_cmd = serde_json::json!({"type": "session_switch", "session_id": session_id});
        let _ = sender.send(WsMessage::Text(switch_cmd.to_string().into())).await;
        // Brief delay to let the client switch
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
    }

    // If we have a session_id, load it and re-infer with full context
    if !session_id.is_empty() {
        let session_messages = {
            let sessions = state.sessions.read().await;
            sessions.get(&session_id).map(|s| s.messages.clone())
        };

        if let Some(history) = session_messages {
            // Build context: system prompt + session history + recompile notification
            let mut messages = history;
            messages.push(crate::provider::Message::text("system", &format!(
                "[SYSTEM] You have just been recompiled successfully. \
                 Review the conversation above and greet the user, confirming the recompile \
                 succeeded and briefly summarising what you were working on before the restart."
            )));

            // Stream the response to the WebSocket
            let provider = match crate::provider::create_provider(&state.config) {
                Ok(p) => p,
                Err(e) => {
                    tracing::error!(error = %e, "Failed to create provider for resume re-inference");
                    let payload = serde_json::json!({"type": "text_delta", "content": format!("✅ {}", msg)});
                    let _ = sender.send(WsMessage::Text(payload.to_string().into())).await;
                    let done = serde_json::json!({"type": "done"});
                    let _ = sender.send(WsMessage::Text(done.to_string().into())).await;
                    return;
                }
            };
            match provider.chat(&messages, None, true).await {
                Ok(rx) => {
                    use crate::inference::stream_consumer::{self, WebSocketSink};
                    let cancel = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                    let mut sink = WebSocketSink { sender, cancel };
                    let _ = stream_consumer::consume_stream(rx, &mut sink).await;
                    let done = serde_json::json!({"type": "done"});
                    let _ = sender.send(WsMessage::Text(done.to_string().into())).await;
                    tracing::info!("Post-recompile resume delivered via re-inference");
                    return;
                }
                Err(e) => {
                    tracing::error!(error = %e, "Resume re-inference failed — falling back to canned message");
                }
            }
        }
    }

    // Fallback: canned message (no session context available)
    let payload = serde_json::json!({"type": "text_delta", "content": format!("✅ {}", msg)});
    let _ = sender.send(WsMessage::Text(payload.to_string().into())).await;
    let done = serde_json::json!({"type": "done"});
    let _ = sender.send(WsMessage::Text(done.to_string().into())).await;
}

/// A completed tool chain awaiting next-turn evaluation.
/// On the NEXT user message, we check if the user implicitly approved or rejected
/// the result, providing delayed reinforcement signal.
// PendingToolChain is imported from ws_learning.

/// Handle an incoming text message from the WebSocket client.
async fn handle_text_message(
    text: &str,
    state: &AppState,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
    pending_chain: &mut Option<PendingToolChain>,
    stop_flag: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    let parsed: serde_json::Value = match serde_json::from_str(text) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(raw_len = text.len(), error = %e, "WS: Invalid JSON received");
            send_ws(sender, "error", &serde_json::json!({"message": format!("Invalid JSON: {}", e)})).await;
            return;
        }
    };

    let msg_type = parsed["type"].as_str().unwrap_or("");
    tracing::debug!(msg_type = %msg_type, payload_len = text.len(), "WS: Incoming message");

    match msg_type {
        "chat" => {
            let content_len = parsed["content"].as_str().map(|s| s.len()).unwrap_or(0);
            let session = parsed["session_id"].as_str().unwrap_or("");
            tracing::info!(msg_type = "chat", content_len, session = %session, "WS: Chat message dispatched");
            handle_chat_message(&parsed, state, sender, pending_chain, &stop_flag).await;
        }
        "regenerate" => {
            let session_id = parsed["session_id"].as_str().unwrap_or("");
            tracing::info!(session = %session_id, "WS: Regenerate requested");
            handle_regenerate(&parsed, state, sender, pending_chain, &stop_flag).await;
        }
        "edit_and_resend" => {
            let session_id = parsed["session_id"].as_str().unwrap_or("");
            tracing::info!(session = %session_id, "WS: Edit and resend requested");
            handle_edit_and_resend(&parsed, state, sender, pending_chain, &stop_flag).await;
        }
        "stop" => {
            // Cancel any running inference — the stream consumer checks this flag
            // on every event and drops the receiver when it sees cancellation,
            // killing the HTTP connection to the inference server.
            state.cancel_flag.store(true, std::sync::atomic::Ordering::Relaxed);
            stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
            tracing::info!("WS: Stop requested — cancelling inference and ReAct loop");
            send_ws(sender, "stopped", &serde_json::json!({})).await;
        }
        "stop_react" => {
            state.cancel_flag.store(true, std::sync::atomic::Ordering::Relaxed);
            stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
            tracing::info!("WS: ReAct loop stop requested — cancelling inference");
        }
        "plan_decision" => {
            let session_id = parsed["session_id"].as_str().unwrap_or("");
            tracing::info!(session = %session_id, "WS: Plan decision received");
            handle_plan_decision(&parsed, state, sender, pending_chain, &stop_flag).await;
        }
        "set_autonomy" => {
            let level = parsed["level"].as_str().unwrap_or("supervised");
            let max_turns = parsed["max_turns"].as_u64().unwrap_or(200) as usize;
            let report_interval = parsed["report_interval"].as_u64().unwrap_or(5) as usize;
            let pause_on_fail = parsed["pause_on_failure"].as_bool().unwrap_or(true);
            let allow_destructive = parsed["allow_destructive"].as_bool().unwrap_or(false);
            let autonomy_level = match level {
                "interactive" => crate::inference::autonomy::AutonomyLevel::Interactive,
                "autonomous" => crate::inference::autonomy::AutonomyLevel::Autonomous,
                _ => crate::inference::autonomy::AutonomyLevel::Supervised,
            };
            let config = crate::inference::autonomy::AutonomyConfig {
                level: autonomy_level,
                max_turns,
                report_interval_steps: report_interval,
                pause_on_failure: pause_on_fail,
                allow_destructive,
            };
            tracing::info!(level = %level, max_turns, "WS: Autonomy config updated");
            send_ws(sender, "autonomy_set", &serde_json::json!({
                "level": level,
                "max_turns": config.max_turns,
                "report_interval": config.report_interval_steps,
                "pause_on_failure": config.pause_on_failure,
                "allow_destructive": config.allow_destructive,
            })).await;
        }
        "get_autonomy" => {
            let default_config = crate::inference::autonomy::AutonomyConfig::default();
            send_ws(sender, "autonomy_status", &serde_json::json!({
                "level": format!("{:?}", default_config.level).to_lowercase(),
                "max_turns": default_config.max_turns,
                "report_interval": default_config.report_interval_steps,
                "pause_on_failure": default_config.pause_on_failure,
                "allow_destructive": default_config.allow_destructive,
            })).await;
        }
        _ => {
            tracing::warn!(msg_type = %msg_type, raw = %text, "WS: Unknown message type received");
            send_ws(sender, "error", &serde_json::json!({"message": format!("Unknown type: {}", msg_type)})).await;
        }
    }
}

/// Handle a chat message — route through the full dual-layer inference engine.
async fn handle_chat_message(
    msg: &serde_json::Value,
    state: &AppState,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
    pending_chain: &mut Option<PendingToolChain>,
    stop_flag: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    let content = msg["content"].as_str().unwrap_or("");
    let session_id = msg["session_id"].as_str().unwrap_or("");
    let agent_id = msg["agent_id"].as_str();

    if content.is_empty() {
        send_ws(sender, "error", &serde_json::json!({"message": "Empty message"})).await;
        return;
    }

    tracing::info!(session = %session_id, content_len = content.len(), agent = ?agent_id, "Chat message received");

    // Evaluate PREVIOUS tool chain based on user's implicit feedback
    if let Some(chain) = pending_chain.take() {
        spawn_delayed_reinforcement(state, &chain, content);
    }

    send_ws(sender, "ack", &serde_json::json!({"session_id": session_id})).await;

    // ── Build full inference context (prompts + memory + history + consolidation) ──
    let images: Vec<String> = msg["images"]
        .as_array()
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect())
        .unwrap_or_default();

    // Select tools BEFORE building context so consolidation can account for tool overhead
    let tools = schema::layer1_tools();
    let tools_chars = tools.to_string().len();

    let ctx = crate::web::ws_context::build_chat_context(
        state, content, session_id, agent_id, images, "web", tools_chars,
    ).await;
    let mut messages = ctx.messages;
    let provider = state.provider.as_ref();

    tracing::info!(session = %session_id, content_len = content.len(), "L1 inference START");

    let rx = match provider.chat(&messages, Some(&tools), true).await {
        Ok(rx) => rx,
        Err(e) => {
            tracing::error!(error = %e, "L1 inference provider.chat FAILED");
            send_ws(sender, "error", &serde_json::json!({"message": format!("Provider error: {}", e)})).await;
            return;
        }
    };

    // Reset cancel flag before starting new inference
    state.cancel_flag.store(false, std::sync::atomic::Ordering::Relaxed);

    // Consume stream using the unified consumer with WebSocket sink
    use crate::inference::stream_consumer::{self, ConsumeResult, WebSocketSink};
    let cancel = state.cancel_flag.clone();
    let mut sink = WebSocketSink { sender, cancel };
    let result = stream_consumer::consume_stream(rx, &mut sink).await;

    // Handle spiral: re-prompt with thinking disabled
    let result = match result {
        ConsumeResult::Spiral { .. } => {
            tracing::info!("WS: re-prompting after spiral detection");
            stream_consumer::reprompt_after_spiral(
                provider, &mut messages, Some(&tools), &mut sink,
            ).await
        }
        other => other,
    };

    match result {
        ConsumeResult::Reply { text, thinking } => {
            crate::tools::introspect_tool::log_reasoning_event(
                &state.config.general.data_dir, session_id,
                &serde_json::json!({"type":"inference","result":"reply","text_len":text.len(),"thinking_len":thinking.as_ref().map(|t|t.len()).unwrap_or(0)}));
            deliver_reply(state, provider, sink.sender, &mut messages, &tools, content, session_id, &text, &thinking).await;
        }
        ConsumeResult::Escalate { objective, plan, planned_turns } => {
            crate::tools::introspect_tool::log_reasoning_event(
                &state.config.general.data_dir, session_id,
                &serde_json::json!({"type":"inference","result":"escalate","objective":&objective,"planned_turns":planned_turns}));
            tracing::info!(objective = %objective, planned_turns, "L1 result: Escalate → ReAct");
            stop_flag.store(false, std::sync::atomic::Ordering::Relaxed);
            send_ws(sink.sender, "status", &serde_json::json!({"message": format!("ReAct loop activated ({} turns planned)", planned_turns)})).await;
            run_react_loop(state, provider, messages, &objective, plan.as_deref(), planned_turns, content, session_id, sink.sender, stop_flag).await;
        }
        ConsumeResult::PlanProposal { title, plan_markdown, estimated_turns } => {
            tracing::info!(title = %title, turns = estimated_turns, "L1 result: PlanProposal → awaiting user approval");
            let plan = crate::web::ws_plans::save_pending_plan(session_id, &title, &plan_markdown, estimated_turns);
            send_ws(sink.sender, "plan_proposal", &serde_json::json!({
                "title": plan.title,
                "plan_markdown": plan.plan_markdown,
                "estimated_turns": plan.estimated_turns,
                "revision": plan.revision,
                "session_id": session_id,
            })).await;
        }
        ConsumeResult::ToolCall { id, name, arguments } => {
            crate::tools::introspect_tool::log_reasoning_event(
                &state.config.general.data_dir, session_id,
                &serde_json::json!({"type":"inference","result":"tool_call","tool":&name}));
            let tc = schema::ToolCall { id, name, arguments };
            run_l1_tool_chain(state, provider, sink.sender, &mut messages, &tools, content, session_id, tc, pending_chain, stop_flag).await;
        }
        ConsumeResult::Error(e) => {
            tracing::error!(error = %e, "L1 result: Error");
            send_ws(sink.sender, "error", &serde_json::json!({"message": e})).await;
        }
        ConsumeResult::Cancelled { text, .. } => {
            if text.is_empty() {
                send_ws(sink.sender, "error", &serde_json::json!({"message": "Inference cancelled by user"})).await;
            } else {
                // Deliver partial text — same as old consume_with_cancel behavior
                deliver_reply(state, provider, sink.sender, &mut messages, &tools, content, session_id, &text, &None).await;
            }
        }
        _ => {}
    }
}

/// Handle a regenerate request — pop last assistant response and re-run inference.
async fn handle_regenerate(
    msg: &serde_json::Value,
    state: &AppState,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
    pending_chain: &mut Option<PendingToolChain>,
    stop_flag: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    let session_id = msg["session_id"].as_str().unwrap_or("");
    if session_id.is_empty() {
        send_ws(sender, "error", &serde_json::json!({"message": "Missing session_id"})).await;
        return;
    }

    // Load session and remove last assistant message(s) + the user message being regenerated
    {
        let mut sessions = state.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            // Pop trailing assistant messages
            while session.messages.last().map_or(false, |m| m.role == "assistant") {
                session.messages.pop();
            }
            // Also pop the user message — build_chat_context will re-add it
            if session.messages.last().map_or(false, |m| m.role == "user") {
                session.messages.pop();
            }
            let updated = session.clone();
            let _ = sessions.update(&updated);
        }
    }

    // Find the last user message to re-send
    let last_user_content = {
        let sessions = state.sessions.read().await;
        sessions.get(session_id)
            .and_then(|s| s.messages.iter().rev().find(|m| m.role == "user"))
            .map(|m| m.text_content().to_string())
            .unwrap_or_default()
    };

    if last_user_content.is_empty() {
        send_ws(sender, "error", &serde_json::json!({"message": "No user message to regenerate from"})).await;
        return;
    }

    // Re-dispatch as a chat message (reuses the full inference pipeline)
    let chat_msg = serde_json::json!({
        "type": "chat",
        "content": last_user_content,
        "session_id": session_id,
    });
    handle_chat_message(&chat_msg, state, sender, pending_chain, stop_flag).await;
}

/// Handle an edit-and-resend — truncate session to message index with new content, re-run.
async fn handle_edit_and_resend(
    msg: &serde_json::Value,
    state: &AppState,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
    pending_chain: &mut Option<PendingToolChain>,
    stop_flag: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    let session_id = msg["session_id"].as_str().unwrap_or("");
    let new_content = msg["content"].as_str().unwrap_or("");
    let message_index = msg["message_index"].as_u64().unwrap_or(0) as usize;

    if session_id.is_empty() || new_content.is_empty() {
        send_ws(sender, "error", &serde_json::json!({"message": "Missing session_id or content"})).await;
        return;
    }

    // Truncate session to the edited message and replace content
    {
        let mut sessions = state.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.messages.truncate(message_index);
            // The new user message will be added by handle_chat_message
            let updated = session.clone();
            let _ = sessions.update(&updated);
        }
    }

    let chat_msg = serde_json::json!({
        "type": "chat",
        "content": new_content,
        "session_id": session_id,
    });
    handle_chat_message(&chat_msg, state, sender, pending_chain, stop_flag).await;
}

/// Handle a plan decision — approve, revise, or cancel.
async fn handle_plan_decision(
    msg: &serde_json::Value,
    state: &AppState,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
    pending_chain: &mut Option<PendingToolChain>,
    stop_flag: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    let session_id = msg["session_id"].as_str().unwrap_or("");
    let approved = msg["approved"].as_bool().unwrap_or(false);
    let notes = msg["notes"].as_str().unwrap_or("");

    if session_id.is_empty() {
        send_ws(sender, "error", &serde_json::json!({"message": "Missing session_id"})).await;
        return;
    }

    let plan = match crate::web::ws_plans::load_pending_plan(session_id) {
        Some(p) => p,
        None => {
            send_ws(sender, "error", &serde_json::json!({"message": "No pending plan found for this session"})).await;
            return;
        }
    };

    if approved {
        tracing::info!(session = %session_id, title = %plan.title, turns = plan.estimated_turns, "Plan APPROVED — entering ReAct loop");
        crate::web::ws_plans::delete_pending_plan(session_id);

        // Build context with the plan as the objective
        let plan_tools = schema::layer1_tools();
        let plan_tools_chars = plan_tools.to_string().len();
        let ctx = crate::web::ws_context::build_chat_context(
            state, &plan.title, session_id, None, Vec::new(), "web", plan_tools_chars,
        ).await;

        stop_flag.store(false, std::sync::atomic::Ordering::Relaxed);
        send_ws(sender, "status", &serde_json::json!({
            "message": format!("Executing approved plan: {} ({} turns)", plan.title, plan.estimated_turns)
        })).await;

        let provider = state.provider.as_ref();
        run_react_loop(
            state, provider, ctx.messages,
            &plan.title, Some(&plan.plan_markdown),
            plan.estimated_turns, &plan.title, session_id,
            sender, stop_flag,
        ).await;
    } else if !notes.is_empty() {
        tracing::info!(session = %session_id, notes_len = notes.len(), "Plan REVISION requested");

        // Feed the notes back to the model as a user message requesting plan revision
        let revision_prompt = format!(
            "The user has reviewed your implementation plan \"{}\" and provided the following feedback:\n\n{}\n\nPlease revise the plan accordingly and call `propose_plan` again with the updated plan.",
            plan.title, notes
        );

        let chat_msg = serde_json::json!({
            "type": "chat",
            "content": revision_prompt,
            "session_id": session_id,
        });
        handle_chat_message(&chat_msg, state, sender, pending_chain, stop_flag).await;
    } else {
        tracing::info!(session = %session_id, "Plan CANCELLED");
        crate::web::ws_plans::delete_pending_plan(session_id);
        send_ws(sender, "text_delta", &serde_json::json!({"content": "Plan cancelled. What would you like to do instead?"})).await;
        ingest_assistant_turn(state, "Plan cancelled. What would you like to do instead?", session_id).await;
        send_ws(sender, "done", &serde_json::json!({})).await;
    }
}

// L1 tool chain handler extracted to ws_l1.rs for governance compliance.
use crate::web::ws_l1::{deliver_reply, run_l1_tool_chain};


// ReAct loop execution is in crate::web::ws_react.
use crate::web::ws_react::run_react_loop;

// Learning functions (ingest_assistant_turn, spawn_insight_extraction,
// spawn_delayed_reinforcement, classify_user_feedback, derive_procedure_name)
// are in crate::web::ws_learning.
use ws_learning::{ingest_assistant_turn, spawn_delayed_reinforcement};

use crate::web::ws_stream::send_ws;

#[cfg(test)]
mod tests {
    #[test]
    fn test_ws_module_compiles() {
        assert!(true);
    }
}
