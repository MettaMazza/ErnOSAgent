// Ern-OS — Layer 1 tool chain handler (extracted from ws.rs for governance compliance).

use crate::provider::Message;
use crate::tools::schema;
use crate::web::state::AppState;
use crate::web::ws_learning::PendingToolChain;
use crate::web::ws_stream::{ConsumeResult, consume_silently, audit_and_retry, send_ws};
use crate::web::ws_learning::{ingest_assistant_turn, spawn_insight_extraction};
use crate::web::ws_react::run_react_loop;
use axum::extract::ws::{Message as WsMessage, WebSocket};

/// Deliver a reply: audit, stream, ingest, extract insights.
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
    let approved_text = audit_and_retry(state, provider, sender, messages, tools, user_query, text).await;
    send_ws(sender, "text_delta", &serde_json::json!({"content": &approved_text})).await;
    ingest_assistant_turn(state, &approved_text, session_id).await;
    spawn_insight_extraction(state, user_query, &approved_text);
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

        let result = crate::web::tool_dispatch::execute_tool_with_state(state, &current_tc).await;
        tracing::info!(tool = %current_tc.name, success = result.success, output_len = result.output.len(), "L1 tool: execution complete");

        send_ws(sender, "tool_completed", &serde_json::json!({
            "id": &current_tc.id, "name": &current_tc.name,
            "result": &result.output, "success": result.success,
        })).await;

        messages.push(Message::assistant_tool_call(&current_tc.id, &current_tc.name, &current_tc.arguments));
        messages.push(Message::tool_result(&current_tc.id, &result.output));

        let rx_next = match provider.chat(messages, Some(tools), true).await {
            Ok(rx) => rx,
            Err(e) => { tracing::error!(error = %e, "L1 re-inference FAILED"); break; }
        };

        match consume_silently(rx_next, sender).await {
            ConsumeResult::Reply { text, thinking } => {
                deliver_reply(state, provider, sender, messages, tools, content, session_id, &text, &thinking).await;
                chain_reply = text;
                stash_chain(pending_chain, chain_tools, content, &chain_reply, session_id);
                return;
            }
            ConsumeResult::ToolCall { id, name, arguments } => {
                tracing::info!(tool = %name, id = %id, iteration = tool_iter + 1, "L1 chain → another ToolCall");
                current_tc = schema::ToolCall { id, name, arguments };
            }
            ConsumeResult::Escalate { objective, plan, planned_turns } => {
                tracing::info!(objective = %objective, planned_turns, "L1 chain → Escalate");
                stop_flag.store(false, std::sync::atomic::Ordering::Relaxed);
                send_ws(sender, "status", &serde_json::json!({"message": format!("ReAct loop activated ({} turns planned)", planned_turns)})).await;
                run_react_loop(state, provider, messages.clone(), &objective, plan.as_deref(), planned_turns, content, session_id, sender, stop_flag).await;
                return;
            }
            ConsumeResult::PlanProposal { title, plan_markdown, estimated_turns } => {
                tracing::info!(title = %title, turns = estimated_turns, "L1 chain → PlanProposal");
                let plan = crate::web::ws_plans::save_pending_plan(session_id, &title, &plan_markdown, estimated_turns);
                send_ws(sender, "plan_proposal", &serde_json::json!({
                    "title": plan.title,
                    "plan_markdown": plan.plan_markdown,
                    "estimated_turns": plan.estimated_turns,
                    "revision": plan.revision,
                    "session_id": session_id,
                })).await;
                send_ws(sender, "done", &serde_json::json!({})).await;
                return;
            }
            ConsumeResult::Error(e) => {
                tracing::error!(error = %e, "L1 chain → Error");
                send_ws(sender, "error", &serde_json::json!({"message": e})).await;
                break;
            }
        }
    }

    stash_chain(pending_chain, chain_tools, content, &chain_reply, session_id);
    send_ws(sender, "done", &serde_json::json!({})).await;
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
