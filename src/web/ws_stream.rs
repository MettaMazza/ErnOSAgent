//! WebSocket stream consumption and observer audit utilities.

use crate::observer;
use crate::provider::{Message, StreamEvent};
use crate::web::state::AppState;
use crate::web::training_capture;
use axum::extract::ws::{Message as WsMessage, WebSocket};
use futures_util::SinkExt;

/// Result of consuming a provider stream.
pub enum ConsumeResult {
    Reply { text: String, thinking: Option<String> },
    Escalate { objective: String, plan: Option<String>, planned_turns: usize },
    PlanProposal { title: String, plan_markdown: String, estimated_turns: usize },
    ToolCall { id: String, name: String, arguments: String },
    Error(String),
}

/// Consume provider stream silently — buffer text, only forward thinking.
/// Text is held back until Observer approval.
pub async fn consume_silently(
    mut rx: tokio::sync::mpsc::Receiver<StreamEvent>,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
) -> ConsumeResult {
    let mut text = String::new();
    let mut thinking = String::new();
    let mut tool_calls: Vec<(String, String, String)> = Vec::new();
    let mut event_count: u64 = 0;
    let start = std::time::Instant::now();

    while let Some(event) = rx.recv().await {
        event_count += 1;
        match &event {
            StreamEvent::TextDelta(delta) => {
                text.push_str(delta);
            }
            StreamEvent::ThinkingDelta(delta) => {
                thinking.push_str(delta);
                send_ws(sender, "thinking_delta", &serde_json::json!({"content": delta})).await;
            }
            StreamEvent::ToolCall { id, name, arguments } => {
                tracing::debug!(tool = %name, id = %id, "Stream: ToolCall received");
                tool_calls.push((id.clone(), name.clone(), arguments.clone()));
            }
            StreamEvent::Done => {
                tracing::debug!(events = event_count, elapsed_ms = start.elapsed().as_millis() as u64, "Stream: Done");
                break;
            }
            StreamEvent::Error(e) => {
                tracing::error!(error = %e, events = event_count, "Stream: Error");
                return ConsumeResult::Error(e.clone());
            }
        }
    }

    if event_count == 0 || (!text.is_empty() && tool_calls.is_empty()) {
        tracing::debug!(
            text_len = text.len(), thinking_len = thinking.len(),
            tool_calls = tool_calls.len(), events = event_count,
            "consume_silently: stream ended"
        );
    }

    // Check for propose_plan first (user-approved planning takes priority)
    if let Some((_, _, args)) = tool_calls.iter().find(|(_, name, _)| name == "propose_plan") {
        let parsed: serde_json::Value = serde_json::from_str(args).unwrap_or_default();
        let turns = parsed["estimated_turns"].as_u64().unwrap_or(10) as usize;
        tracing::info!(turns, "consume_silently result: PlanProposal");
        return ConsumeResult::PlanProposal {
            title: parsed["title"].as_str().unwrap_or("Plan").to_string(),
            plan_markdown: parsed["plan_markdown"].as_str().unwrap_or("").to_string(),
            estimated_turns: turns.max(3).min(50),
        };
    }

    if let Some((_, _, args)) = tool_calls.iter().find(|(_, name, _)| name == "start_react_system") {
        let parsed: serde_json::Value = serde_json::from_str(args).unwrap_or_default();
        let planned = parsed["planned_turns"].as_u64().unwrap_or(10) as usize;
        tracing::info!(planned_turns = planned, "consume_silently result: Escalate");
        return ConsumeResult::Escalate {
            objective: parsed["objective"].as_str().unwrap_or("").to_string(),
            plan: parsed["plan"].as_str().map(|s| s.to_string()),
            planned_turns: planned.max(3).min(50),
        };
    }

    if let Some((id, name, arguments)) = tool_calls.into_iter().next() {
        tracing::info!(tool = %name, id = %id, "consume_silently result: ToolCall");
        return ConsumeResult::ToolCall { id, name, arguments };
    }

    tracing::info!(text_len = text.len(), "consume_silently result: Reply");
    ConsumeResult::Reply {
        text,
        thinking: if thinking.is_empty() { None } else { Some(thinking) },
    }
}

/// Observer-gated audit loop: audit response, retry silently if rejected.
/// Returns the approved text (or last attempt after max retries / bailout).
///
/// Uses the new 19-rule observer with structured rejection feedback and
/// bailout override to prevent infinite rejection loops.
pub async fn audit_and_retry(
    state: &AppState,
    provider: &dyn crate::provider::Provider,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
    messages: &mut Vec<Message>,
    tools: &serde_json::Value,
    user_query: &str,
    initial_text: &str,
) -> String {
    let max_retries = 2;  // Bailout after 2 consecutive rejections
    let mut current_text = initial_text.to_string();

    if !state.config.observer.enabled || current_text.is_empty() {
        return current_text;
    }

    for attempt in 0..=max_retries {
        send_ws(sender, "audit_running", &serde_json::json!({})).await;

        match observer::audit_response(provider, messages, &current_text, &build_l1_tool_context(messages), user_query).await {
            Ok(output) if output.result.verdict.is_allowed() => {
                send_ws(sender, "audit_completed", &serde_json::json!({
                    "approved": true,
                    "confidence": output.result.confidence,
                    "category": &output.result.failure_category,
                })).await;
                training_capture::capture_approved(state, user_query, &current_text, output.result.confidence);
                return current_text;
            }
            Ok(output) => {
                let rejected_text = current_text.clone();
                let reject_reason = output.result.what_went_wrong.clone();
                let category = output.result.failure_category.clone();

                tracing::info!(
                    attempt,
                    category = %category,
                    reason = %reject_reason,
                    "Response rejected by observer — retrying"
                );

                send_ws(sender, "audit_completed", &serde_json::json!({
                    "approved": false,
                    "category": &category,
                    "reason": &reject_reason,
                })).await;

                // On last attempt, use bailout override instead of retrying
                if attempt >= max_retries {
                    tracing::warn!(
                        rejections = attempt + 1,
                        "Observer bailout — forcing response through"
                    );
                    let bailout = observer::format_bailout_override(attempt + 1);
                    messages.push(Message::text("assistant", &rejected_text));
                    messages.push(Message::text("system", &bailout));

                    if let Ok(rx) = provider.chat(messages, Some(tools), true).await {
                        let result = consume_silently(rx, sender).await;
                        if let ConsumeResult::Reply { text, thinking: _ } = result {
                            current_text = text;
                        }
                    }
                    break;
                }

                // Normal retry with SELF-CHECK feedback
                let feedback = observer::format_rejection_feedback(&output.result);
                messages.push(Message::text("assistant", &rejected_text));
                messages.push(Message::text("system", &feedback));

                if let Ok(rx) = provider.chat(messages, Some(tools), true).await {
                    let result = consume_silently(rx, sender).await;
                    if let ConsumeResult::Reply { text, thinking: _ } = result {
                        training_capture::capture_rejection(
                            state, user_query, &rejected_text, &text, &reject_reason,
                        );
                        current_text = text;
                    }
                }
            }
            Err(e) => {
                tracing::warn!(error = %e, "Observer failed — fail-open");
                return current_text;
            }
        }
    }

    current_text
}

/// Send a typed JSON message to the WebSocket client.
pub async fn send_ws(
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
    msg_type: &str,
    payload: &serde_json::Value,
) {
    let mut msg = payload.clone();
    msg["type"] = serde_json::json!(msg_type);
    let _ = sender.send(WsMessage::Text(msg.to_string().into())).await;
}

/// Build tool context from L1 message history for observer audit.
fn build_l1_tool_context(messages: &[Message]) -> String {
    let mut entries: Vec<String> = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        if msg.role == "tool" {
            let tool_call_id = msg.tool_call_id.as_deref().unwrap_or("");
            let result_text = msg.text_content();
            let preview = if result_text.len() > 200 {
                format!("{}...", &result_text[..200])
            } else {
                result_text.clone()
            };

            let mut tool_name = "unknown".to_string();
            for j in (0..i).rev() {
                if messages[j].role == "assistant" {
                    if let Some(tcs) = &messages[j].tool_calls {
                        for tc in tcs {
                            if tc["id"].as_str() == Some(tool_call_id) {
                                tool_name = tc["function"]["name"]
                                    .as_str()
                                    .unwrap_or("unknown")
                                    .to_string();
                                break;
                            }
                        }
                        if tool_name != "unknown" { break; }
                    }
                }
            }
            entries.push(format!("[{}] {} → {}", entries.len() + 1, tool_name, preview));
        }
    }
    if entries.is_empty() {
        String::new()
    } else {
        format!("L1 tools executed ({} calls):\n{}", entries.len(), entries.join("\n"))
    }
}
