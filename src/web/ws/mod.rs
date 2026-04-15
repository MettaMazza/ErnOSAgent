// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! WebSocket handler — real-time chat streaming.
//!
//! Split into submodules:
//! - `chat`: core inference path (handle_chat, build_context, stream events)
//! - `sessions`: session switch/create handlers

pub(crate) mod chat;
pub(crate) mod pipeline;
mod sessions;

use crate::web::state::SharedState;
use axum::extract::ws::{self, WebSocket};
use axum::extract::{State, WebSocketUpgrade};
use axum::response::IntoResponse;
use serde::{Deserialize, Serialize};
use std::sync::atomic::Ordering;

/// Upgrade HTTP to WebSocket.
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<SharedState>,
) -> impl IntoResponse {
    tracing::info!("WebSocket connection upgrading");
    ws.on_upgrade(move |socket| handle_socket(socket, state))
}

// ── Wire protocol ────────────────────────────────────────────────────

#[derive(Deserialize)]
#[serde(tag = "type")]
pub(crate) enum ClientMessage {
    #[serde(rename = "chat")]
    Chat {
        message: String,
        #[serde(default)]
        images: Vec<String>,
    },
    #[serde(rename = "cancel")]
    Cancel,
    #[serde(rename = "regenerate")]
    Regenerate,
    #[serde(rename = "switch_session")]
    SwitchSession { session_id: String },
    #[serde(rename = "new_session")]
    NewSession,
}

#[derive(Serialize)]
#[serde(tag = "type")]
pub(crate) enum ServerMessage {
    #[serde(rename = "token")]
    Token { content: String },
    #[serde(rename = "thinking")]
    Thinking { content: String },
    #[serde(rename = "tool_call")]
    ToolCall { name: String, arguments: String },
    #[serde(rename = "tool_result")]
    ToolResult { name: String, output: String, success: bool },
    #[serde(rename = "audit")]
    Audit { verdict: String, confidence: f32 },
    #[serde(rename = "react_turn")]
    ReactTurn { turn: usize },
    #[serde(rename = "done")]
    Done { response: String, context_usage: f32, #[serde(skip_serializing_if = "Vec::is_empty")] images: Vec<String> },
    #[serde(rename = "cancelled")]
    Cancelled,
    #[serde(rename = "error")]
    Error { message: String },
    #[serde(rename = "neural_snapshot")]
    NeuralSnapshot { snapshot: serde_json::Value },
    #[serde(rename = "session_loaded")]
    SessionLoaded { session_id: String, title: String, messages: Vec<MessageDto> },
    #[serde(rename = "status")]
    Status { model: String, context_usage: f32, memory: String, is_generating: bool },
}

#[derive(Serialize)]
pub(crate) struct MessageDto {
    role: String,
    content: String,
}

// ── Socket handler ───────────────────────────────────────────────────

async fn handle_socket(mut socket: WebSocket, state: SharedState) {
    tracing::info!("WebSocket connected");

    if let Ok(msg) = build_status_message(&state).await {
        let _ = send_json(&mut socket, &msg).await;
    }

    {
        let st = state.read().await;
        let session = st.session_mgr.active();
        let loaded = ServerMessage::SessionLoaded {
            session_id: session.id.clone(),
            title: session.title.clone(),
            messages: session.messages.iter().map(|m| MessageDto {
                role: m.role.clone(),
                content: m.content.clone(),
            }).collect(),
        };
        let _ = send_json(&mut socket, &loaded).await;
    }

    // Check for post-recompile resume state
    if let Some(resume_prompt) = consume_resume_state() {
        tracing::info!("Post-recompile resume detected — triggering system notification");
        chat::handle_chat(&mut socket, &state, &resume_prompt, Vec::new()).await;
    }

    while let Some(msg) = socket.recv().await {
        let msg = match msg {
            Ok(ws::Message::Text(text)) => text,
            Ok(ws::Message::Close(_)) => { tracing::info!("WebSocket closed by client"); break; }
            Ok(_) => continue,
            Err(e) => { tracing::warn!(error = %e, "WebSocket receive error"); break; }
        };

        let client_msg: ClientMessage = match serde_json::from_str(&msg) {
            Ok(m) => m,
            Err(e) => {
                let _ = send_json(&mut socket, &ServerMessage::Error {
                    message: format!("Invalid message: {}", e),
                }).await;
                continue;
            }
        };

        match client_msg {
            ClientMessage::Chat { message, images } => {
                // Signal autonomy cancellation — user message preempts running autonomy jobs
                // NOTE: idle timer is NOT reset here — it resets when the turn ENDS
                {
                    let st = state.read().await;
                    st.autonomy_cancel.store(true, Ordering::SeqCst);
                }
                chat::handle_chat(&mut socket, &state, &message, images).await
            }
            ClientMessage::Cancel => {
                tracing::info!("Cancel requested — signalling abort");
                let st = state.read().await;
                st.cancel_token.store(true, Ordering::SeqCst);
            }
            ClientMessage::Regenerate => {
                // Pop the last assistant response and re-run with a fresh perspective
                let last_user_msg = {
                    let mut st = state.write().await;
                    let session = st.session_mgr.active_mut();
                    // Remove trailing assistant/tool messages
                    while session.messages.last().map(|m| m.role.as_str()) == Some("assistant")
                        || session.messages.last().map(|m| m.role.as_str()) == Some("tool")
                    {
                        session.messages.pop();
                    }
                    // Grab the user message text, then pop it too (handle_chat will re-add it)
                    let user_text = session.messages.iter().rev()
                        .find(|m| m.role == "user")
                        .map(|m| m.content.clone());
                    if user_text.is_some() {
                        // Pop the user message so we can re-inject it after the regen hint
                        if session.messages.last().map(|m| m.role.as_str()) == Some("user") {
                            session.messages.pop();
                        }
                        // Inject a regen hint so the model knows to take a different angle
                        session.messages.push(crate::provider::Message {
                            role: "system".to_string(),
                            content: "[REGENERATION REQUESTED] The user was not satisfied with your \
                                previous response and has asked you to try again. Take a fresh \
                                perspective — vary your reasoning, structure, tone, and approach. \
                                Do not repeat or closely mirror your previous attempt. Be more \
                                creative, more thorough, or more concise as the situation demands."
                                .to_string(),
                            images: Vec::new(),
                        });
                    }
                    user_text
                };
                if let Some(user_msg) = last_user_msg {
                    tracing::info!(prompt_len = user_msg.len(), "Regenerate — re-running with regen hint");
                    chat::handle_chat(&mut socket, &state, &user_msg, Vec::new()).await;
                } else {
                    let _ = send_json(&mut socket, &ServerMessage::Error {
                        message: "No user message to regenerate from".to_string(),
                    }).await;
                }
            }
            ClientMessage::SwitchSession { session_id } => {
                sessions::handle_switch_session(&mut socket, &state, &session_id).await;
            }
            ClientMessage::NewSession => sessions::handle_new_session(&mut socket, &state).await,
        }
    }

    tracing::info!("WebSocket disconnected");
}

// ── Helpers ──────────────────────────────────────────────────────────

pub(crate) async fn send_json(socket: &mut WebSocket, msg: &ServerMessage) -> Result<(), ()> {
    match serde_json::to_string(msg) {
        Ok(json) => socket.send(ws::Message::Text(json.into())).await.map_err(|_| ()),
        Err(_) => Err(()),
    }
}

async fn build_status_message(state: &SharedState) -> Result<ServerMessage, ()> {
    let st = state.read().await;
    let usage = crate::inference::context::context_usage(
        &st.session_mgr.active().messages,
        st.model_spec.context_length,
    );
    let memory = st.memory_mgr.status_summary().await;

    Ok(ServerMessage::Status {
        model: st.model_spec.name.clone(),
        context_usage: usage,
        memory,
        is_generating: st.is_generating,
    })
}

/// Check for a post-recompile resume state file.
/// If found, reads it, deletes it (one-shot), and returns a contextual prompt
/// for the agent to generate its own post-upgrade notification.
fn consume_resume_state() -> Option<String> {
    let resume_path = std::path::Path::new("memory/core/resume.json");
    if !resume_path.exists() {
        return None;
    }

    let content = std::fs::read_to_string(resume_path).ok()?;
    // Consume the file so it doesn't fire again
    let _ = std::fs::remove_file(resume_path);

    let parsed: serde_json::Value = serde_json::from_str(&content).ok()?;
    let compiled_at = parsed.get("compiled_at")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");

    // Read the recompile changelog for context
    let changelog = std::fs::read_to_string("memory/core/recompile_log.md")
        .ok()
        .and_then(|log| {
            // Take only the last entry (most recent recompile)
            log.rsplit("## Recompile —")
                .next()
                .map(|entry| entry.chars().take(500).collect::<String>())
        })
        .unwrap_or_default();

    tracing::info!(
        compiled_at = compiled_at,
        "Consumed resume state — generating post-upgrade notification"
    );

    Some(format!(
        "[SYSTEM: You have just been recompiled and restarted at {}. \
        Greet the user and let them know the upgrade was successful. \
        Be brief and natural — do not be robotic. \
        If there is changelog context below, mention what changed at a high level.\n\
        Changelog context: {}]",
        compiled_at,
        if changelog.is_empty() { "No changelog available.".to_string() } else { changelog }
    ))
}
