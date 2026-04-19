//! Video call handler — real-time video communication via WebSocket.
//! Pipeline: Browser Camera → WebSocket (JPEG frames + audio) → Gemma 4 (vision + audio) → TTS → Speaker
//!
//! Protocol:
//! Client → Server: JSON {"type": "video_frame", "frame": "<base64 JPEG>", "audio": "<base64 WAV>"}
//! Server → Client: JSON {"type": "response", "text": "..."} + Binary (TTS WAV)

use crate::provider::Message;
use crate::web::state::AppState;
use axum::{
    extract::{State, WebSocketUpgrade, ws::{Message as WsMessage, WebSocket}},
    response::IntoResponse,
};
use futures_util::{StreamExt, SinkExt};
use std::time::Instant;

const MIN_FRAME_INTERVAL_MS: u128 = 1000; // 1 fps max for analysis

/// Video WebSocket upgrade handler.
pub async fn ws_video_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_video(socket, state))
}

/// Handle a video call WebSocket session.
async fn handle_video(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();

    tracing::info!("Video call connected");

    let welcome = serde_json::json!({
        "type": "video_connected",
        "model": state.model_spec.name,
        "supports_vision": state.model_spec.supports_vision,
    });
    if sender.send(WsMessage::Text(welcome.to_string().into())).await.is_err() {
        return;
    }

    let session_id = format!("video_{}", uuid::Uuid::new_v4());
    let mut conversation: Vec<Message> = vec![
        Message::text("system", "You are in a live video call. You can see the user's camera feed and hear their voice. Respond naturally and conversationally about what you see and hear. Keep responses concise."),
    ];

    let mut last_process = Instant::now();

    while let Some(msg_result) = receiver.next().await {
        let msg = match msg_result {
            Ok(m) => m,
            Err(_) => break,
        };

        match msg {
            WsMessage::Text(text) => {
                if let Ok(data) = serde_json::from_str::<serde_json::Value>(&text) {
                    match data["type"].as_str() {
                        Some("video_frame") => {
                            // Rate limit
                            if last_process.elapsed().as_millis() < MIN_FRAME_INTERVAL_MS {
                                continue;
                            }
                            last_process = Instant::now();

                            let frame_b64 = data["frame"].as_str().unwrap_or("");
                            let audio_b64 = data["audio"].as_str();

                            let result = process_video_turn(
                                &state, &mut conversation,
                                frame_b64, audio_b64, &session_id,
                            ).await;

                            match result {
                                Ok((response, tts_audio)) => {
                                    let resp = serde_json::json!({
                                        "type": "response",
                                        "text": response,
                                    });
                                    let _ = sender.send(WsMessage::Text(resp.to_string().into())).await;

                                    if let Some(audio) = tts_audio {
                                        let _ = sender.send(WsMessage::Binary(audio.into())).await;
                                    }
                                }
                                Err(e) => {
                                    let err = serde_json::json!({
                                        "type": "video_error",
                                        "error": e.to_string(),
                                    });
                                    let _ = sender.send(WsMessage::Text(err.to_string().into())).await;
                                }
                            }
                        }
                        Some("video_end") => {
                            tracing::info!(session = %session_id, "Video call ended by client");
                            break;
                        }
                        _ => {}
                    }
                }
            }
            WsMessage::Close(_) => break,
            _ => {}
        }
    }

    // Ingest into memory
    let turns = conversation.len().saturating_sub(1);
    if turns > 0 {
        let mut mem = state.memory.write().await;
        mem.ingest_turn("system", &format!("[Video call: {} turns]", turns), &session_id);
    }

    tracing::info!(session = %session_id, "Video call disconnected");
}

/// Process a single video turn: frame + optional audio → inference → TTS.
async fn process_video_turn(
    state: &AppState,
    conversation: &mut Vec<Message>,
    frame_b64: &str,
    audio_b64: Option<&str>,
    _session_id: &str,
) -> anyhow::Result<(String, Option<Vec<u8>>)> {
    // Build multimodal message with frame and optional audio
    let user_msg = if let Some(audio) = audio_b64 {
        Message::multimodal_av("user", audio, frame_b64)
    } else {
        // Vision-only frame
        Message::multipart("user", "[video frame]", vec![
            format!("data:image/jpeg;base64,{}", frame_b64),
        ])
    };

    conversation.push(user_msg);

    // Keep conversation short for real-time performance (last 10 messages + system)
    if conversation.len() > 12 {
        let system = conversation[0].clone();
        let recent: Vec<_> = conversation[conversation.len()-10..].to_vec();
        *conversation = std::iter::once(system).chain(recent).collect();
    }

    // Run inference
    let response_text = state.provider.chat_sync(conversation, None).await?;
    conversation.push(Message::text("assistant", &response_text));

    // Generate TTS
    let tts_audio = super::voice::generate_tts_from_state(state, &response_text).await.ok();

    Ok((response_text, tts_audio))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_interval_constant() {
        assert!(MIN_FRAME_INTERVAL_MS >= 200);
    }

    #[test]
    fn test_session_id_format() {
        let id = format!("video_{}", uuid::Uuid::new_v4());
        assert!(id.starts_with("video_"));
    }
}
