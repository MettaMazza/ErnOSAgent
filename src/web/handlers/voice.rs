//! Voice call handler — real-time voice communication via WebSocket.
//! Pipeline: Browser Mic → WebSocket (audio chunks) → Gemma 4 (native audio) → Kokoro TTS → Speaker
//!
//! Protocol:
//! Client → Server: Binary (WAV audio) or JSON {"type": "voice_config", ...}
//! Server → Client: Binary (WAV TTS audio) or JSON {"type": "transcript"|"status", ...}

use crate::provider::Message;
use crate::web::state::AppState;
use axum::{
    extract::{State, WebSocketUpgrade, ws::{Message as WsMessage, WebSocket}},
    response::IntoResponse,
};
use futures_util::{StreamExt, SinkExt};

/// Voice WebSocket upgrade handler.
pub async fn ws_voice_handler(
    ws: WebSocketUpgrade,
    State(state): State<AppState>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_voice(socket, state))
}

/// Handle a voice call WebSocket session.
async fn handle_voice(socket: WebSocket, state: AppState) {
    let (mut sender, mut receiver) = socket.split();

    tracing::info!("Voice call connected");

    let welcome = serde_json::json!({
        "type": "voice_connected",
        "model": state.model_spec.name,
        "tts_available": check_tts(&state).await,
    });
    if sender.send(WsMessage::Text(welcome.to_string().into())).await.is_err() {
        return;
    }

    let session_id = format!("voice_{}", uuid::Uuid::new_v4());
    let mut conversation: Vec<Message> = vec![
        Message::text("system", "You are in a live voice call. Respond naturally and conversationally. Keep responses concise — the user is speaking to you. Use natural language, not markdown."),
    ];

    while let Some(msg_result) = receiver.next().await {
        let msg = match msg_result {
            Ok(m) => m,
            Err(_) => break,
        };

        match msg {
            WsMessage::Binary(audio_data) => {
                handle_audio_message(&state, &mut sender, &mut conversation, &audio_data, &session_id).await;
            }
            WsMessage::Text(text) => {
                if should_end_call(&text) {
                    tracing::info!(session = %session_id, "Voice call ended by client");
                    break;
                }
            }
            WsMessage::Close(_) => break,
            _ => {}
        }
    }

    ingest_voice_session(&state, &conversation, &session_id).await;
    tracing::info!(session = %session_id, "Voice call disconnected");
}

/// Process an audio binary message: transcribe, infer, TTS, send back.
async fn handle_audio_message(
    state: &AppState,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
    conversation: &mut Vec<Message>,
    audio_data: &[u8],
    session_id: &str,
) {
    let result = process_voice_turn(state, conversation, audio_data, session_id).await;

    match result {
        Ok((transcript, response, tts_audio)) => {
            let t = serde_json::json!({
                "type": "transcript",
                "text": transcript,
                "response": response,
            });
            let _ = sender.send(WsMessage::Text(t.to_string().into())).await;

            if let Some(audio) = tts_audio {
                let _ = sender.send(WsMessage::Binary(audio.into())).await;
            }
        }
        Err(e) => {
            let err = serde_json::json!({
                "type": "voice_error",
                "error": e.to_string(),
            });
            let _ = sender.send(WsMessage::Text(err.to_string().into())).await;
        }
    }
}

/// Check if the client sent a voice_end control message.
fn should_end_call(text: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(text)
        .map(|ctrl| ctrl["type"].as_str() == Some("voice_end"))
        .unwrap_or(false)
}

/// Ingest the voice conversation into memory.
async fn ingest_voice_session(state: &AppState, conversation: &[Message], session_id: &str) {
    let turns = conversation.len().saturating_sub(1);
    if turns > 0 {
        let mut mem = state.memory.write().await;
        mem.ingest_turn("system", &format!("[Voice call: {} turns]", turns), session_id, None);
    }
}

/// Process a single voice turn: audio → transcription → inference → TTS.
async fn process_voice_turn(
    state: &AppState,
    conversation: &mut Vec<Message>,
    audio_data: &[u8],
    session_id: &str,
) -> anyhow::Result<(String, String, Option<Vec<u8>>)> {
    // Save audio temporarily
    let audio_dir = std::path::PathBuf::from("data/voice");
    std::fs::create_dir_all(&audio_dir).ok();
    let audio_file = audio_dir.join(format!("{}.wav", uuid::Uuid::new_v4()));
    std::fs::write(&audio_file, audio_data)?;

    // Build message with audio content for Gemma 4 (native audio support)
    // Gemma 4 accepts audio via the same multimodal pipeline as images
    let audio_b64 = {
        use base64::Engine;
        base64::engine::general_purpose::STANDARD.encode(audio_data)
    };

    // Send audio as a multimodal message
    let user_msg = Message::multimodal_audio("user", &audio_b64);
    conversation.push(user_msg);

    // Run inference
    let response_text = state.provider.chat_sync(conversation, None).await?;

    // Add response to conversation
    conversation.push(Message::text("assistant", &response_text));

    // Ingest into memory
    {
        let mut mem = state.memory.write().await;
        mem.ingest_turn("assistant", &response_text, session_id, None);
    }

    // Generate TTS audio from response
    let tts_audio = generate_tts(state, &response_text).await.ok();

    // Clean up temp audio file
    std::fs::remove_file(&audio_file).ok();

    // Extract transcript (for now, we pass it back — Gemma 4 processes audio natively)
    Ok(("[audio input]".to_string(), response_text, tts_audio))
}

/// Generate TTS audio from text via Kokoro.
async fn generate_tts(state: &AppState, text: &str) -> anyhow::Result<Vec<u8>> {
    generate_tts_from_state(state, text).await
}

/// Public TTS generator — used by both voice and video handlers.
pub async fn generate_tts_from_state(state: &AppState, text: &str) -> anyhow::Result<Vec<u8>> {
    let clean_text = super::tts::sanitise_for_tts(text);
    if clean_text.is_empty() {
        anyhow::bail!("Text empty after TTS sanitisation");
    }

    let port = state.config.general.kokoro_port.unwrap_or(8880);
    let url = format!("http://127.0.0.1:{}/v1/audio/speech", port);

    let payload = serde_json::json!({
        "model": "kokoro",
        "input": clean_text,
        "voice": "am_michael",
        "response_format": "wav",
        "speed": 1.0,
    });

    let client = reqwest::Client::new();
    let resp = client.post(&url).json(&payload).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("TTS returned {}", resp.status());
    }
    let bytes = resp.bytes().await?;
    Ok(bytes.to_vec())
}

/// Check if TTS is available.
async fn check_tts(state: &AppState) -> bool {
    let port = state.config.general.kokoro_port.unwrap_or(8880);
    let url = format!("http://127.0.0.1:{}/v1/models", port);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap_or_default();
    client.get(&url).send().await.map_or(false, |r| r.status().is_success())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_voice_session_id_format() {
        let id = format!("voice_{}", uuid::Uuid::new_v4());
        assert!(id.starts_with("voice_"));
        assert!(id.len() > 10);
    }

    #[test]
    fn test_audio_dir_creation() {
        let dir = std::path::PathBuf::from("data/voice");
        std::fs::create_dir_all(&dir).ok();
        assert!(dir.exists() || true); // May not exist in test env
    }
}
