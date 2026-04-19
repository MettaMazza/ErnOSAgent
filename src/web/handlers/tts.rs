//! TTS proxy handler — proxies requests to local Kokoro ONNX TTS server.
//! Endpoint: POST /api/tts
//! The Kokoro server runs on localhost:8880 (OpenAI-compatible /v1/audio/speech).

use crate::web::state::AppState;
use axum::{extract::State, response::IntoResponse, Json};

/// POST /api/tts — Generate speech from text via local Kokoro server.
pub async fn synthesize(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let text = body["text"].as_str().unwrap_or("");
    let voice = body["voice"].as_str().unwrap_or("am_michael");
    let speed = body["speed"].as_f64().unwrap_or(1.0);

    if text.is_empty() {
        return axum::response::Response::builder()
            .status(400)
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                serde_json::json!({"error": "Missing text"}).to_string(),
            ))
            .unwrap();
    }

    let kokoro_url = format!(
        "http://127.0.0.1:{}/v1/audio/speech",
        state.config.general.kokoro_port.unwrap_or(8880)
    );

    let payload = serde_json::json!({
        "model": "kokoro",
        "input": text,
        "voice": voice,
        "response_format": "wav",
        "speed": speed,
    });

    let client = reqwest::Client::new();
    match client.post(&kokoro_url).json(&payload).send().await {
        Ok(resp) if resp.status().is_success() => {
            let bytes = resp.bytes().await.unwrap_or_default();
            axum::response::Response::builder()
                .status(200)
                .header("content-type", "audio/wav")
                .body(axum::body::Body::from(bytes))
                .unwrap()
        }
        Ok(resp) => {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            tracing::warn!(status, body = %body, "Kokoro TTS error");
            axum::response::Response::builder()
                .status(502)
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({"error": "Kokoro TTS error", "detail": body}).to_string(),
                ))
                .unwrap()
        }
        Err(e) => {
            tracing::warn!(error = %e, "Kokoro TTS unreachable");
            axum::response::Response::builder()
                .status(503)
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({"error": "Kokoro TTS server unreachable", "hint": "Start with: python start-kokoro.py"}).to_string(),
                ))
                .unwrap()
        }
    }
}

/// GET /api/tts/status — Check if Kokoro TTS is available.
pub async fn tts_status(State(state): State<AppState>) -> impl IntoResponse {
    let port = state.config.general.kokoro_port.unwrap_or(8880);
    let url = format!("http://127.0.0.1:{}/v1/models", port);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap_or_default();
    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            Json(serde_json::json!({"available": true, "port": port}))
        }
        _ => Json(serde_json::json!({"available": false, "port": port})),
    }
}
