//! Codes IDE handler — status endpoint for code-server integration.

use crate::web::state::AppState;
use axum::{extract::State, Json};

/// GET /api/codes/status — Check if code-server is running and return connection info.
pub async fn codes_status(State(state): State<AppState>) -> Json<serde_json::Value> {
    let port = state.config.codes.port;
    let enabled = state.config.codes.enabled;
    let url = format!("http://127.0.0.1:{}/healthz", port);

    if !enabled {
        return Json(serde_json::json!({
            "enabled": false,
            "available": false,
            "port": port,
        }));
    }

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap_or_default();

    let available = client.get(&url).send().await
        .map_or(false, |r| r.status().is_success());

    Json(serde_json::json!({
        "enabled": true,
        "available": available,
        "port": port,
        "url": format!("http://localhost:{}", port),
    }))
}
