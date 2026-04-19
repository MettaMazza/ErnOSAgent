//! Platform handler — REST API for managing platform adapters.

use crate::web::state::AppState;
use axum::{extract::State, Json};

/// GET /api/platforms — list all platform adapters and their status.
pub async fn list_platforms(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let reg = state.platforms.read().await;
    let statuses: Vec<serde_json::Value> = reg.statuses().iter().map(|s| {
        serde_json::json!({
            "name": s.name,
            "connected": s.connected,
            "error": s.error,
        })
    }).collect();

    Json(serde_json::json!({
        "platforms": statuses,
        "summary": reg.status_summary(),
    }))
}

/// POST /api/platforms/:name/connect — connect a specific platform.
pub async fn connect_platform(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Json<serde_json::Value> {
    let mut reg = state.platforms.write().await;
    match reg.connect_by_name(&name).await {
        Ok(_) => Json(serde_json::json!({
            "success": true,
            "message": format!("{} connected", name),
        })),
        Err(e) => Json(serde_json::json!({
            "success": false,
            "error": e.to_string(),
        })),
    }
}

/// POST /api/platforms/:name/disconnect — disconnect a specific platform.
pub async fn disconnect_platform(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Json<serde_json::Value> {
    let mut reg = state.platforms.write().await;
    match reg.disconnect_by_name(&name).await {
        Ok(_) => Json(serde_json::json!({
            "success": true,
            "message": format!("{} disconnected", name),
        })),
        Err(e) => Json(serde_json::json!({
            "success": false,
            "error": e.to_string(),
        })),
    }
}
