//! REST API handlers for the atomic checkpoint system.

use axum::extract::{Json, Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use crate::web::state::AppState;

#[derive(Serialize)]
pub struct CheckpointResponse {
    pub id: String,
    pub label: String,
    pub created_at: String,
    pub git_commit: String,
}

#[derive(Deserialize)]
pub struct CreateCheckpointRequest {
    pub label: String,
}

#[derive(Deserialize)]
pub struct RestoreRequest {
    pub checkpoint_id: String,
}

/// POST /api/state-checkpoint — create a new checkpoint.
pub async fn create_checkpoint(
    State(state): State<AppState>,
    Json(body): Json<CreateCheckpointRequest>,
) -> Result<Json<CheckpointResponse>, StatusCode> {
    let data_dir = std::path::Path::new(&state.config.general.data_dir);
    match crate::checkpoint::snapshot::create_checkpoint(&body.label, data_dir).await {
        Ok(cp) => Ok(Json(CheckpointResponse {
            id: cp.id,
            label: cp.label,
            created_at: cp.created_at.to_rfc3339(),
            git_commit: cp.git_commit,
        })),
        Err(e) => {
            tracing::error!(error = %e, "Failed to create checkpoint");
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// GET /api/state-checkpoint — list all checkpoints.
pub async fn list_checkpoints(
    State(state): State<AppState>,
) -> Json<Vec<CheckpointResponse>> {
    let data_dir = std::path::Path::new(&state.config.general.data_dir);
    let checkpoints = crate::checkpoint::snapshot::list_checkpoints(data_dir)
        .unwrap_or_default();

    Json(checkpoints.into_iter().map(|cp| CheckpointResponse {
        id: cp.id,
        label: cp.label,
        created_at: cp.created_at.to_rfc3339(),
        git_commit: cp.git_commit,
    }).collect())
}

/// POST /api/state-checkpoint/restore — restore from a checkpoint.
pub async fn restore_checkpoint(
    State(state): State<AppState>,
    Json(body): Json<RestoreRequest>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let data_dir = std::path::Path::new(&state.config.general.data_dir);
    match crate::checkpoint::restore::restore_checkpoint(data_dir, &body.checkpoint_id).await {
        Ok(result) => Ok(Json(serde_json::json!({
            "success": result.success,
            "message": result.message,
            "commit": result.restored_commit,
            "code_changed": result.code_changed,
            "memory_restored": result.memory_restored,
            "sessions_restored": result.sessions_restored,
        }))),
        Err(e) => {
            tracing::error!(error = %e, "Failed to restore checkpoint");
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
    }
}

/// DELETE /api/state-checkpoint/:id — delete a checkpoint.
pub async fn delete_checkpoint(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> StatusCode {
    let data_dir = std::path::Path::new(&state.config.general.data_dir);
    match crate::checkpoint::snapshot::delete_checkpoint(data_dir, &id) {
        Ok(_) => StatusCode::NO_CONTENT,
        Err(e) => {
            tracing::error!(error = %e, "Failed to delete checkpoint");
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}
