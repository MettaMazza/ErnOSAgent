// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Session CRUD and export routes.

use super::{api_error, session_to_detail, RenameRequest, SessionDetail, SessionListItem};
use crate::web::state::SharedState;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;

use super::ApiError;

pub async fn list_sessions(State(state): State<SharedState>) -> Json<Vec<SessionListItem>> {
    let st = state.read().await;
    let items: Vec<SessionListItem> = st
        .session_mgr
        .list()
        .iter()
        .map(|s| SessionListItem {
            id: s.id.clone(),
            title: s.title.clone(),
            model: s.model.clone(),
            message_count: s.message_count,
            updated_at: s.updated_at.to_rfc3339(),
        })
        .collect();
    Json(items)
}

pub async fn get_session(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<SessionDetail>, (StatusCode, Json<ApiError>)> {
    let mut st = state.write().await;
    if st.session_mgr.active_id() == id {
        let session = st.session_mgr.active();
        return Ok(Json(session_to_detail(session)));
    }
    st.session_mgr
        .switch_to(&id)
        .map_err(|e| api_error(StatusCode::NOT_FOUND, &e.to_string()))?;
    let session = st.session_mgr.active();
    Ok(Json(session_to_detail(session)))
}

pub async fn create_session(
    State(state): State<SharedState>,
) -> Result<Json<SessionDetail>, (StatusCode, Json<ApiError>)> {
    let mut st = state.write().await;
    let model = st.config.general.active_model.clone();
    let provider = st.config.general.active_provider.clone();

    st.session_mgr
        .new_session(&model, &provider)
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;

    let session = st.session_mgr.active();
    tracing::info!(session_id = %session.id, "Web UI: created new session");
    Ok(Json(session_to_detail(session)))
}

pub async fn delete_session(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, Json<ApiError>)> {
    let mut st = state.write().await;
    st.session_mgr
        .delete(&id)
        .map_err(|e| api_error(StatusCode::BAD_REQUEST, &e.to_string()))?;

    tracing::info!(session_id = %id, "Web UI: deleted session");
    Ok(StatusCode::NO_CONTENT)
}

pub async fn rename_session(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Json(body): Json<RenameRequest>,
) -> Result<StatusCode, (StatusCode, Json<ApiError>)> {
    let mut st = state.write().await;
    if st.session_mgr.active_id() != id {
        st.session_mgr
            .switch_to(&id)
            .map_err(|e| api_error(StatusCode::NOT_FOUND, &e.to_string()))?;
    }

    st.session_mgr
        .rename(&body.title)
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))?;

    Ok(StatusCode::OK)
}

pub async fn export_session(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<SessionDetail>, (StatusCode, Json<ApiError>)> {
    let mut st = state.write().await;
    if st.session_mgr.active_id() == id {
        let session = st.session_mgr.active();
        return Ok(Json(session_to_detail(session)));
    }

    let prev_id = st.session_mgr.active_id().to_string();
    st.session_mgr
        .switch_to(&id)
        .map_err(|e| api_error(StatusCode::NOT_FOUND, &e.to_string()))?;

    let session = st.session_mgr.active();
    let detail = session_to_detail(session);
    let _ = st.session_mgr.switch_to(&prev_id);

    Ok(Json(detail))
}

/// Request body for the reaction endpoint.
#[derive(serde::Deserialize)]
pub struct ReactionBody {
    pub reaction: String, // "up" or "down"
}

/// Record a user reaction (👍/👎) on the current session's last response.
/// - "up" → golden buffer (user-approved SFT data)
/// - "down" → marks the last response as needing preference training
pub async fn react_to_message(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Json(body): Json<ReactionBody>,
) -> Result<StatusCode, (StatusCode, Json<ApiError>)> {
    let st = state.read().await;

    // Ensure we're on the right session
    if st.session_mgr.active_id() != id {
        return Err(api_error(StatusCode::NOT_FOUND, "Session not active"));
    }

    let session = st.session_mgr.active();
    let messages = &session.messages;

    // Find the last user→assistant pair
    let last_user = messages.iter().rev().find(|m| m.role == "user");
    let last_assistant = messages.iter().rev().find(|m| m.role == "assistant");

    let (user_msg, assistant_msg) = match (last_user, last_assistant) {
        (Some(u), Some(a)) => (u.content.clone(), a.content.clone()),
        _ => {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "No message pair to react to",
            ))
        }
    };

    match body.reaction.as_str() {
        "up" => {
            if let Some(ref buffers) = st.training_buffers {
                let _ = buffers.golden.record(
                    &st.core_prompt,
                    &user_msg,
                    &assistant_msg,
                    &id,
                    &st.model_spec.name,
                );
                tracing::info!(session = %id, "👍 Reaction recorded → golden buffer");
            }
        }
        "down" => {
            if let Some(ref buffers) = st.training_buffers {
                let _ = buffers.preference.record(
                    &st.core_prompt,
                    &user_msg,
                    &assistant_msg,
                    "", // chosen_response will be filled during training
                    "user_rejected",
                    &id,
                    &st.model_spec.name,
                );
                tracing::info!(session = %id, "👎 Reaction recorded → preference buffer");
            }
        }
        _ => {
            return Err(api_error(
                StatusCode::BAD_REQUEST,
                "Invalid reaction type, expected 'up' or 'down'",
            ))
        }
    }

    Ok(StatusCode::OK)
}
