//! Onboarding & identity handlers.

use crate::web::state::AppState;
use axum::{extract::State, response::IntoResponse, Json};

#[derive(serde::Deserialize, serde::Serialize)]
pub struct UserProfile {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub onboarding_complete: bool,
}

pub fn profile_path(state: &AppState) -> std::path::PathBuf {
    state.config.general.data_dir.join("user_profile.json")
}

pub fn load_profile(state: &AppState) -> Option<UserProfile> {
    let path = profile_path(state);
    if path.exists() {
        std::fs::read_to_string(&path).ok().and_then(|s| serde_json::from_str(&s).ok())
    } else {
        None
    }
}

pub async fn status(State(state): State<AppState>) -> impl IntoResponse {
    let profile = load_profile(&state);
    Json(serde_json::json!({
        "complete": profile.as_ref().map(|p| p.onboarding_complete).unwrap_or(false),
        "profile": profile.map(|p| serde_json::json!({ "name": p.name, "description": p.description })),
    }))
}

pub async fn save_profile(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let profile = UserProfile {
        name: body["name"].as_str().unwrap_or("User").to_string(),
        description: body["description"].as_str().unwrap_or("").to_string(),
        onboarding_complete: false,
    };
    let path = profile_path(&state);
    match serde_json::to_string_pretty(&profile) {
        Ok(json) => match std::fs::write(&path, &json) {
            Ok(_) => { tracing::info!(path = %path.display(), "User profile saved"); Json(serde_json::json!({ "ok": true })) }
            Err(e) => Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
        },
        Err(e) => Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
    }
}

pub async fn complete(State(state): State<AppState>) -> impl IntoResponse {
    let path = profile_path(&state);
    tracing::info!(path = %path.display(), "Completing onboarding — loading profile");
    let mut profile = load_profile(&state).unwrap_or(UserProfile {
        name: "User".to_string(), description: String::new(), onboarding_complete: false,
    });
    profile.onboarding_complete = true;

    match serde_json::to_string_pretty(&profile) {
        Ok(json) => match std::fs::write(&path, &json) {
            Ok(_) => {
                tracing::info!(path = %path.display(), "Onboarding marked complete on disk");
                Json(serde_json::json!({ "ok": true }))
            }
            Err(e) => {
                tracing::error!(path = %path.display(), error = %e, "Failed to write onboarding profile");
                Json(serde_json::json!({ "ok": false, "error": e.to_string() }))
            }
        },
        Err(e) => {
            tracing::error!(error = %e, "Failed to serialize onboarding profile");
            Json(serde_json::json!({ "ok": false, "error": e.to_string() }))
        }
    }
}
