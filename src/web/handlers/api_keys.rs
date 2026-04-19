//! API key management handlers — persist search provider keys to data/api_keys.json.

use crate::web::state::AppState;
use axum::{extract::State, response::IntoResponse, Json};

const API_KEY_NAMES: &[&str] = &[
    "BRAVE_API_KEY",
    "SERPER_API_KEY",
    "TAVILY_API_KEY",
    "SERPAPI_API_KEY",
];

/// Load persisted API keys into process env vars at startup.
pub fn load_into_env(state: &AppState) {
    let keys = load(state);
    let mut loaded = 0;
    for (name, value) in &keys {
        if !value.is_empty() {
            std::env::set_var(name, value);
            loaded += 1;
        }
    }
    if loaded > 0 {
        tracing::info!(count = loaded, "Loaded API keys from disk into env");
    }
}

fn keys_path(state: &AppState) -> std::path::PathBuf {
    state.config.general.data_dir.join("api_keys.json")
}

fn load(state: &AppState) -> std::collections::HashMap<String, String> {
    let path = keys_path(state);
    if path.exists() {
        std::fs::read_to_string(&path).ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default()
    } else {
        std::collections::HashMap::new()
    }
}

fn mask_key(key: &str) -> String {
    if key.len() <= 8 { return "••••••••".to_string(); }
    format!("{}••••••••{}", &key[..4], &key[key.len()-4..])
}

pub async fn get_keys(State(state): State<AppState>) -> impl IntoResponse {
    let keys = load(&state);
    let mut masked = serde_json::Map::new();
    for &name in API_KEY_NAMES {
        let val = keys.get(name).cloned().unwrap_or_default();
        masked.insert(name.to_string(), serde_json::json!({
            "set": !val.is_empty(),
            "masked": if val.is_empty() { String::new() } else { mask_key(&val) },
        }));
    }
    Json(serde_json::json!({ "keys": masked }))
}

pub async fn save_keys(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let mut keys = load(&state);
    let path = keys_path(&state);

    if let Some(updates) = body.as_object() {
        for (name, value) in updates {
            if !API_KEY_NAMES.contains(&name.as_str()) { continue; }
            if let Some(val) = value.as_str() {
                if val.is_empty() {
                    keys.remove(name);
                    std::env::remove_var(name);
                } else {
                    keys.insert(name.clone(), val.to_string());
                    std::env::set_var(name, val);
                }
            }
        }
    }

    match serde_json::to_string_pretty(&keys) {
        Ok(json) => match std::fs::write(&path, &json) {
            Ok(_) => {
                tracing::info!(path = %path.display(), count = keys.len(), "API keys saved");
                Json(serde_json::json!({ "ok": true }))
            }
            Err(e) => {
                tracing::error!(error = %e, "Failed to write API keys");
                Json(serde_json::json!({ "ok": false, "error": e.to_string() }))
            }
        },
        Err(e) => Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
    }
}
