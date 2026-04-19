//! System handlers — health, status, models, factory reset, tools, training, interpretability, steering, learning, observer, logs.

use crate::web::state::AppState;
use axum::{extract::State, response::IntoResponse, Json};

pub async fn health_check() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

pub async fn system_status(State(state): State<AppState>) -> impl IntoResponse {
    let memory = state.memory.read().await;
    let sessions = state.sessions.read().await;
    let provider_healthy = state.provider.health().await;

    Json(serde_json::json!({
        "model": {
            "name": state.model_spec.name,
            "context_length": state.model_spec.context_length,
            "supports_vision": state.model_spec.supports_vision,
            "supports_video": state.model_spec.supports_video,
            "supports_audio": state.model_spec.supports_audio,
            "supports_tool_calling": state.model_spec.supports_tool_calling,
            "supports_thinking": state.model_spec.supports_thinking,
        },
        "memory": memory.status_summary(),
        "provider": state.config.general.active_provider,
        "provider_healthy": provider_healthy,
        "observer": { "enabled": state.config.observer.enabled },
        "sessions": sessions.list().len(),
    }))
}

pub async fn list_models() -> impl IntoResponse {
    let models_dir = std::path::Path::new("./models");
    let mut models = Vec::new();

    if let Ok(entries) = std::fs::read_dir(models_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(ext) = path.extension() {
                if ext == "gguf" {
                    let name = path.file_name().unwrap_or_default().to_string_lossy().to_string();
                    let size_bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
                    let is_mmproj = name.starts_with("mmproj");
                    models.push(serde_json::json!({
                        "name": name,
                        "path": path.to_string_lossy(),
                        "size_gb": format!("{:.1}", size_bytes as f64 / 1_073_741_824.0),
                        "is_mmproj": is_mmproj,
                    }));
                }
            }
        }
    }

    Json(serde_json::json!({ "models": models }))
}

pub async fn factory_reset(State(state): State<AppState>) -> impl IntoResponse {
    tracing::warn!("FACTORY RESET initiated via WebUI");

    { let mut memory = state.memory.write().await; memory.clear(); }

    {
        let mut sessions = state.sessions.write().await;
        let ids: Vec<String> = sessions.list().iter().map(|s| s.id.clone()).collect();
        for id in ids { let _ = sessions.delete(&id); }
    }

    let _ = std::fs::write("data/golden_buffer.json", "[]");
    let _ = std::fs::write("data/rejection_buffer.json", "[]");
    let _ = std::fs::remove_dir_all("data/timeline");
    let _ = std::fs::create_dir_all("data/timeline");
    // Reset onboarding so the welcome flow triggers again
    let _ = std::fs::remove_file("data/user_profile.json");
    // Clear observer and sleep history
    let _ = std::fs::remove_file("data/observer_history.json");
    let _ = std::fs::remove_file("data/sleep_history.json");

    tracing::info!("Factory reset complete — all data wiped (including onboarding)");
    Json(serde_json::json!({ "ok": true, "message": "Factory reset complete. All data cleared." }))
}

pub async fn tools_catalog() -> impl IntoResponse {
    let l1 = crate::tools::schema::layer1_tools();
    let l2 = crate::tools::schema::layer2_tools();
    Json(serde_json::json!({ "layer1": l1, "layer2": l2 }))
}

pub async fn training_buffers(State(state): State<AppState>) -> impl IntoResponse {
    let golden: Vec<serde_json::Value> = std::fs::read_to_string("data/golden_buffer.json")
        .ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
    let rejections: Vec<serde_json::Value> = std::fs::read_to_string("data/rejection_buffer.json")
        .ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
    let _ = state;
    Json(serde_json::json!({
        "golden": { "count": golden.len(), "entries": golden },
        "rejections": { "count": rejections.len(), "entries": rejections },
    }))
}

pub async fn interp_features() -> impl IntoResponse {
    let features = crate::interpretability::features::labeled_features();
    let entries: Vec<serde_json::Value> = features.iter().map(|f| {
        serde_json::json!({
            "index": f.index, "label": f.label,
            "category": f.category, "baseline_activation": f.baseline_activation,
        })
    }).collect();
    Json(serde_json::json!({ "count": entries.len(), "features": entries }))
}

pub async fn interp_snapshots() -> impl IntoResponse {
    let dir = std::path::Path::new("data/snapshots");
    let mut snapshots = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut paths: Vec<_> = entries.flatten()
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "json"))
            .collect();
        paths.sort_by_key(|e| std::cmp::Reverse(e.file_name()));
        for entry in paths.iter().take(50) {
            if let Ok(content) = std::fs::read_to_string(entry.path()) {
                if let Ok(snap) = serde_json::from_str::<serde_json::Value>(&content) {
                    snapshots.push(serde_json::json!({ "file": entry.file_name().to_string_lossy(), "data": snap }));
                }
            }
        }
    }
    Json(serde_json::json!({ "count": snapshots.len(), "snapshots": snapshots }))
}

pub async fn interp_sae(State(state): State<AppState>) -> impl IntoResponse {
    let sae = state.sae.read().await;
    let (input_dim, hidden_dim, model_loaded) = match sae.as_ref() {
        Some(s) => (s.model_dim, s.num_features, true),
        None => {
            let c = crate::interpretability::trainer::TrainConfig::default();
            (c.model_dim, c.num_features, false)
        }
    };
    let config = crate::interpretability::trainer::TrainConfig::default();
    Json(serde_json::json!({
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "sparsity_coefficient": config.l1_coefficient,
        "architecture": "JumpReLU",
        "model_loaded": model_loaded,
        "feature_count": crate::interpretability::features::labeled_features().len(),
    }))
}

pub async fn steering_vectors() -> impl IntoResponse {
    let dir = std::path::Path::new("data/steering");
    match crate::steering::vectors::VectorStore::new(dir) {
        Ok(s) => {
            let vectors: Vec<serde_json::Value> = s.list().iter().map(|v| {
                serde_json::json!({
                    "name": v.name, "path": v.path, "strength": v.strength,
                    "active": v.active, "description": v.description,
                })
            }).collect();
            let active_count = s.active_vectors().len();
            Json(serde_json::json!({ "count": vectors.len(), "active_count": active_count, "vectors": vectors }))
        }
        Err(_) => Json(serde_json::json!({ "count": 0, "active_count": 0, "vectors": [] })),
    }
}

pub async fn learning_status(State(state): State<AppState>) -> impl IntoResponse {
    let golden_count = state.golden_buffer.read().await.count();
    let rejection_count = state.rejection_buffer.read().await.count();
    let adapter_dir = std::path::Path::new("data/adapters");
    let adapter_count = crate::learning::lora::adapters::AdapterStore::new(adapter_dir)
        .map(|s| s.count()).unwrap_or(0);
    let sleep_count: usize = std::fs::read_to_string("data/sleep_history.json")
        .ok().and_then(|s| serde_json::from_str::<Vec<serde_json::Value>>(&s).ok())
        .map(|v| v.len()).unwrap_or(0);
    Json(serde_json::json!({
        "golden_buffer_size": golden_count, "rejection_buffer_size": rejection_count,
        "adapter_count": adapter_count, "sleep_cycles": sleep_count,
        "supported_methods": ["SFT", "ORPO", "SimPO", "KTO", "DPO", "GRPO"],
    }))
}

pub async fn learning_adapters() -> impl IntoResponse {
    let dir = std::path::Path::new("data/adapters");
    match crate::learning::lora::adapters::AdapterStore::new(dir) {
        Ok(store) => {
            let adapters: Vec<serde_json::Value> = store.list().iter().map(|a| {
                serde_json::json!({
                    "id": a.id, "name": a.name, "method": a.method,
                    "path": a.path, "created_at": a.created_at.to_rfc3339(),
                    "param_count": a.param_count,
                })
            }).collect();
            Json(serde_json::json!({ "count": adapters.len(), "adapters": adapters }))
        }
        Err(_) => Json(serde_json::json!({ "count": 0, "adapters": [] })),
    }
}

pub async fn learning_sleep_history() -> impl IntoResponse {
    let entries: Vec<serde_json::Value> = std::fs::read_to_string("data/sleep_history.json")
        .ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
    Json(serde_json::json!({ "count": entries.len(), "entries": entries }))
}

pub async fn observer_history() -> impl IntoResponse {
    let entries: Vec<serde_json::Value> = std::fs::read_to_string("data/observer_history.json")
        .ok().and_then(|s| serde_json::from_str(&s).ok()).unwrap_or_default();
    Json(serde_json::json!({ "count": entries.len(), "entries": entries }))
}

pub async fn logs_recent() -> impl IntoResponse {
    let today = chrono::Local::now().format("%Y-%m-%d").to_string();
    let log_path = format!("data/logs/ern-os.log.{}", today);
    let mut entries = Vec::new();
    if let Ok(content) = std::fs::read_to_string(&log_path) {
        let lines: Vec<&str> = content.lines().collect();
        let start = if lines.len() > 200 { lines.len() - 200 } else { 0 };
        for line in &lines[start..] {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(line) {
                entries.push(parsed);
            }
        }
    }
    Json(serde_json::json!({ "count": entries.len(), "entries": entries }))
}

/// GET /api/self-edits — Read self-edit audit trail.
pub async fn self_edits(State(state): State<AppState>) -> impl IntoResponse {
    let path = state.config.general.data_dir.join("self_edit_log.jsonl");
    let mut entries = Vec::new();
    if let Ok(content) = std::fs::read_to_string(&path) {
        for line in content.lines().rev().take(100) {
            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(line) {
                entries.push(parsed);
            }
        }
    }
    Json(serde_json::json!({ "count": entries.len(), "entries": entries }))
}

/// GET /api/checkpoints — List file checkpoints.
pub async fn checkpoints(State(state): State<AppState>) -> impl IntoResponse {
    let dir = state.config.general.data_dir.join("checkpoints");
    let mut entries = Vec::new();
    if let Ok(read_dir) = std::fs::read_dir(&dir) {
        let mut files: Vec<_> = read_dir.flatten().collect();
        files.sort_by_key(|e| std::cmp::Reverse(
            e.metadata().ok().and_then(|m| m.modified().ok()).unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        ));
        for entry in files.iter().take(50) {
            let name = entry.file_name().to_string_lossy().to_string();
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            let modified = entry.metadata().ok()
                .and_then(|m| m.modified().ok())
                .map(|t| {
                    let dt: chrono::DateTime<chrono::Local> = t.into();
                    dt.to_rfc3339()
                })
                .unwrap_or_default();
            entries.push(serde_json::json!({
                "name": name, "size_bytes": size, "modified": modified,
            }));
        }
    }
    Json(serde_json::json!({ "count": entries.len(), "checkpoints": entries }))
}

pub async fn get_prompt(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> impl IntoResponse {
    if !matches!(name.as_str(), "core" | "identity" | "observer") {
        return Json(serde_json::json!({ "error": "Invalid prompt name. Use 'core', 'identity', or 'observer'." }));
    }
    let path = state.config.general.data_dir.join("prompts").join(format!("{}.md", name));
    match std::fs::read_to_string(&path) {
        Ok(content) => Json(serde_json::json!({ "name": name, "content": content })),
        Err(e) => Json(serde_json::json!({ "error": e.to_string() })),
    }
}

pub async fn save_prompt(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    if !matches!(name.as_str(), "core" | "identity" | "observer") {
        return Json(serde_json::json!({ "error": "Invalid prompt name. Use 'core', 'identity', or 'observer'." }));
    }
    let content = match body["content"].as_str() {
        Some(c) => c,
        None => return Json(serde_json::json!({ "error": "Missing 'content' field" })),
    };
    let path = state.config.general.data_dir.join("prompts").join(format!("{}.md", name));
    match std::fs::write(&path, content) {
        Ok(_) => {
            tracing::info!(name = %name, len = content.len(), "Prompt file updated");
            Json(serde_json::json!({ "ok": true, "name": name }))
        }
        Err(e) => Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
    }
}
