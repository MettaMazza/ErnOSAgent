// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Status, memory, learning, config, observer, and model list routes.

use super::{MemoryStatusResponse, StatusResponse};
use crate::web::state::SharedState;
use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;
use serde::Serialize;

pub async fn status(State(state): State<SharedState>) -> Json<StatusResponse> {
    let st = state.read().await;
    let usage = crate::inference::context::context_usage(
        &st.session_mgr.active().messages,
        st.model_spec.context_length,
    );

    Json(StatusResponse {
        model_name: st.model_spec.name.clone(),
        model_provider: st.model_spec.provider.clone(),
        context_length: st.model_spec.context_length,
        context_usage: usage,
        is_generating: st.is_generating,
        capabilities: st.model_spec.capabilities.modality_badges(),
    })
}

pub async fn memory_status(State(state): State<SharedState>) -> Json<MemoryStatusResponse> {
    let st = state.read().await;
    let summary = st.memory_mgr.status_summary().await;
    let entity_count = st.memory_mgr.kg_entity_count().await;
    let relation_count = st.memory_mgr.kg_relation_count().await;

    Json(MemoryStatusResponse {
        summary,
        kg_available: st.memory_mgr.kg_available(),
        kg_entity_count: entity_count,
        kg_relation_count: relation_count,
        lessons_count: st.memory_mgr.lessons.count(),
        procedures_count: st.memory_mgr.procedures.count(),
        scratchpad_count: st.memory_mgr.scratchpad.count(),
        timeline_count: st.memory_mgr.timeline.entry_count(),
        embeddings_count: st.memory_mgr.embeddings.count(),
        consolidation_count: st.memory_mgr.consolidation.consolidation_count(),
    })
}

#[derive(Serialize)]
pub struct LearningStatusResponse {
    pub enabled: bool,
    pub golden_count: usize,
    pub preference_count: usize,
    pub rejection_count: usize,
    pub observer_audit_count: usize,
    pub distilled_lessons_count: usize,
    pub threshold: String,
    pub teacher_state: String,
    pub can_train: bool,
    pub summary: String,
    pub adapters: Vec<AdapterVersionDto>,
}

#[derive(Serialize)]
pub struct AdapterVersionDto {
    pub id: String,
    pub created: String,
    pub golden_count: usize,
    pub preference_count: usize,
    pub training_loss: f32,
    pub healthy: bool,
}

pub async fn learning_status(State(state): State<SharedState>) -> Json<LearningStatusResponse> {
    let st = state.read().await;

    let (golden, preference, rejection, observer_audit, enabled, summary) =
        match &st.training_buffers {
            Some(buffers) => (
                buffers.golden.count(),
                buffers.preference.count(),
                buffers.rejection.count(),
                buffers.observer.count(),
                true,
                buffers.status(),
            ),
            None => (0, 0, 0, 0, false, "Learning disabled".to_string()),
        };

    let (teacher_state, can_train, threshold) = match &st.teacher {
        Some(teacher) => {
            let state_str = futures::executor::block_on(teacher.state()).to_string();
            let is_idle = state_str == "idle";
            let meets_threshold = golden >= teacher.config().golden_threshold
                || preference >= teacher.config().preference_threshold;
            (
                state_str,
                is_idle && meets_threshold && enabled,
                format!(
                    "Golden: {}/{}, Preference: {}/{}",
                    golden,
                    teacher.config().golden_threshold,
                    preference,
                    teacher.config().preference_threshold,
                ),
            )
        }
        None => ("not initialised".to_string(), false, "—".to_string()),
    };

    // Count distilled lessons from file
    let distilled_lessons_count = st
        .teacher
        .as_ref()
        .map(|t| {
            let path = t.config().training_dir.join("distilled_lessons.json");
            if path.exists() {
                std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
                    .and_then(|v| v.as_array().map(|a| a.len()))
                    .unwrap_or(0)
            } else {
                0
            }
        })
        .unwrap_or(0);

    // Adapter history from manifest
    let adapters = match &st.adapter_manifest {
        Some(manifest_lock) => {
            let manifest = manifest_lock.lock().await;
            manifest
                .history
                .iter()
                .map(|v| AdapterVersionDto {
                    id: v.id.clone(),
                    created: v.created.to_rfc3339(),
                    golden_count: v.golden_count,
                    preference_count: v.preference_count,
                    training_loss: v.training_loss,
                    healthy: v.health_check_passed,
                })
                .collect()
        }
        None => Vec::new(),
    };

    Json(LearningStatusResponse {
        enabled,
        golden_count: golden,
        preference_count: preference,
        rejection_count: rejection,
        observer_audit_count: observer_audit,
        distilled_lessons_count,
        threshold,
        teacher_state,
        can_train,
        summary,
        adapters,
    })
}

/// POST /api/learning/train — trigger a LoRA training cycle manually.
pub async fn trigger_training(
    State(state): State<SharedState>,
) -> Result<StatusCode, (StatusCode, Json<super::ApiError>)> {
    let (teacher, buffers, manifest) = {
        let st = state.read().await;
        let teacher = st.teacher.clone().ok_or_else(|| {
            super::api_error(StatusCode::SERVICE_UNAVAILABLE, "Teacher not initialised")
        })?;
        let buffers = st.training_buffers.clone().ok_or_else(|| {
            super::api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "Training buffers not initialised",
            )
        })?;
        let manifest = st.adapter_manifest.clone().ok_or_else(|| {
            super::api_error(
                StatusCode::SERVICE_UNAVAILABLE,
                "Adapter manifest not initialised",
            )
        })?;
        (teacher, buffers, manifest)
    };

    if teacher.is_training() {
        return Err(super::api_error(
            StatusCode::CONFLICT,
            "Training already in progress",
        ));
    }

    // Determine training kind
    let kind = match teacher.should_train(&buffers).await {
        Some(kind) => kind,
        None => {
            // Force SFT if there's any golden data, ORPO if any preference data
            if buffers.golden.count() > 0 {
                crate::learning::teacher::TrainingKind::Sft
            } else if buffers.preference.count() > 0 {
                crate::learning::teacher::TrainingKind::Orpo
            } else {
                return Err(super::api_error(
                    StatusCode::BAD_REQUEST,
                    "No training data available — approve messages with 👍 or reject with 👎 first",
                ));
            }
        }
    };

    // Spawn training in background so we don't block the request
    tokio::spawn(async move {
        tracing::info!(kind = ?kind, "Manual training triggered via dashboard");
        let mut manifest_guard = manifest.lock().await;
        match teacher
            .run_training_cycle(&buffers, &mut manifest_guard, kind)
            .await
        {
            Ok(()) => tracing::info!("Manual training cycle completed successfully"),
            Err(e) => tracing::error!(error = %e, "Manual training cycle failed"),
        }
    });

    Ok(StatusCode::ACCEPTED)
}

pub async fn list_models(State(state): State<SharedState>) -> Json<Vec<String>> {
    let st = state.read().await;

    // Start with the currently loaded model(s) from the provider
    let mut names: Vec<String> = match st.provider.list_models().await {
        Ok(models) => models.into_iter().map(|m| m.name).collect(),
        Err(_) => vec![st.config.general.active_model.clone()],
    };

    // Scan the local models directory for available .gguf files
    let models_dir = std::path::Path::new("models");
    if models_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(models_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(ext) = path.extension() {
                    if ext == "gguf" {
                        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                            // Skip multimodal projector files (mmproj-*)
                            if !stem.starts_with("mmproj-") {
                                let name = stem.to_string();
                                if !names.contains(&name) {
                                    names.push(name);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Json(names)
}

// ── Config route ─────────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ConfigResponse {
    provider: String,
    model: String,
    context_window: u64,
    data_dir: String,
    observer_enabled: bool,
    web_port: u16,
    steering_summary: String,
}

pub async fn get_config(State(state): State<SharedState>) -> Json<ConfigResponse> {
    let st = state.read().await;
    Json(ConfigResponse {
        provider: st.config.general.active_provider.clone(),
        model: st.config.general.active_model.clone(),
        context_window: st.model_spec.context_length,
        data_dir: st.config.general.data_dir.display().to_string(),
        observer_enabled: st.config.observer.enabled,
        web_port: st.config.web.port,
        steering_summary: st.steering_config.status_summary(),
    })
}

// ── Observer routes ──────────────────────────────────────────────────

#[derive(Serialize)]
pub struct ObserverConfigResponse {
    enabled: bool,
    model: String,
    think: bool,
}

pub async fn get_observer(State(state): State<SharedState>) -> Json<ObserverConfigResponse> {
    let st = state.read().await;
    Json(ObserverConfigResponse {
        enabled: st.config.observer.enabled,
        model: if st.config.observer.model.is_empty() {
            st.config.general.active_model.clone()
        } else {
            st.config.observer.model.clone()
        },
        think: st.config.observer.think,
    })
}

pub async fn toggle_observer(State(state): State<SharedState>) -> Json<serde_json::Value> {
    let mut st = state.write().await;
    st.config.observer.enabled = !st.config.observer.enabled;
    let enabled = st.config.observer.enabled;
    tracing::info!(enabled, "Observer toggled via dashboard");
    Json(serde_json::json!({ "enabled": enabled }))
}
