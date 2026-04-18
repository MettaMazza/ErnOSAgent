// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Scheduler REST API routes.

use super::{api_error, ApiError};
use crate::scheduler::job::{CreateJobRequest, ScheduledJob};
use crate::web::state::SharedState;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::Json;

/// GET /api/scheduler/jobs — list all jobs.
pub async fn list_jobs(
    State(state): State<SharedState>,
) -> Result<Json<Vec<ScheduledJob>>, (StatusCode, Json<ApiError>)> {
    let st = state.read().await;
    match &st.scheduler {
        Some(s) => Ok(Json(s.list_jobs().await)),
        None => Err(api_error(
            StatusCode::SERVICE_UNAVAILABLE,
            "Scheduler not enabled",
        )),
    }
}

/// POST /api/scheduler/jobs — create a new job.
pub async fn create_job(
    State(state): State<SharedState>,
    Json(body): Json<CreateJobRequest>,
) -> Result<Json<ScheduledJob>, (StatusCode, Json<ApiError>)> {
    let st = state.read().await;
    let scheduler = st
        .scheduler
        .as_ref()
        .ok_or_else(|| api_error(StatusCode::SERVICE_UNAVAILABLE, "Scheduler not enabled"))?;

    let job = ScheduledJob {
        id: uuid::Uuid::new_v4().to_string(),
        name: body.name,
        instruction: body.instruction,
        schedule: body.schedule,
        enabled: true,
        created_at: chrono::Utc::now(),
        last_run: None,
        last_result: None,
        delivery_channel: body.delivery_channel,
    };

    scheduler
        .add_job(job)
        .await
        .map(Json)
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, &e.to_string()))
}

/// PUT /api/scheduler/jobs/{id} — update a job.
pub async fn update_job(
    State(state): State<SharedState>,
    Path(id): Path<String>,
    Json(body): Json<CreateJobRequest>,
) -> Result<Json<ScheduledJob>, (StatusCode, Json<ApiError>)> {
    let st = state.read().await;
    let scheduler = st
        .scheduler
        .as_ref()
        .ok_or_else(|| api_error(StatusCode::SERVICE_UNAVAILABLE, "Scheduler not enabled"))?;

    scheduler
        .update_job(&id, body)
        .await
        .map(Json)
        .map_err(|e| api_error(StatusCode::NOT_FOUND, &e.to_string()))
}

/// DELETE /api/scheduler/jobs/{id} — delete a job.
pub async fn delete_job(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, Json<ApiError>)> {
    let st = state.read().await;
    let scheduler = st
        .scheduler
        .as_ref()
        .ok_or_else(|| api_error(StatusCode::SERVICE_UNAVAILABLE, "Scheduler not enabled"))?;

    scheduler
        .delete_job(&id)
        .await
        .map(|_| StatusCode::NO_CONTENT)
        .map_err(|e| api_error(StatusCode::NOT_FOUND, &e.to_string()))
}

/// POST /api/scheduler/jobs/{id}/toggle — enable/disable a job.
pub async fn toggle_job(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    let st = state.read().await;
    let scheduler = st
        .scheduler
        .as_ref()
        .ok_or_else(|| api_error(StatusCode::SERVICE_UNAVAILABLE, "Scheduler not enabled"))?;

    scheduler
        .toggle_job(&id)
        .await
        .map(|enabled| Json(serde_json::json!({ "enabled": enabled })))
        .map_err(|e| api_error(StatusCode::NOT_FOUND, &e.to_string()))
}

/// POST /api/scheduler/jobs/{id}/run — execute a job immediately.
pub async fn run_job_now(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    let scheduler = {
        let st = state.read().await;
        st.scheduler
            .clone()
            .ok_or_else(|| api_error(StatusCode::SERVICE_UNAVAILABLE, "Scheduler not enabled"))?
    };

    let result = scheduler
        .run_now(&id, &state)
        .await
        .map_err(|e| api_error(StatusCode::NOT_FOUND, &e.to_string()))?;

    Ok(Json(serde_json::json!({
        "success": result.success,
        "output": result.output,
        "duration_ms": result.duration_ms,
    })))
}

/// GET /api/scheduler/jobs/{id}/logs — get execution history for a job.
pub async fn job_logs(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<ApiError>)> {
    let st = state.read().await;
    let scheduler = st
        .scheduler
        .as_ref()
        .ok_or_else(|| api_error(StatusCode::SERVICE_UNAVAILABLE, "Scheduler not enabled"))?;

    let job = scheduler
        .get_job(&id)
        .await
        .ok_or_else(|| api_error(StatusCode::NOT_FOUND, "Job not found"))?;

    Ok(Json(serde_json::json!({
        "job_id": job.id,
        "job_name": job.name,
        "last_run": job.last_run,
        "last_result": job.last_result,
    })))
}
