//! Scheduler CRUD handlers.

use crate::web::state::AppState;
use axum::{extract::State, response::IntoResponse, Json};

pub async fn status(State(state): State<AppState>) -> impl IntoResponse {
    let store = state.scheduler.read().await;
    let jobs: Vec<serde_json::Value> = store.list().iter().map(|j| {
        serde_json::json!({
            "id": j.id, "name": j.name, "description": j.description,
            "schedule": format!("{}", j.schedule), "schedule_raw": j.schedule,
            "task": format!("{}", j.task), "task_raw": j.task,
            "enabled": j.enabled, "builtin": j.builtin,
            "created_at": j.created_at.to_rfc3339(),
            "last_run": j.last_run.map(|t| t.to_rfc3339()),
            "last_result": j.last_result, "run_count": j.run_count,
        })
    }).collect();
    let enabled_count = store.list().iter().filter(|j| j.enabled).count();
    Json(serde_json::json!({
        "running": true, "tick_interval_seconds": 15,
        "total_jobs": jobs.len(), "enabled_jobs": enabled_count, "jobs": jobs,
    }))
}

pub async fn create_job(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let name = body["name"].as_str().unwrap_or("unnamed").to_string();
    let description = body["description"].as_str().unwrap_or("").to_string();

    let schedule = parse_schedule(&body);
    let schedule = match schedule {
        Ok(s) => s,
        Err(e) => return (axum::http::StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": e}))),
    };

    let task = parse_task(&body);

    let job = crate::scheduler::job::CronJob {
        id: uuid::Uuid::new_v4().to_string(),
        name, description, schedule, task,
        enabled: true, created_at: chrono::Utc::now(),
        last_run: None, last_result: None, run_count: 0, builtin: false,
    };

    let mut store = state.scheduler.write().await;
    match store.add(job) {
        Ok(id) => (axum::http::StatusCode::CREATED, Json(serde_json::json!({"id": id, "status": "created"}))),
        Err(e) => (axum::http::StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({"error": format!("{}", e)}))),
    }
}

fn parse_schedule(body: &serde_json::Value) -> Result<crate::scheduler::job::JobSchedule, String> {
    match body["schedule_type"].as_str() {
        Some("cron") => {
            let expr = body["schedule_value"].as_str().unwrap_or("0 */5 * * * *");
            Ok(crate::scheduler::job::JobSchedule::Cron(expr.to_string()))
        }
        Some("once") => {
            let dt_str = body["schedule_value"].as_str().unwrap_or("");
            dt_str.parse::<chrono::DateTime<chrono::Utc>>()
                .map(crate::scheduler::job::JobSchedule::Once)
                .map_err(|_| "Invalid datetime for once schedule".to_string())
        }
        _ => {
            let secs = body["schedule_value"].as_u64().unwrap_or(300);
            Ok(crate::scheduler::job::JobSchedule::Interval(secs))
        }
    }
}

fn parse_task(body: &serde_json::Value) -> crate::scheduler::job::JobTask {
    use crate::scheduler::job::JobTask;
    match body["task_type"].as_str() {
        Some("sleep_cycle") => JobTask::SleepCycle,
        Some("lesson_decay") => JobTask::LessonDecay,
        Some("memory_consolidate") => JobTask::MemoryConsolidate,
        Some("snapshot_capture") => JobTask::SnapshotCapture,
        Some("synaptic_prune") => JobTask::SynapticPrune,
        Some("buffer_flush") => JobTask::BufferFlush,
        Some("log_rotate") => JobTask::LogRotate,
        Some("health_check") => JobTask::HealthCheck,
        Some("custom") => {
            let cmd = body["custom_command"].as_str().unwrap_or("echo ok").to_string();
            JobTask::Custom(cmd)
        }
        _ => JobTask::HealthCheck,
    }
}

pub async fn delete_job(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let mut store = state.scheduler.write().await;
    match store.remove(&id) {
        Ok(true) => Json(serde_json::json!({"status": "deleted"})),
        Ok(false) => Json(serde_json::json!({"status": "not_found"})),
        Err(e) => Json(serde_json::json!({"error": format!("{}", e)})),
    }
}

pub async fn toggle_job(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
) -> impl IntoResponse {
    let mut store = state.scheduler.write().await;
    match store.toggle(&id) {
        Ok(new_state) => Json(serde_json::json!({"enabled": new_state})),
        Err(e) => Json(serde_json::json!({"error": format!("{}", e)})),
    }
}

pub async fn history(State(state): State<AppState>) -> impl IntoResponse {
    let store = state.scheduler.read().await;
    let history: Vec<serde_json::Value> = store.get_history().iter().rev().take(50).map(|h| {
        serde_json::json!({
            "job_id": h.job_id, "job_name": h.job_name, "task": h.task,
            "started_at": h.started_at.to_rfc3339(), "duration_ms": h.duration_ms,
            "success": h.success, "result": h.result,
        })
    }).collect();
    Json(serde_json::json!({ "count": history.len(), "entries": history }))
}
