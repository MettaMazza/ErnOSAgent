// Ern-OS — Background scheduler
//! Job-driven cron engine — runs registered jobs on their schedules,
//! records execution history, and exposes state for the WebUI.

pub mod job;
pub mod store;

use crate::web::state::AppState;
use job::{JobExecution, JobTask};
use tokio::task::JoinHandle;

/// Start the background scheduler. Returns a JoinHandle for the spawned task.
pub fn start(state: AppState) -> JoinHandle<()> {
    tracing::info!("Scheduler started — job-driven cron engine");
    tokio::spawn(async move {
        let tick_interval = tokio::time::Duration::from_secs(15);
        let mut last_tick = chrono::Utc::now();

        loop {
            tokio::time::sleep(tick_interval).await;
            let now = chrono::Utc::now();
            tick(&state, now, last_tick).await;
            last_tick = now;
        }
    })
}

/// Single scheduler tick — check all jobs and run those that are due.
async fn tick(state: &AppState, now: chrono::DateTime<chrono::Utc>, last_tick: chrono::DateTime<chrono::Utc>) {
    let due_jobs: Vec<(String, JobTask)> = {
        let store = state.scheduler.read().await;
        store.jobs.iter()
            .filter(|j| j.is_due(now, last_tick))
            .map(|j| (j.id.clone(), j.task.clone()))
            .collect()
    };

    for (job_id, task) in due_jobs {
        let start_time = chrono::Utc::now();
        let (success, result) = execute_task(&task, state).await;
        let elapsed = chrono::Utc::now().signed_duration_since(start_time).num_milliseconds() as u64;

        let job_name = {
            let store = state.scheduler.read().await;
            store.jobs.iter().find(|j| j.id == job_id).map(|j| j.name.clone()).unwrap_or_default()
        };

        let exec = JobExecution {
            job_id: job_id.clone(),
            job_name,
            task: format!("{}", task),
            started_at: start_time,
            duration_ms: elapsed,
            success,
            result: result.clone(),
        };

        tracing::info!(
            job = %exec.job_name, task = %exec.task, success,
            duration_ms = elapsed, "Job executed"
        );

        let mut store = state.scheduler.write().await;
        store.record_execution(&job_id, exec);
    }
}

/// Execute a single task. Returns (success, result_message).
async fn execute_task(task: &JobTask, state: &AppState) -> (bool, String) {
    match task {
        JobTask::SleepCycle => run_sleep(state).await,
        JobTask::LessonDecay => run_lesson_decay(state).await,
        JobTask::MemoryConsolidate => run_memory_consolidate(state).await,
        JobTask::SnapshotCapture => run_snapshot_capture().await,
        JobTask::SynapticPrune => run_synaptic_prune(state).await,
        JobTask::BufferFlush => run_buffer_flush(state).await,
        JobTask::LogRotate => run_log_rotate().await,
        JobTask::HealthCheck => run_health_check(state).await,
        JobTask::Custom(cmd) => run_custom_command(cmd).await,
    }
}

// ─── Task Implementations ───

async fn run_sleep(state: &AppState) -> (bool, String) {
    let config = crate::learning::sleep::SleepConfig::default();
    let mut golden = state.golden_buffer.write().await;
    let mut rejection = state.rejection_buffer.write().await;
    let mut memory = state.memory.write().await;

    match crate::learning::sleep::run_sleep_cycle(
        &config, &mut golden, &mut rejection, &mut memory,
    ).await {
        Ok(report) => {
            let msg = format!(
                "golden_trained={}, pairs_trained={}, edges_decayed={}",
                report.golden_trained, report.pairs_trained, report.edges_decayed
            );
            (true, msg)
        }
        Err(e) => (false, format!("{}", e)),
    }
}

async fn run_lesson_decay(state: &AppState) -> (bool, String) {
    let mut memory = state.memory.write().await;
    let count = memory.lessons.count();
    if count == 0 {
        return (true, "No lessons to decay".into());
    }
    match memory.lessons.decay_unused(0.98, 0.3) {
        Ok(evicted) => (true, format!("evicted={}, remaining={}", evicted, memory.lessons.count())),
        Err(e) => (false, format!("{}", e)),
    }
}

async fn run_memory_consolidate(state: &AppState) -> (bool, String) {
    let memory = state.memory.read().await;
    let summary = memory.status_summary();
    (true, format!("status: {}", summary))
}

async fn run_snapshot_capture() -> (bool, String) {
    let dir = std::path::Path::new("data/snapshots");
    if let Err(e) = std::fs::create_dir_all(dir) {
        return (false, format!("Failed to create snapshots dir: {}", e));
    }

    let snapshot = crate::interpretability::NeuralSnapshot {
        id: uuid::Uuid::new_v4().to_string(),
        timestamp: chrono::Utc::now(),
        top_features: Vec::new(),
        context_summary: "Scheduled snapshot capture".into(),
        divergence_from_baseline: 0.0,
    };

    let filename = format!("snapshot_{}.json", snapshot.timestamp.format("%Y%m%d_%H%M%S"));
    let path = dir.join(&filename);
    match std::fs::write(&path, serde_json::to_string_pretty(&snapshot).unwrap_or_default()) {
        Ok(_) => (true, format!("Saved {}", filename)),
        Err(e) => (false, format!("Write failed: {}", e)),
    }
}

async fn run_synaptic_prune(state: &AppState) -> (bool, String) {
    let mut memory = state.memory.write().await;
    memory.synaptic.decay_all(0.95);
    let edges = memory.synaptic.edge_count();
    (true, format!("edges_remaining={}", edges))
}

async fn run_buffer_flush(state: &AppState) -> (bool, String) {
    let golden = state.golden_buffer.read().await;
    let rejection = state.rejection_buffer.read().await;
    (true, format!("golden={}, rejection={}", golden.count(), rejection.count()))
}

async fn run_log_rotate() -> (bool, String) {
    let log_dir = std::path::Path::new("data/logs");
    if !log_dir.exists() {
        return (true, "No log directory".into());
    }

    let mut rotated = 0usize;
    if let Ok(entries) = std::fs::read_dir(log_dir) {
        let cutoff = chrono::Utc::now() - chrono::Duration::days(7);
        for entry in entries.flatten() {
            let path = entry.path();
            if let Ok(meta) = path.metadata() {
                if let Ok(modified) = meta.modified() {
                    let modified_dt: chrono::DateTime<chrono::Utc> = modified.into();
                    if modified_dt < cutoff {
                        let _ = std::fs::remove_file(&path);
                        rotated += 1;
                    }
                }
            }
        }
    }
    (true, format!("rotated={} old files", rotated))
}

async fn run_health_check(state: &AppState) -> (bool, String) {
    let memory = state.memory.read().await;
    let golden = state.golden_buffer.read().await;
    let sessions = state.sessions.read().await;
    let status = format!(
        "memory_ok={}, lessons={}, golden={}, sessions={}",
        true,
        memory.lessons.count(),
        golden.count(),
        sessions.list().len(),
    );
    (true, status)
}

async fn run_custom_command(cmd: &str) -> (bool, String) {
    match tokio::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .output()
        .await
    {
        Ok(output) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let result = if stdout.is_empty() { stderr.to_string() } else { stdout.to_string() };
            (output.status.success(), result.chars().take(500).collect())
        }
        Err(e) => (false, format!("Exec failed: {}", e)),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_sleep_config_defaults() {
        let config = crate::learning::sleep::SleepConfig::default();
        assert_eq!(config.min_golden_samples, 10);
        assert_eq!(config.decay_factor, 0.95);
    }
}
