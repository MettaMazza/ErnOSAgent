// Ern-OS — Scheduler job store
//! Persistent job registry with CRUD operations and execution history.

use super::job::{CronJob, JobExecution, JobSchedule, JobTask};
use anyhow::Result;
use chrono::Utc;
use std::path::{Path, PathBuf};

/// Manages all scheduled jobs + execution history.
pub struct JobStore {
    jobs_path: PathBuf,
    history_path: PathBuf,
    pub jobs: Vec<CronJob>,
    pub history: Vec<JobExecution>,
}

impl JobStore {
    /// Load or create the job store. Merges with default system jobs on first boot.
    pub fn load(data_dir: &Path) -> Result<Self> {
        let jobs_path = data_dir.join("scheduler.json");
        let history_path = data_dir.join("scheduler_history.json");
        std::fs::create_dir_all(data_dir)?;

        let mut jobs: Vec<CronJob> = if jobs_path.exists() {
            let content = std::fs::read_to_string(&jobs_path)?;
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            Vec::new()
        };

        // Ensure all builtin jobs exist (merge on upgrade)
        for default in default_system_jobs() {
            if !jobs.iter().any(|j| j.name == default.name) {
                jobs.push(default);
            }
        }

        let history: Vec<JobExecution> = if history_path.exists() {
            let content = std::fs::read_to_string(&history_path)?;
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            Vec::new()
        };

        let store = Self { jobs_path, history_path, jobs, history };
        store.save()?;

        tracing::info!(
            jobs = store.jobs.len(),
            history = store.history.len(),
            "Scheduler store loaded"
        );
        Ok(store)
    }

    /// Persist jobs and history to disk.
    pub fn save(&self) -> Result<()> {
        std::fs::write(&self.jobs_path, serde_json::to_string_pretty(&self.jobs)?)?;
        std::fs::write(&self.history_path, serde_json::to_string_pretty(&self.history)?)?;
        Ok(())
    }

    /// Add a new job. Returns the job ID.
    pub fn add(&mut self, job: CronJob) -> Result<String> {
        let id = job.id.clone();
        self.jobs.push(job);
        self.save()?;
        Ok(id)
    }

    /// Remove a job by ID. Builtin jobs cannot be removed.
    pub fn remove(&mut self, id: &str) -> Result<bool> {
        if let Some(job) = self.jobs.iter().find(|j| j.id == id) {
            if job.builtin {
                anyhow::bail!("Cannot delete builtin job '{}'", job.name);
            }
        }
        let before = self.jobs.len();
        self.jobs.retain(|j| j.id != id);
        let removed = self.jobs.len() < before;
        if removed { self.save()?; }
        Ok(removed)
    }

    /// Toggle a job's enabled state.
    pub fn toggle(&mut self, id: &str) -> Result<bool> {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.id == id) {
            job.enabled = !job.enabled;
            let new_state = job.enabled;
            self.save()?;
            Ok(new_state)
        } else {
            anyhow::bail!("Job '{}' not found", id);
        }
    }

    /// Record a job execution and update the job's last_run.
    pub fn record_execution(&mut self, job_id: &str, exec: JobExecution) {
        if let Some(job) = self.jobs.iter_mut().find(|j| j.id == job_id) {
            job.last_run = Some(exec.started_at);
            job.last_result = Some(if exec.success { "ok".into() } else { exec.result.clone() });
            job.run_count += 1;

            // Auto-disable one-shot jobs
            if matches!(job.schedule, JobSchedule::Once(_)) {
                job.enabled = false;
            }
        }

        self.history.push(exec);

        // Cap history at 200 entries
        if self.history.len() > 200 {
            let drain = self.history.len() - 200;
            self.history.drain(..drain);
        }

        // Best-effort save
        let _ = self.save();
    }

    /// Get all jobs (for API).
    pub fn list(&self) -> &[CronJob] {
        &self.jobs
    }

    /// Get execution history (for API).
    pub fn get_history(&self) -> &[JobExecution] {
        &self.history
    }
}

/// Default system jobs created on first boot — only tasks that do real work.
fn default_system_jobs() -> Vec<CronJob> {
    let now = Utc::now();
    vec![
        CronJob {
            id: uuid::Uuid::new_v4().to_string(),
            name: "sleep_cycle".into(),
            description: "Training consolidation — drains buffers and runs LoRA training".into(),
            schedule: JobSchedule::Interval(300),
            task: JobTask::SleepCycle,
            enabled: true,
            created_at: now, last_run: None, last_result: None,
            run_count: 0, builtin: true,
        },
        CronJob {
            id: uuid::Uuid::new_v4().to_string(),
            name: "lesson_decay".into(),
            description: "Decay unused lesson confidence — Hebbian forgetting".into(),
            schedule: JobSchedule::Interval(300),
            task: JobTask::LessonDecay,
            enabled: true,
            created_at: now, last_run: None, last_result: None,
            run_count: 0, builtin: true,
        },
        CronJob {
            id: uuid::Uuid::new_v4().to_string(),
            name: "log_rotate".into(),
            description: "Clean up log files older than 7 days".into(),
            schedule: JobSchedule::Cron("0 0 0 * * *".into()),
            task: JobTask::LogRotate,
            enabled: true,
            created_at: now, last_run: None, last_result: None,
            run_count: 0, builtin: true,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_load_creates_defaults() {
        let tmp = TempDir::new().unwrap();
        let store = JobStore::load(tmp.path()).unwrap();
        assert_eq!(store.jobs.len(), 3);
        assert!(store.jobs.iter().all(|j| j.builtin));
    }

    #[test]
    fn test_add_and_remove() {
        let tmp = TempDir::new().unwrap();
        let mut store = JobStore::load(tmp.path()).unwrap();
        let custom = CronJob {
            id: "custom-1".into(),
            name: "my_job".into(),
            description: "test".into(),
            schedule: JobSchedule::Interval(60),
            task: JobTask::Custom("echo hello".into()),
            enabled: true,
            created_at: Utc::now(), last_run: None, last_result: None,
            run_count: 0, builtin: false,
        };
        store.add(custom).unwrap();
        assert_eq!(store.jobs.len(), 4);

        assert!(store.remove("custom-1").unwrap());
        assert_eq!(store.jobs.len(), 3);
    }

    #[test]
    fn test_cannot_remove_builtin() {
        let tmp = TempDir::new().unwrap();
        let mut store = JobStore::load(tmp.path()).unwrap();
        let id = store.jobs[0].id.clone();
        assert!(store.remove(&id).is_err());
    }

    #[test]
    fn test_toggle() {
        let tmp = TempDir::new().unwrap();
        let mut store = JobStore::load(tmp.path()).unwrap();
        let id = store.jobs[0].id.clone();
        assert!(store.jobs[0].enabled);
        store.toggle(&id).unwrap();
        assert!(!store.jobs.iter().find(|j| j.id == id).unwrap().enabled);
    }

    #[test]
    fn test_persistence() {
        let tmp = TempDir::new().unwrap();
        {
            let mut store = JobStore::load(tmp.path()).unwrap();
            let id = store.jobs[0].id.clone();
            store.toggle(&id).unwrap();
        }
        // Reload from disk
        let store2 = JobStore::load(tmp.path()).unwrap();
        // Should have 3 builtins (not duplicated)
        assert_eq!(store2.jobs.len(), 3);
    }
}
