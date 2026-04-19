// Ern-OS — Scheduler job data model
//! CronJob, JobSchedule, JobTask, and JobHistory types.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A scheduled job in the cron engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CronJob {
    pub id: String,
    pub name: String,
    pub description: String,
    pub schedule: JobSchedule,
    pub task: JobTask,
    pub enabled: bool,
    pub created_at: DateTime<Utc>,
    pub last_run: Option<DateTime<Utc>>,
    pub last_result: Option<String>,
    pub run_count: u64,
    /// System jobs cannot be deleted (only toggled).
    pub builtin: bool,
}

/// When a job should run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobSchedule {
    /// Standard cron expression: "*/5 * * * * *" (sec min hour dom month dow)
    Cron(String),
    /// Run every N seconds.
    Interval(u64),
    /// Run once at a specific time.
    Once(DateTime<Utc>),
}

/// What a job does when triggered.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JobTask {
    SleepCycle,
    LessonDecay,
    MemoryConsolidate,
    SnapshotCapture,
    SynapticPrune,
    BufferFlush,
    LogRotate,
    HealthCheck,
    /// User-defined shell command — unrestricted per No-Limits governance.
    Custom(String),
}

impl std::fmt::Display for JobTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SleepCycle => write!(f, "sleep_cycle"),
            Self::LessonDecay => write!(f, "lesson_decay"),
            Self::MemoryConsolidate => write!(f, "memory_consolidate"),
            Self::SnapshotCapture => write!(f, "snapshot_capture"),
            Self::SynapticPrune => write!(f, "synaptic_prune"),
            Self::BufferFlush => write!(f, "buffer_flush"),
            Self::LogRotate => write!(f, "log_rotate"),
            Self::HealthCheck => write!(f, "health_check"),
            Self::Custom(cmd) => write!(f, "custom: {}", cmd),
        }
    }
}

impl std::fmt::Display for JobSchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cron(expr) => write!(f, "cron: {}", expr),
            Self::Interval(secs) => {
                if *secs >= 3600 {
                    write!(f, "every {}h", secs / 3600)
                } else if *secs >= 60 {
                    write!(f, "every {}m", secs / 60)
                } else {
                    write!(f, "every {}s", secs)
                }
            }
            Self::Once(dt) => write!(f, "once: {}", dt.format("%Y-%m-%d %H:%M")),
        }
    }
}

/// A record of a job execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobExecution {
    pub job_id: String,
    pub job_name: String,
    pub task: String,
    pub started_at: DateTime<Utc>,
    pub duration_ms: u64,
    pub success: bool,
    pub result: String,
}

impl CronJob {
    /// Check if this job is due to run at the given time.
    pub fn is_due(&self, now: DateTime<Utc>, last_tick: DateTime<Utc>) -> bool {
        if !self.enabled {
            return false;
        }

        match &self.schedule {
            JobSchedule::Interval(secs) => {
                let since = self.last_run.unwrap_or(self.created_at);
                let elapsed = now.signed_duration_since(since).num_seconds();
                elapsed >= *secs as i64
            }
            JobSchedule::Cron(expr) => {
                use std::str::FromStr;
                if let Ok(schedule) = cron::Schedule::from_str(expr) {
                    // Check if any scheduled time falls between last_tick and now
                    schedule.after(&last_tick)
                        .take_while(|t| *t <= now)
                        .next()
                        .is_some()
                } else {
                    tracing::warn!(expr, job = %self.name, "Invalid cron expression — skipping");
                    false
                }
            }
            JobSchedule::Once(at) => {
                if self.last_run.is_some() {
                    false // Already ran
                } else {
                    now >= *at
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_job(schedule: JobSchedule) -> CronJob {
        CronJob {
            id: "test".into(),
            name: "test_job".into(),
            description: "test".into(),
            schedule,
            task: JobTask::HealthCheck,
            enabled: true,
            created_at: Utc::now() - chrono::Duration::hours(1),
            last_run: None,
            last_result: None,
            run_count: 0,
            builtin: false,
        }
    }

    #[test]
    fn test_interval_due() {
        let job = test_job(JobSchedule::Interval(60));
        let now = Utc::now();
        let tick = now - chrono::Duration::seconds(10);
        assert!(job.is_due(now, tick));
    }

    #[test]
    fn test_interval_not_due() {
        let mut job = test_job(JobSchedule::Interval(3600));
        job.last_run = Some(Utc::now() - chrono::Duration::seconds(10));
        let now = Utc::now();
        let tick = now - chrono::Duration::seconds(10);
        assert!(!job.is_due(now, tick));
    }

    #[test]
    fn test_once_due() {
        let past = Utc::now() - chrono::Duration::minutes(5);
        let job = test_job(JobSchedule::Once(past));
        let now = Utc::now();
        let tick = now - chrono::Duration::seconds(10);
        assert!(job.is_due(now, tick));
    }

    #[test]
    fn test_once_already_ran() {
        let past = Utc::now() - chrono::Duration::minutes(5);
        let mut job = test_job(JobSchedule::Once(past));
        job.last_run = Some(Utc::now());
        let now = Utc::now();
        let tick = now - chrono::Duration::seconds(10);
        assert!(!job.is_due(now, tick));
    }

    #[test]
    fn test_disabled() {
        let mut job = test_job(JobSchedule::Interval(1));
        job.enabled = false;
        let now = Utc::now();
        let tick = now - chrono::Duration::seconds(10);
        assert!(!job.is_due(now, tick));
    }

    #[test]
    fn test_display_schedule() {
        assert_eq!(format!("{}", JobSchedule::Interval(300)), "every 5m");
        assert_eq!(format!("{}", JobSchedule::Interval(7200)), "every 2h");
        assert_eq!(format!("{}", JobSchedule::Interval(30)), "every 30s");
    }

    #[test]
    fn test_display_task() {
        assert_eq!(format!("{}", JobTask::SleepCycle), "sleep_cycle");
        assert_eq!(format!("{}", JobTask::Custom("echo hi".into())), "custom: echo hi");
    }
}
