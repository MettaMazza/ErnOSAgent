// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Scheduled job definitions — cron, one-off, and interval jobs.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// A scheduled job with its execution metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScheduledJob {
    /// Unique job identifier.
    pub id: String,
    /// Human-readable name (e.g. "Morning briefing").
    pub name: String,
    /// Natural language instruction for the ReAct loop.
    pub instruction: String,
    /// When/how often to run.
    pub schedule: JobSchedule,
    /// Whether the job is currently enabled.
    pub enabled: bool,
    /// When the job was created.
    pub created_at: DateTime<Utc>,
    /// Last execution timestamp.
    pub last_run: Option<DateTime<Utc>>,
    /// Result of the last execution.
    pub last_result: Option<JobResult>,
    /// Where to deliver results ("web", or a platform channel ID).
    pub delivery_channel: Option<String>,
}

/// Schedule type — how the job triggers.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum JobSchedule {
    /// Standard cron expression, e.g. "0 9 * * *" (9am daily).
    Cron(String),
    /// Run once at a specific time.
    Once(DateTime<Utc>),
    /// Run every N seconds.
    Interval(u64),
    /// Run when user has been idle for N seconds.
    Idle(u64),
}

/// Result of a single job execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobResult {
    pub success: bool,
    pub output: String,
    pub duration_ms: u64,
    pub timestamp: DateTime<Utc>,
}

/// Request body for creating or updating a job via the API.
#[derive(Debug, Deserialize)]
pub struct CreateJobRequest {
    pub name: String,
    pub instruction: String,
    pub schedule: JobSchedule,
    pub delivery_channel: Option<String>,
}

impl ScheduledJob {
    /// Create a new enabled job.
    pub fn new(name: String, instruction: String, schedule: JobSchedule) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name,
            instruction,
            schedule,
            enabled: true,
            created_at: Utc::now(),
            last_run: None,
            last_result: None,
            delivery_channel: None,
        }
    }

    /// Check if this job is due to execute at the given time.
    ///
    /// For `Idle` schedule type, an `idle_since` timestamp must be provided
    /// via `is_due_with_idle`. This method treats Idle jobs as never due.
    pub fn is_due(&self, now: DateTime<Utc>) -> bool {
        if !self.enabled {
            return false;
        }

        match &self.schedule {
            JobSchedule::Cron(expr) => {
                let schedule = match cron::Schedule::from_str(expr) {
                    Ok(s) => s,
                    Err(_) => return false,
                };
                if let Some(next) = schedule.upcoming(Utc).next() {
                    let diff = (next - now).num_seconds().abs();
                    diff == 0
                } else {
                    false
                }
            }
            JobSchedule::Once(at) => {
                if self.last_run.is_some() {
                    return false;
                }
                let diff = (*at - now).num_seconds().abs();
                diff <= 1
            }
            JobSchedule::Interval(secs) => {
                let interval = Duration::from_secs(*secs);
                match self.last_run {
                    Some(last) => {
                        let elapsed = (now - last).to_std().unwrap_or(Duration::ZERO);
                        elapsed >= interval
                    }
                    None => true,
                }
            }
            // Idle jobs require the idle timer — use is_due_idle()
            JobSchedule::Idle(_) => false,
        }
    }

    /// Check if an idle-type job is due based on the idle timer.
    pub fn is_due_idle(&self, idle_elapsed: Duration) -> bool {
        if !self.enabled {
            return false;
        }
        match &self.schedule {
            JobSchedule::Idle(threshold_secs) => {
                let threshold = Duration::from_secs(*threshold_secs);
                if idle_elapsed < threshold {
                    return false;
                }
                // Don't re-fire within the threshold window after last run
                match self.last_run {
                    Some(last) => {
                        let since_last = (Utc::now() - last)
                            .to_std()
                            .unwrap_or(Duration::ZERO);
                        since_last >= threshold
                    }
                    None => true,
                }
            }
            _ => false,
        }
    }
}

use std::str::FromStr;

/// Validate a cron expression without executing it.
pub fn validate_cron(expr: &str) -> Result<(), String> {
    cron::Schedule::from_str(expr)
        .map(|_| ())
        .map_err(|e| format!("Invalid cron expression: {}", e))
}
