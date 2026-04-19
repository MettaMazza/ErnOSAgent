//! Progress tracking and reporting for autonomous sessions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Snapshot of current progress during autonomous execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressReport {
    pub session_id: String,
    pub started_at: DateTime<Utc>,
    pub elapsed_minutes: u64,
    pub tasks_completed: usize,
    pub tasks_remaining: usize,
    pub tasks_failed: usize,
    pub current_task: Option<String>,
    pub files_modified: Vec<String>,
    pub errors_encountered: usize,
    pub errors_auto_fixed: usize,
}

/// Tracker that accumulates progress state during execution.
#[derive(Debug, Clone)]
pub struct ProgressTracker {
    pub session_id: String,
    pub started_at: DateTime<Utc>,
    pub tasks_completed: usize,
    pub tasks_failed: usize,
    pub current_task: Option<String>,
    pub files_modified: Vec<String>,
    pub errors_encountered: usize,
    pub errors_auto_fixed: usize,
    pub steps_since_report: usize,
}

impl ProgressTracker {
    /// Create a new tracker for a session.
    pub fn new(session_id: &str) -> Self {
        Self {
            session_id: session_id.to_string(),
            started_at: Utc::now(),
            tasks_completed: 0,
            tasks_failed: 0,
            current_task: None,
            files_modified: Vec::new(),
            errors_encountered: 0,
            errors_auto_fixed: 0,
            steps_since_report: 0,
        }
    }

    /// Record a task completion.
    pub fn task_completed(&mut self, task_name: &str) {
        self.tasks_completed += 1;
        self.steps_since_report += 1;
        tracing::info!(task = %task_name, total = self.tasks_completed, "Task completed");
    }

    /// Record a task failure.
    pub fn task_failed(&mut self, task_name: &str) {
        self.tasks_failed += 1;
        self.errors_encountered += 1;
        self.steps_since_report += 1;
        tracing::warn!(task = %task_name, total_failed = self.tasks_failed, "Task failed");
    }

    /// Record an auto-fix attempt.
    pub fn error_auto_fixed(&mut self) {
        self.errors_auto_fixed += 1;
    }

    /// Record a file modification.
    pub fn file_modified(&mut self, path: &str) {
        if !self.files_modified.contains(&path.to_string()) {
            self.files_modified.push(path.to_string());
        }
    }

    /// Set the current task being worked on.
    pub fn set_current_task(&mut self, task: &str) {
        self.current_task = Some(task.to_string());
    }

    /// Reset report counter after sending a report.
    pub fn report_sent(&mut self) {
        self.steps_since_report = 0;
    }

    /// Generate a progress report snapshot.
    pub fn snapshot(&self, tasks_remaining: usize) -> ProgressReport {
        let elapsed = Utc::now().signed_duration_since(self.started_at);
        ProgressReport {
            session_id: self.session_id.clone(),
            started_at: self.started_at,
            elapsed_minutes: elapsed.num_minutes().max(0) as u64,
            tasks_completed: self.tasks_completed,
            tasks_remaining,
            tasks_failed: self.tasks_failed,
            current_task: self.current_task.clone(),
            files_modified: self.files_modified.clone(),
            errors_encountered: self.errors_encountered,
            errors_auto_fixed: self.errors_auto_fixed,
        }
    }
}

/// Format a progress report into a human-readable string.
pub fn format_progress(report: &ProgressReport) -> String {
    let mut out = format!(
        "[Progress Report — {}min elapsed]\n\
         ✅ Completed: {} | ❌ Failed: {} | ⏳ Remaining: {}\n",
        report.elapsed_minutes,
        report.tasks_completed,
        report.tasks_failed,
        report.tasks_remaining,
    );

    if let Some(ref task) = report.current_task {
        out.push_str(&format!("🔄 Current: {}\n", task));
    }

    if !report.files_modified.is_empty() {
        out.push_str(&format!("📁 Files modified: {}\n", report.files_modified.len()));
    }

    if report.errors_auto_fixed > 0 {
        out.push_str(&format!("🔧 Auto-fixed: {}/{} errors\n",
            report.errors_auto_fixed, report.errors_encountered));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_tracker() {
        let tracker = ProgressTracker::new("sess-1");
        assert_eq!(tracker.session_id, "sess-1");
        assert_eq!(tracker.tasks_completed, 0);
        assert_eq!(tracker.steps_since_report, 0);
    }

    #[test]
    fn test_task_completed() {
        let mut tracker = ProgressTracker::new("s");
        tracker.task_completed("build db");
        assert_eq!(tracker.tasks_completed, 1);
        assert_eq!(tracker.steps_since_report, 1);
    }

    #[test]
    fn test_task_failed() {
        let mut tracker = ProgressTracker::new("s");
        tracker.task_failed("api routes");
        assert_eq!(tracker.tasks_failed, 1);
        assert_eq!(tracker.errors_encountered, 1);
    }

    #[test]
    fn test_file_modified_dedup() {
        let mut tracker = ProgressTracker::new("s");
        tracker.file_modified("src/main.rs");
        tracker.file_modified("src/main.rs");
        tracker.file_modified("src/lib.rs");
        assert_eq!(tracker.files_modified.len(), 2);
    }

    #[test]
    fn test_report_sent_resets() {
        let mut tracker = ProgressTracker::new("s");
        tracker.task_completed("a");
        tracker.task_completed("b");
        assert_eq!(tracker.steps_since_report, 2);
        tracker.report_sent();
        assert_eq!(tracker.steps_since_report, 0);
    }

    #[test]
    fn test_snapshot() {
        let mut tracker = ProgressTracker::new("s");
        tracker.task_completed("a");
        tracker.set_current_task("b");
        let report = tracker.snapshot(3);
        assert_eq!(report.tasks_completed, 1);
        assert_eq!(report.tasks_remaining, 3);
        assert_eq!(report.current_task.as_deref(), Some("b"));
    }

    #[test]
    fn test_format_progress() {
        let report = ProgressReport {
            session_id: "s".into(),
            started_at: Utc::now(),
            elapsed_minutes: 5,
            tasks_completed: 3,
            tasks_remaining: 2,
            tasks_failed: 1,
            current_task: Some("Build UI".into()),
            files_modified: vec!["a.rs".into(), "b.rs".into()],
            errors_encountered: 2,
            errors_auto_fixed: 1,
        };
        let text = format_progress(&report);
        assert!(text.contains("5min elapsed"));
        assert!(text.contains("Completed: 3"));
        assert!(text.contains("Build UI"));
        assert!(text.contains("Auto-fixed: 1/2"));
    }
}
