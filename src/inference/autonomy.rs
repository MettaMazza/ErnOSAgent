//! Autonomy configuration — user-selectable levels of agent independence.
//!
//! Controls how much the agent works independently versus checking in
//! with the user. Respects governance: no hardcoded limits on capability,
//! only on reporting cadence.

use serde::{Deserialize, Serialize};

/// Level of agent autonomy.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AutonomyLevel {
    /// Ask before every action.
    Interactive,
    /// Work independently, report every N steps.
    Supervised,
    /// Full autonomy with periodic status reports.
    Autonomous,
}

impl Default for AutonomyLevel {
    fn default() -> Self {
        Self::Supervised
    }
}

/// Configuration for an autonomous session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomyConfig {
    pub level: AutonomyLevel,
    /// Maximum session duration in model turns (not wall clock).
    pub max_turns: usize,
    /// Report progress every N completed tasks.
    pub report_interval_steps: usize,
    /// Pause and ask user on any task failure.
    pub pause_on_failure: bool,
    /// Allow destructive operations (file deletion, system commands).
    pub allow_destructive: bool,
}

impl Default for AutonomyConfig {
    fn default() -> Self {
        Self {
            level: AutonomyLevel::Supervised,
            max_turns: 200,
            report_interval_steps: 5,
            pause_on_failure: true,
            allow_destructive: false,
        }
    }
}

/// Decision about whether to checkpoint/report.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointDecision {
    /// Continue working.
    Continue,
    /// Send a progress report to the user.
    ReportProgress,
    /// Pause and wait for user review.
    PauseForReview,
    /// Session has expired — stop and summarize.
    SessionExpired,
}

/// Check if the agent should pause and report.
pub fn should_checkpoint(
    config: &AutonomyConfig,
    steps_completed: usize,
    steps_since_report: usize,
    total_turns: usize,
) -> CheckpointDecision {
    if total_turns >= config.max_turns {
        return CheckpointDecision::SessionExpired;
    }

    match config.level {
        AutonomyLevel::Interactive => CheckpointDecision::PauseForReview,
        AutonomyLevel::Supervised => {
            if steps_since_report >= config.report_interval_steps {
                CheckpointDecision::ReportProgress
            } else {
                CheckpointDecision::Continue
            }
        }
        AutonomyLevel::Autonomous => {
            // Only report at major checkpoints (every 2x interval)
            let major_interval = config.report_interval_steps * 2;
            if steps_completed > 0 && steps_since_report >= major_interval {
                CheckpointDecision::ReportProgress
            } else {
                CheckpointDecision::Continue
            }
        }
    }
}

/// Check if a destructive operation is allowed under current autonomy.
pub fn is_destructive_allowed(config: &AutonomyConfig) -> bool {
    config.allow_destructive
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = AutonomyConfig::default();
        assert_eq!(config.level, AutonomyLevel::Supervised);
        assert_eq!(config.max_turns, 200);
        assert!(config.pause_on_failure);
        assert!(!config.allow_destructive);
    }

    #[test]
    fn test_interactive_always_pauses() {
        let config = AutonomyConfig { level: AutonomyLevel::Interactive, ..Default::default() };
        assert_eq!(should_checkpoint(&config, 1, 1, 0), CheckpointDecision::PauseForReview);
    }

    #[test]
    fn test_supervised_reports_at_interval() {
        let config = AutonomyConfig {
            level: AutonomyLevel::Supervised,
            report_interval_steps: 3,
            ..Default::default()
        };
        assert_eq!(should_checkpoint(&config, 3, 2, 0), CheckpointDecision::Continue);
        assert_eq!(should_checkpoint(&config, 3, 3, 0), CheckpointDecision::ReportProgress);
    }

    #[test]
    fn test_autonomous_reports_at_double_interval() {
        let config = AutonomyConfig {
            level: AutonomyLevel::Autonomous,
            report_interval_steps: 5,
            ..Default::default()
        };
        assert_eq!(should_checkpoint(&config, 5, 5, 0), CheckpointDecision::Continue);
        assert_eq!(should_checkpoint(&config, 10, 10, 0), CheckpointDecision::ReportProgress);
    }

    #[test]
    fn test_session_expired() {
        let config = AutonomyConfig { max_turns: 10, ..Default::default() };
        assert_eq!(should_checkpoint(&config, 0, 0, 10), CheckpointDecision::SessionExpired);
        assert_eq!(should_checkpoint(&config, 0, 0, 15), CheckpointDecision::SessionExpired);
    }

    #[test]
    fn test_destructive_allowed() {
        let config = AutonomyConfig { allow_destructive: true, ..Default::default() };
        assert!(is_destructive_allowed(&config));
        let config2 = AutonomyConfig::default();
        assert!(!is_destructive_allowed(&config2));
    }

    #[test]
    fn test_autonomy_level_default() {
        assert_eq!(AutonomyLevel::default(), AutonomyLevel::Supervised);
    }
}
