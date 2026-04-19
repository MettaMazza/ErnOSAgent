//! Checkpoint restore — roll back system state from a snapshot.

use anyhow::{Context, Result};
use std::path::Path;

use super::snapshot::{self, Checkpoint};

/// Result of a checkpoint restore operation.
#[derive(Debug, Clone)]
pub struct RestoreResult {
    pub success: bool,
    pub restored_commit: String,
    pub code_changed: bool,
    pub memory_restored: bool,
    pub sessions_restored: bool,
    pub message: String,
}

/// Restore system state from a checkpoint.
pub async fn restore_checkpoint(
    data_dir: &Path, checkpoint_id: &str,
) -> Result<RestoreResult> {
    let checkpoints = snapshot::list_checkpoints(data_dir)?;
    let checkpoint = checkpoints.iter()
        .find(|c| c.id == checkpoint_id)
        .context("Checkpoint not found")?;

    tracing::info!(
        id = %checkpoint.id, label = %checkpoint.label,
        commit = %checkpoint.git_commit,
        "Restoring checkpoint"
    );

    // 1. Stash current changes
    let _ = run_git(&["stash", "--include-untracked"]).await;

    // 2. Restore git state
    let code_changed = restore_git_state(&checkpoint.git_commit).await;

    // 3. Restore memory
    let memory_restored = restore_archive(&checkpoint.memory_archive, data_dir, "memory").await;

    // 4. Restore sessions
    let sessions_restored = restore_archive(&checkpoint.sessions_archive, data_dir, "sessions").await;

    // 5. Restore config if present
    restore_config(data_dir, &checkpoint.config_json);

    let result = RestoreResult {
        success: true,
        restored_commit: checkpoint.git_commit.clone(),
        code_changed,
        memory_restored,
        sessions_restored,
        message: format!(
            "Restored to checkpoint '{}' (commit {})",
            checkpoint.label, &checkpoint.git_commit[..7.min(checkpoint.git_commit.len())]
        ),
    };

    tracing::info!(
        code = code_changed, memory = memory_restored, sessions = sessions_restored,
        "Checkpoint restore complete"
    );

    Ok(result)
}

/// Checkout a specific git commit.
async fn restore_git_state(commit: &str) -> bool {
    let output = run_git(&["checkout", commit]).await;
    output
}

/// Extract a tar.gz archive over a data subdirectory.
async fn restore_archive(
    archive: &Path, data_dir: &Path, subdir: &str,
) -> bool {
    if !archive.exists() || std::fs::metadata(archive).map(|m| m.len() == 0).unwrap_or(true) {
        return false;
    }

    let target = data_dir.join(subdir);
    let _ = std::fs::remove_dir_all(&target);
    let _ = std::fs::create_dir_all(&target);

    let result = tokio::process::Command::new("tar")
        .args(["xzf"])
        .arg(archive)
        .arg("-C")
        .arg(data_dir)
        .output()
        .await;

    match result {
        Ok(output) => output.status.success(),
        Err(e) => {
            tracing::warn!(error = %e, "Archive extraction failed");
            false
        }
    }
}

/// Restore config file.
fn restore_config(data_dir: &Path, config_json: &str) {
    if config_json.is_empty() { return; }
    let config_path = data_dir.join("../ern-os.toml");
    let _ = std::fs::write(&config_path, config_json);
}

/// Run a git command and return success status.
async fn run_git(args: &[&str]) -> bool {
    tokio::process::Command::new("git")
        .args(args)
        .output()
        .await
        .map(|o| o.status.success())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_restore_result_fields() {
        let result = RestoreResult {
            success: true,
            restored_commit: "abc123".into(),
            code_changed: true,
            memory_restored: true,
            sessions_restored: false,
            message: "restored".into(),
        };
        assert!(result.success);
        assert!(result.code_changed);
        assert!(!result.sessions_restored);
    }

    #[test]
    fn test_restore_config_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        restore_config(tmp.path(), ""); // should not crash
    }

    #[tokio::test]
    async fn test_restore_nonexistent_checkpoint() {
        let tmp = tempfile::TempDir::new().unwrap();
        let result = restore_checkpoint(tmp.path(), "fake-id").await;
        assert!(result.is_err());
    }
}
