//! Checkpoint snapshot — create atomic state captures.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A complete system state checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub id: String,
    pub label: String,
    pub created_at: DateTime<Utc>,
    pub git_commit: String,
    pub git_dirty: bool,
    pub memory_archive: PathBuf,
    pub sessions_archive: PathBuf,
    pub config_json: String,
}

/// Create an atomic checkpoint of the entire system state.
pub async fn create_checkpoint(label: &str, data_dir: &Path) -> Result<Checkpoint> {
    let checkpoint_id = uuid::Uuid::new_v4().to_string();
    let checkpoint_dir = data_dir.join("checkpoints").join(&checkpoint_id);
    std::fs::create_dir_all(&checkpoint_dir)
        .context("Failed to create checkpoint directory")?;

    let git_commit = capture_git_commit().await;
    let git_dirty = check_git_dirty().await;
    let memory_archive = archive_directory(data_dir, "memory", &checkpoint_dir).await?;
    let sessions_archive = archive_directory(data_dir, "sessions", &checkpoint_dir).await?;
    let config_json = capture_config(data_dir)?;

    let checkpoint = Checkpoint {
        id: checkpoint_id,
        label: label.to_string(),
        created_at: Utc::now(),
        git_commit,
        git_dirty,
        memory_archive,
        sessions_archive,
        config_json,
    };

    save_checkpoint_metadata(&checkpoint_dir, &checkpoint)?;

    tracing::info!(
        id = %checkpoint.id, label = %checkpoint.label,
        commit = %checkpoint.git_commit,
        "Checkpoint created"
    );

    Ok(checkpoint)
}

/// List all available checkpoints, sorted by creation date (newest first).
pub fn list_checkpoints(data_dir: &Path) -> Result<Vec<Checkpoint>> {
    let dir = data_dir.join("checkpoints");
    if !dir.exists() { return Ok(Vec::new()); }

    let mut checkpoints = Vec::new();
    for entry in std::fs::read_dir(&dir)? {
        let path = entry?.path();
        let meta_path = path.join("checkpoint.json");
        if meta_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&meta_path) {
                if let Ok(cp) = serde_json::from_str::<Checkpoint>(&content) {
                    checkpoints.push(cp);
                }
            }
        }
    }
    checkpoints.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(checkpoints)
}

/// Delete a checkpoint by ID.
pub fn delete_checkpoint(data_dir: &Path, checkpoint_id: &str) -> Result<()> {
    let dir = data_dir.join("checkpoints").join(checkpoint_id);
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
        tracing::info!(id = %checkpoint_id, "Checkpoint deleted");
    }
    Ok(())
}

/// Capture current git HEAD commit hash.
async fn capture_git_commit() -> String {
    tokio::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output().await
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

/// Check if working tree has uncommitted changes.
async fn check_git_dirty() -> bool {
    tokio::process::Command::new("git")
        .args(["diff", "--quiet", "HEAD"])
        .output().await
        .map(|o| !o.status.success())
        .unwrap_or(true)
}

/// Archive a data subdirectory as tar.gz.
async fn archive_directory(
    data_dir: &Path, subdir: &str, checkpoint_dir: &Path,
) -> Result<PathBuf> {
    let source = data_dir.join(subdir);
    let archive_name = format!("{}.tar.gz", subdir);
    let archive_path = checkpoint_dir.join(&archive_name);

    if !source.exists() {
        std::fs::write(&archive_path, b"")?;
        return Ok(archive_path);
    }

    let output = tokio::process::Command::new("tar")
        .args(["czf"])
        .arg(&archive_path)
        .arg("-C")
        .arg(data_dir)
        .arg(subdir)
        .output()
        .await
        .context("Failed to create archive")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("tar failed: {}", stderr);
    }
    Ok(archive_path)
}

/// Capture current config as JSON.
fn capture_config(data_dir: &Path) -> Result<String> {
    let config_path = data_dir.join("../ern-os.toml");
    if config_path.exists() {
        std::fs::read_to_string(&config_path).context("Failed to read config")
    } else {
        Ok(String::new())
    }
}

/// Save checkpoint metadata to disk.
fn save_checkpoint_metadata(dir: &Path, checkpoint: &Checkpoint) -> Result<()> {
    let path = dir.join("checkpoint.json");
    let json = serde_json::to_string_pretty(checkpoint)?;
    std::fs::write(&path, json)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_checkpoints_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let result = list_checkpoints(tmp.path()).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_checkpoint_serialization() {
        let cp = Checkpoint {
            id: "test-id".into(),
            label: "test label".into(),
            created_at: Utc::now(),
            git_commit: "abc123".into(),
            git_dirty: false,
            memory_archive: PathBuf::from("memory.tar.gz"),
            sessions_archive: PathBuf::from("sessions.tar.gz"),
            config_json: "{}".into(),
        };
        let json = serde_json::to_string(&cp).unwrap();
        let parsed: Checkpoint = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "test-id");
        assert_eq!(parsed.label, "test label");
    }

    #[test]
    fn test_delete_nonexistent() {
        let tmp = tempfile::TempDir::new().unwrap();
        let result = delete_checkpoint(tmp.path(), "nonexistent");
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_create_and_list() {
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("memory")).unwrap();
        std::fs::create_dir_all(tmp.path().join("sessions")).unwrap();

        let cp = create_checkpoint("test checkpoint", tmp.path()).await.unwrap();
        assert_eq!(cp.label, "test checkpoint");

        let list = list_checkpoints(tmp.path()).unwrap();
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].id, cp.id);
    }
}
