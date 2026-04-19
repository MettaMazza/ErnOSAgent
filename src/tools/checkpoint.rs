// Ern-OS — High-performance, model-neutral Rust AI agent engine
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Checkpoint system — auto-snapshot files before destructive operations.
//!
//! Stores copies in `data/checkpoints/` with a JSON registry.
//! Allows rollback to any snapshot and auto-prunes stale entries.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A single checkpoint entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointEntry {
    pub id: String,
    pub original_path: String,
    pub snapshot_path: String,
    pub created_at: String,
    pub size_bytes: u64,
}

/// Registry of all checkpoints, persisted as JSON.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct CheckpointRegistry {
    entries: Vec<CheckpointEntry>,
}

/// Manages file snapshots for safe rollback.
pub struct CheckpointManager {
    checkpoint_dir: PathBuf,
}

impl CheckpointManager {
    /// Create a new manager at the default location.
    pub fn new(data_dir: &Path) -> Self {
        let checkpoint_dir = data_dir.join("checkpoints");
        let _ = std::fs::create_dir_all(&checkpoint_dir);
        Self { checkpoint_dir }
    }

    /// Snapshot a file before modification. Returns the checkpoint ID.
    pub fn snapshot(&self, file_path: &Path) -> anyhow::Result<String> {
        if !file_path.exists() {
            anyhow::bail!("Cannot snapshot non-existent file: {}", file_path.display());
        }

        let id = format!("ckpt_{}", uuid::Uuid::new_v4().to_string()[..8].to_string());
        let ext = file_path.extension().and_then(|e| e.to_str()).unwrap_or("bak");
        let snapshot_name = format!("{}.{}", id, ext);
        let snapshot_path = self.checkpoint_dir.join(&snapshot_name);

        std::fs::copy(file_path, &snapshot_path)?;

        let metadata = std::fs::metadata(file_path)?;
        let entry = CheckpointEntry {
            id: id.clone(),
            original_path: file_path.to_string_lossy().to_string(),
            snapshot_path: snapshot_path.to_string_lossy().to_string(),
            created_at: chrono::Utc::now().to_rfc3339(),
            size_bytes: metadata.len(),
        };

        let mut registry = self.load_registry();
        registry.entries.push(entry);
        self.save_registry(&registry);

        tracing::debug!(id = %id, path = %file_path.display(), "Checkpoint created");
        Ok(id)
    }

    /// Rollback a file to a checkpoint.
    pub fn rollback(&self, checkpoint_id: &str) -> anyhow::Result<String> {
        let registry = self.load_registry();
        let entry = registry.entries.iter()
            .find(|e| e.id == checkpoint_id)
            .ok_or_else(|| anyhow::anyhow!("Checkpoint '{}' not found", checkpoint_id))?;

        let snapshot = Path::new(&entry.snapshot_path);
        if !snapshot.exists() {
            anyhow::bail!("Snapshot file missing: {}", entry.snapshot_path);
        }

        std::fs::copy(snapshot, &entry.original_path)?;
        tracing::info!(id = %checkpoint_id, path = %entry.original_path, "Rollback applied");
        Ok(format!("Rolled back {} to checkpoint {}", entry.original_path, checkpoint_id))
    }

    /// List all checkpoints.
    pub fn list(&self) -> Vec<CheckpointEntry> {
        self.load_registry().entries
    }

    /// Prune checkpoints older than `max_age_hours`.
    pub fn prune(&self, max_age_hours: i64) -> usize {
        let mut registry = self.load_registry();
        let cutoff = chrono::Utc::now() - chrono::Duration::hours(max_age_hours);
        let mut pruned = 0;

        registry.entries.retain(|entry| {
            let keep = chrono::DateTime::parse_from_rfc3339(&entry.created_at)
                .map(|dt| dt > cutoff)
                .unwrap_or(true);

            if !keep {
                let _ = std::fs::remove_file(&entry.snapshot_path);
                pruned += 1;
            }
            keep
        });

        self.save_registry(&registry);
        tracing::info!(pruned, remaining = registry.entries.len(), "Checkpoints pruned");
        pruned
    }

    fn registry_path(&self) -> PathBuf {
        self.checkpoint_dir.join("registry.json")
    }

    fn load_registry(&self) -> CheckpointRegistry {
        let path = self.registry_path();
        std::fs::read_to_string(&path)
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok())
            .unwrap_or_default()
    }

    fn save_registry(&self, registry: &CheckpointRegistry) {
        let path = self.registry_path();
        if let Ok(json) = serde_json::to_string_pretty(registry) {
            let _ = std::fs::write(&path, json);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn test_dir() -> PathBuf {
        let dir = PathBuf::from("target/test_checkpoints");
        let _ = fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn test_snapshot_and_list() {
        let dir = test_dir().join("snap_list");
        let mgr = CheckpointManager::new(&dir);

        let test_file = dir.join("test_snap.txt");
        fs::write(&test_file, "original content").unwrap();

        let id = mgr.snapshot(&test_file).unwrap();
        assert!(id.starts_with("ckpt_"));

        let entries = mgr.list();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].id, id);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_rollback() {
        let dir = test_dir().join("rollback");
        let mgr = CheckpointManager::new(&dir);

        let test_file = dir.join("rollback_test.txt");
        fs::write(&test_file, "original").unwrap();

        let id = mgr.snapshot(&test_file).unwrap();
        fs::write(&test_file, "modified").unwrap();
        assert_eq!(fs::read_to_string(&test_file).unwrap(), "modified");

        mgr.rollback(&id).unwrap();
        assert_eq!(fs::read_to_string(&test_file).unwrap(), "original");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_snapshot_nonexistent_file() {
        let dir = test_dir().join("nofile");
        let mgr = CheckpointManager::new(&dir);
        assert!(mgr.snapshot(Path::new("/nonexistent/file.txt")).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_prune_keeps_recent() {
        let dir = test_dir().join("prune");
        let mgr = CheckpointManager::new(&dir);

        let test_file = dir.join("prune_test.txt");
        fs::write(&test_file, "data").unwrap();
        mgr.snapshot(&test_file).unwrap();

        let pruned = mgr.prune(24);
        assert_eq!(pruned, 0); // Just created, should be kept
        assert_eq!(mgr.list().len(), 1);

        let _ = fs::remove_dir_all(&dir);
    }
}
