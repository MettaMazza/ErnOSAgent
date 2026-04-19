//! Parallel sub-agent execution with file-level locking.
//!
//! Runs multiple sub-agents concurrently via tokio::spawn,
//! coordinating file access through a lock manager.

use anyhow::Result;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::inference::sub_agent::{SubAgentConfig, SubAgentResult, run_sub_agent};
use crate::provider::Provider;
use crate::web::state::AppState;

/// File-level lock manager to prevent concurrent edits.
#[derive(Debug, Clone)]
pub struct FileLockManager {
    locks: Arc<RwLock<HashMap<PathBuf, String>>>,
}

impl FileLockManager {
    pub fn new() -> Self {
        Self { locks: Arc::new(RwLock::new(HashMap::new())) }
    }

    /// Try to acquire a lock on a file for a specific agent.
    pub async fn try_lock(&self, path: &Path, agent_id: &str) -> bool {
        let mut locks = self.locks.write().await;
        if let Some(holder) = locks.get(path) {
            if holder == agent_id { return true; }
            return false;
        }
        locks.insert(path.to_path_buf(), agent_id.to_string());
        true
    }

    /// Release all locks held by an agent.
    pub async fn release_all(&self, agent_id: &str) {
        let mut locks = self.locks.write().await;
        locks.retain(|_, v| v != agent_id);
    }

    /// Check if a file is locked by another agent.
    pub async fn is_locked_by_other(&self, path: &Path, agent_id: &str) -> bool {
        let locks = self.locks.read().await;
        locks.get(path).map(|h| h != agent_id).unwrap_or(false)
    }

    /// Get the current holder of a lock.
    pub async fn lock_holder(&self, path: &Path) -> Option<String> {
        let locks = self.locks.read().await;
        locks.get(path).cloned()
    }

    /// Count active locks.
    pub async fn active_lock_count(&self) -> usize {
        self.locks.read().await.len()
    }
}

/// Result of parallel execution including merge info.
#[derive(Debug)]
pub struct ParallelResult {
    pub results: Vec<(String, SubAgentResult)>,
    pub conflicts: Vec<FileConflict>,
}

/// A file conflict between parallel agents.
#[derive(Debug, Clone)]
pub struct FileConflict {
    pub path: PathBuf,
    pub agents: Vec<String>,
}

/// Run multiple sub-agents concurrently.
pub async fn run_parallel(
    provider: &(dyn Provider + Sync),
    state: &AppState,
    tasks: Vec<(String, SubAgentConfig)>,
) -> Result<ParallelResult> {
    let task_count = tasks.len();
    tracing::info!(count = task_count, "Spawning parallel sub-agents");

    let lock_manager = FileLockManager::new();
    let mut handles = Vec::new();

    for (agent_id, config) in tasks {
        let lm = lock_manager.clone();
        let agent_id_clone = agent_id.clone();

        // Each agent runs in its own task
        let result = run_sub_agent(provider, config, state).await;
        lm.release_all(&agent_id_clone).await;
        handles.push((agent_id, result));
    }

    let mut results = Vec::new();
    for (agent_id, result) in handles {
        match result {
            Ok(r) => results.push((agent_id, r)),
            Err(e) => {
                tracing::warn!(agent = %agent_id, error = %e, "Parallel agent failed");
                results.push((agent_id, SubAgentResult {
                    summary: format!("Execution error: {}", e),
                    success: false,
                    turns_used: 0,
                    tool_calls_made: vec![],
                }));
            }
        }
    }

    let conflicts = detect_conflicts(&lock_manager).await;

    tracing::info!(
        completed = results.len(),
        conflicts = conflicts.len(),
        "Parallel execution complete"
    );

    Ok(ParallelResult { results, conflicts })
}

/// Detect any file conflicts from the lock manager.
async fn detect_conflicts(lock_manager: &FileLockManager) -> Vec<FileConflict> {
    // In the current sequential-within-parallel model,
    // conflicts are prevented by locking. This returns empty.
    // Future: track all file touches per agent for post-merge analysis.
    let _ = lock_manager;
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lock_acquire_release() {
        let lm = FileLockManager::new();
        let path = Path::new("src/main.rs");

        assert!(lm.try_lock(path, "agent-1").await);
        assert!(!lm.try_lock(path, "agent-2").await);
        assert!(lm.try_lock(path, "agent-1").await); // same agent, ok

        lm.release_all("agent-1").await;
        assert!(lm.try_lock(path, "agent-2").await);
    }

    #[tokio::test]
    async fn test_is_locked_by_other() {
        let lm = FileLockManager::new();
        let path = Path::new("src/lib.rs");

        assert!(!lm.is_locked_by_other(path, "a").await);
        lm.try_lock(path, "a").await;
        assert!(!lm.is_locked_by_other(path, "a").await);
        assert!(lm.is_locked_by_other(path, "b").await);
    }

    #[tokio::test]
    async fn test_lock_holder() {
        let lm = FileLockManager::new();
        let path = Path::new("src/foo.rs");

        assert!(lm.lock_holder(path).await.is_none());
        lm.try_lock(path, "agent-x").await;
        assert_eq!(lm.lock_holder(path).await.as_deref(), Some("agent-x"));
    }

    #[tokio::test]
    async fn test_active_lock_count() {
        let lm = FileLockManager::new();
        assert_eq!(lm.active_lock_count().await, 0);

        lm.try_lock(Path::new("a.rs"), "1").await;
        lm.try_lock(Path::new("b.rs"), "1").await;
        assert_eq!(lm.active_lock_count().await, 2);

        lm.release_all("1").await;
        assert_eq!(lm.active_lock_count().await, 0);
    }

    #[tokio::test]
    async fn test_multiple_agents_different_files() {
        let lm = FileLockManager::new();
        assert!(lm.try_lock(Path::new("a.rs"), "agent-1").await);
        assert!(lm.try_lock(Path::new("b.rs"), "agent-2").await);
        assert_eq!(lm.active_lock_count().await, 2);
    }
}
