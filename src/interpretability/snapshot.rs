// Ern-OS — Neural state snapshots

use super::NeuralSnapshot;
use anyhow::Result;
use std::path::{Path, PathBuf};

/// Snapshot store — persists neural state captures to disk.
pub struct SnapshotStore {
    dir: PathBuf,
    snapshots: Vec<NeuralSnapshot>,
}

impl SnapshotStore {
    pub fn new(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)?;
        let mut store = Self { dir: dir.to_path_buf(), snapshots: Vec::new() };
        store.load()?;
        Ok(store)
    }

    pub fn capture(&mut self, top_features: Vec<(usize, f32)>, context: &str, divergence: f32) -> Result<String> {
        let snapshot = NeuralSnapshot {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            top_features,
            context_summary: context.to_string(),
            divergence_from_baseline: divergence,
        };
        let path = self.dir.join(format!("{}.json", snapshot.id));
        std::fs::write(&path, serde_json::to_string_pretty(&snapshot)?)?;
        let id = snapshot.id.clone();
        self.snapshots.push(snapshot);
        Ok(id)
    }

    pub fn recent(&self, n: usize) -> &[NeuralSnapshot] {
        let start = self.snapshots.len().saturating_sub(n);
        &self.snapshots[start..]
    }

    pub fn count(&self) -> usize { self.snapshots.len() }

    fn load(&mut self) -> Result<()> {
        if !self.dir.exists() { return Ok(()); }
        for entry in std::fs::read_dir(&self.dir)? {
            let path = entry?.path();
            if path.extension().map_or(false, |e| e == "json") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(s) = serde_json::from_str::<NeuralSnapshot>(&content) {
                        self.snapshots.push(s);
                    }
                }
            }
        }
        self.snapshots.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_capture_and_count() {
        let tmp = TempDir::new().unwrap();
        let mut store = SnapshotStore::new(tmp.path()).unwrap();
        store.capture(vec![(0, 2.5), (5, 1.8)], "test context", 0.3).unwrap();
        assert_eq!(store.count(), 1);
    }
}
