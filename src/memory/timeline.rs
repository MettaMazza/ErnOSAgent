//! Tier 3: Timeline Memory — verbatim session archive with vector-embedded retrieval.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEntry {
    pub session_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub transcript: String,
    pub summary: Option<String>,
}

pub struct TimelineStore {
    dir: PathBuf,
    entries: Vec<TimelineEntry>,
}

impl TimelineStore {
    pub fn new(timeline_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(timeline_dir)
            .with_context(|| format!("Failed to create timeline dir: {}", timeline_dir.display()))?;

        let mut store = Self { dir: timeline_dir.to_path_buf(), entries: Vec::new() };
        store.load_entries()?;
        Ok(store)
    }

    pub fn archive(&mut self, session_id: &str, transcript: &str) -> Result<()> {
        let entry = TimelineEntry {
            session_id: session_id.to_string(),
            timestamp: chrono::Utc::now(),
            transcript: transcript.to_string(),
            summary: None,
        };

        let filename = format!("{}_{}.json", session_id, entry.timestamp.format("%Y%m%d_%H%M%S"));
        let path = self.dir.join(filename);
        let content = serde_json::to_string_pretty(&entry)?;
        std::fs::write(&path, content)?;

        self.entries.push(entry);
        tracing::info!(session_id = %session_id, "Archived timeline entry");
        Ok(())
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    fn load_entries(&mut self) -> Result<()> {
        if !self.dir.exists() { return Ok(()); }

        for entry in std::fs::read_dir(&self.dir)? {
            let path = entry?.path();
            if path.extension().map_or(false, |ext| ext == "json") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(entry) = serde_json::from_str::<TimelineEntry>(&content) {
                        self.entries.push(entry);
                    }
                }
            }
        }

        self.entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_archive_and_count() {
        let tmp = TempDir::new().unwrap();
        let mut store = TimelineStore::new(tmp.path()).unwrap();
        assert_eq!(store.entry_count(), 0);

        store.archive("sess1", "Hello world").unwrap();
        assert_eq!(store.entry_count(), 1);
    }
}
