// Ern-OS — Tier 3: Timeline — verbatim session archive

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelineEntry {
    pub session_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub transcript: String,
}

pub struct TimelineStore {
    dir: PathBuf,
    entries: Vec<TimelineEntry>,
}

impl TimelineStore {
    pub fn new(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create timeline dir: {}", dir.display()))?;
        let mut store = Self { dir: dir.to_path_buf(), entries: Vec::new() };
        store.load_entries()?;
        Ok(store)
    }

    pub fn archive(&mut self, session_id: &str, transcript: &str) -> Result<()> {
        tracing::info!(module = "timeline", fn_name = "archive", "timeline::archive called");
        let entry = TimelineEntry {
            session_id: session_id.to_string(),
            timestamp: chrono::Utc::now(),
            transcript: transcript.to_string(),
        };
        let filename = format!(
            "{}_{}_{}.json",
            entry.timestamp.format("%Y%m%d_%H%M%S_%3f"),
            &session_id[..session_id.len().min(8)],
            &uuid::Uuid::new_v4().to_string()[..6]
        );
        let path = self.dir.join(filename);
        std::fs::write(&path, serde_json::to_string_pretty(&entry)?)?;
        self.entries.push(entry);
        self.entries.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        Ok(())
    }

    pub fn recent(&self, n: usize) -> &[TimelineEntry] {
        tracing::info!(module = "timeline", fn_name = "recent", "timeline::recent called");
        &self.entries[..n.min(self.entries.len())]
    }

    pub fn search(&self, query: &str, limit: usize) -> Vec<&TimelineEntry> {
        let q = query.to_lowercase();
        self.entries.iter()
            .filter(|e| e.transcript.to_lowercase().contains(&q))
            .take(limit)
            .collect()
    }

    pub fn entry_count(&self) -> usize {
        tracing::info!(module = "timeline", fn_name = "entry_count", "timeline::entry_count called"); self.entries.len() }

    pub fn clear_entries(&mut self) { self.entries.clear(); }

    fn load_entries(&mut self) -> Result<()> {
        if !self.dir.exists() { return Ok(()); }
        for entry in std::fs::read_dir(&self.dir)? {
            let path = entry?.path();
            if path.extension().map_or(false, |e| e == "json") {
                if let Ok(content) = std::fs::read_to_string(&path) {
                    if let Ok(e) = serde_json::from_str::<TimelineEntry>(&content) {
                        self.entries.push(e);
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
        store.archive("s1", "Hello").unwrap();
        assert_eq!(store.entry_count(), 1);
    }

    #[test]
    fn test_search() {
        let tmp = TempDir::new().unwrap();
        let mut store = TimelineStore::new(tmp.path()).unwrap();
        store.archive("s1", "Rust borrow checker").unwrap();
        store.archive("s1", "Python GIL").unwrap();
        assert_eq!(store.search("rust", 10).len(), 1);
    }
}
