// Ern-OS — Tier 5: Scratchpad — pinned key-value notes

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScratchpadEntry {
    pub key: String,
    pub value: String,
    #[serde(default)]
    pub pinned: bool,
}

pub struct ScratchpadStore {
    entries: Vec<ScratchpadEntry>,
    file_path: Option<PathBuf>,
}

impl ScratchpadStore {
    pub fn new() -> Self { Self { entries: Vec::new(), file_path: None } }

    pub fn open(path: &Path) -> Result<Self> {
        tracing::info!(module = "scratchpad", fn_name = "open", "scratchpad::open called");
        let mut store = Self { entries: Vec::new(), file_path: Some(path.to_path_buf()) };
        if path.exists() {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read scratchpad: {}", path.display()))?;
            store.entries = serde_json::from_str(&content)?;
        }
        Ok(store)
    }

    fn persist(&self) -> Result<()> {
        if let Some(ref path) = self.file_path {
            if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
            std::fs::write(path, serde_json::to_string_pretty(&self.entries)?)?;
        }
        Ok(())
    }

    pub fn pin(&mut self, key: &str, value: &str) -> Result<()> {
        tracing::info!(module = "scratchpad", fn_name = "pin", "scratchpad::pin called");
        if let Some(entry) = self.entries.iter_mut().find(|e| e.key == key) {
            entry.value = value.to_string();
            entry.pinned = true;
        } else {
            self.entries.push(ScratchpadEntry {
                key: key.to_string(), value: value.to_string(), pinned: true,
            });
        }
        self.persist()
    }

    pub fn unpin(&mut self, key: &str) -> Result<()> {
        tracing::info!(module = "scratchpad", fn_name = "unpin", "scratchpad::unpin called");
        self.entries.retain(|e| e.key != key);
        self.persist()
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries.iter().find(|e| e.key == key).map(|e| e.value.as_str())
    }

    pub fn all(&self) -> &[ScratchpadEntry] { &self.entries }
    pub fn count(&self) -> usize { self.entries.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_pin_get() {
        let mut store = ScratchpadStore::new();
        store.pin("lang", "Rust").unwrap();
        assert_eq!(store.get("lang"), Some("Rust"));
    }

    #[test]
    fn test_unpin() {
        let mut store = ScratchpadStore::new();
        store.pin("k", "v").unwrap();
        store.unpin("k").unwrap();
        assert!(store.get("k").is_none());
    }

    #[test]
    fn test_persist_reload() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("scratchpad.json");
        { let mut s = ScratchpadStore::open(&path).unwrap(); s.pin("a", "b").unwrap(); }
        { let s = ScratchpadStore::open(&path).unwrap(); assert_eq!(s.count(), 1); }
    }
}
