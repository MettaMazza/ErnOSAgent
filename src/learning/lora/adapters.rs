// Ern-OS — LoRA adapter management

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A saved LoRA adapter on disk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdapterInfo {
    pub id: String,
    pub name: String,
    pub method: String,
    pub path: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub param_count: usize,
}

/// Adapter store — manages saved adapters.
pub struct AdapterStore {
    dir: PathBuf,
    adapters: Vec<AdapterInfo>,
}

impl AdapterStore {
    pub fn new(dir: &Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let mut store = Self { dir: dir.to_path_buf(), adapters: Vec::new() };
        store.scan()?;
        Ok(store)
    }

    fn scan(&mut self) -> anyhow::Result<()> {
        let manifest_path = self.dir.join("adapters.json");
        if manifest_path.exists() {
            let content = std::fs::read_to_string(&manifest_path)?;
            self.adapters = serde_json::from_str(&content)?;
        }
        Ok(())
    }

    pub fn register(&mut self, name: &str, method: &str, path: &str, params: usize) -> anyhow::Result<()> {
        self.adapters.push(AdapterInfo {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_string(), method: method.to_string(),
            path: path.to_string(), created_at: chrono::Utc::now(),
            param_count: params,
        });
        let manifest = self.dir.join("adapters.json");
        std::fs::write(&manifest, serde_json::to_string_pretty(&self.adapters)?)?;
        Ok(())
    }

    pub fn list(&self) -> &[AdapterInfo] { &self.adapters }
    pub fn count(&self) -> usize { self.adapters.len() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_register_and_list() {
        let tmp = TempDir::new().unwrap();
        let mut store = AdapterStore::new(tmp.path()).unwrap();
        store.register("sft_v1", "sft", "adapters/sft_v1.bin", 1000).unwrap();
        assert_eq!(store.count(), 1);
    }
}
