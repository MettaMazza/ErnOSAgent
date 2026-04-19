// Ern-OS — Training manifest — tracks all training runs

use serde::{Deserialize, Serialize};
use anyhow::Result;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestEntry {
    pub id: String,
    pub method: String,
    pub samples: usize,
    pub final_loss: f64,
    pub adapter_path: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub struct TrainingManifest {
    entries: Vec<ManifestEntry>,
    file_path: Option<PathBuf>,
}

impl TrainingManifest {
    pub fn new() -> Self { Self { entries: Vec::new(), file_path: None } }

    pub fn open(path: &Path) -> Result<Self> {
        tracing::info!(module = "manifest", fn_name = "open", "manifest::open called");
        let mut m = Self { entries: Vec::new(), file_path: Some(path.to_path_buf()) };
        if path.exists() {
            m.entries = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        }
        Ok(m)
    }

    pub fn record(&mut self, method: &str, samples: usize, loss: f64, adapter: Option<&str>) -> Result<()> {
        tracing::info!(module = "manifest", fn_name = "record", "manifest::record called");
        self.entries.push(ManifestEntry {
            id: uuid::Uuid::new_v4().to_string(),
            method: method.to_string(), samples, final_loss: loss,
            adapter_path: adapter.map(|s| s.to_string()),
            timestamp: chrono::Utc::now(),
        });
        self.persist()
    }

    pub fn history(&self) -> &[ManifestEntry] {
        tracing::info!(module = "manifest", fn_name = "history", "manifest::history called"); &self.entries }
    pub fn total_runs(&self) -> usize { self.entries.len() }

    fn persist(&self) -> Result<()> {
        if let Some(ref path) = self.file_path {
            if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
            std::fs::write(path, serde_json::to_string_pretty(&self.entries)?)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_record() {
        let mut m = TrainingManifest::new();
        m.record("SFT", 100, 0.05, Some("adapters/sft_v1")).unwrap();
        assert_eq!(m.total_runs(), 1);
    }
}
