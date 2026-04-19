// Ern-OS — Rejection buffer for preference learning

use serde::{Deserialize, Serialize};
use anyhow::Result;
use std::path::{Path, PathBuf};

/// A preference pair: chosen (approved) vs rejected response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreferencePair {
    pub id: String,
    pub input: String,
    pub chosen: String,
    pub rejected: String,
    pub rejection_reason: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Rejection buffer — stores preference pairs for DPO/ORPO training.
pub struct RejectionBuffer {
    pairs: Vec<PreferencePair>,
    file_path: Option<PathBuf>,
}

impl RejectionBuffer {
    pub fn new() -> Self { Self { pairs: Vec::new(), file_path: None } }

    pub fn open(path: &Path) -> Result<Self> {
        tracing::info!(module = "rejection_buffer", fn_name = "open", "rejection_buffer::open called");
        let mut buf = Self { pairs: Vec::new(), file_path: Some(path.to_path_buf()) };
        if path.exists() {
            let content = std::fs::read_to_string(path)?;
            buf.pairs = serde_json::from_str(&content)?;
        }
        Ok(buf)
    }

    pub fn add_pair(&mut self, input: &str, chosen: &str, rejected: &str, reason: &str) -> Result<()> {
        tracing::info!(module = "rejection_buffer", fn_name = "add_pair", "rejection_buffer::add_pair called");
        self.pairs.push(PreferencePair {
            id: uuid::Uuid::new_v4().to_string(),
            input: input.to_string(), chosen: chosen.to_string(),
            rejected: rejected.to_string(), rejection_reason: reason.to_string(),
            timestamp: chrono::Utc::now(),
        });
        self.persist()
    }

    pub fn drain_all(&mut self) -> Vec<PreferencePair> {
        tracing::info!(module = "rejection_buffer", fn_name = "drain_all", "rejection_buffer::drain_all called");
        std::mem::take(&mut self.pairs)
    }

    pub fn count(&self) -> usize { self.pairs.len() }

    fn persist(&self) -> Result<()> {
        if let Some(ref path) = self.file_path {
            if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
            std::fs::write(path, serde_json::to_string(&self.pairs)?)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_pair() {
        let mut buf = RejectionBuffer::new();
        buf.add_pair("q", "good answer", "bad answer", "too short").unwrap();
        assert_eq!(buf.count(), 1);
    }

    #[test]
    fn test_drain_all() {
        let mut buf = RejectionBuffer::new();
        buf.add_pair("q", "good", "bad", "reason").unwrap();
        let pairs = buf.drain_all();
        assert_eq!(pairs.len(), 1);
        assert_eq!(buf.count(), 0);
    }
}
