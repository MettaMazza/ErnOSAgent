// Ern-OS — Tier 2: Consolidation engine
// Ported from ErnOSAgent with structural improvements

use crate::provider::Message;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationRecord {
    pub id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub messages_consolidated: usize,
    pub summary: String,
    pub original_token_estimate: usize,
    pub summary_token_estimate: usize,
}

pub struct ConsolidationEngine {
    records: Vec<ConsolidationRecord>,
    file_path: Option<PathBuf>,
}

impl ConsolidationEngine {
    pub fn new() -> Self {
        Self { records: Vec::new(), file_path: None }
    }

    pub fn open(path: &Path) -> Result<Self> {
        tracing::info!(module = "consolidation", fn_name = "open", "consolidation::open called");
        let mut engine = Self { records: Vec::new(), file_path: Some(path.to_path_buf()) };
        if path.exists() {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read consolidation: {}", path.display()))?;
            engine.records = serde_json::from_str(&content)
                .with_context(|| "Failed to parse consolidation file")?;
        }
        Ok(engine)
    }

    fn persist(&self) -> Result<()> {
        if let Some(ref path) = self.file_path {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let content = serde_json::to_string_pretty(&self.records)?;
            std::fs::write(path, content)?;
        }
        Ok(())
    }

    pub fn needs_consolidation(&self, usage_pct: f32, threshold: f32) -> bool {
        tracing::info!(module = "consolidation", fn_name = "needs_consolidation", "consolidation::needs_consolidation called");
        usage_pct >= threshold
    }

    pub fn split_for_consolidation(&self, messages: &[Message]) -> (Vec<Message>, Vec<Message>) {
        if messages.len() <= 2 {
            return (Vec::new(), messages.to_vec());
        }
        let split = (messages.len() as f64 * 0.6) as usize;
        let split = split.max(1);
        (messages[..split].to_vec(), messages[split..].to_vec())
    }

    pub fn record_consolidation(
        &mut self, count: usize, summary: &str, original_chars: usize,
    ) -> Result<()> {
        tracing::info!(module = "consolidation", fn_name = "record_consolidation", "consolidation::record_consolidation called");
        self.records.push(ConsolidationRecord {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now(),
            messages_consolidated: count,
            summary: summary.to_string(),
            original_token_estimate: original_chars / 4,
            summary_token_estimate: summary.len() / 4,
        });
        self.persist()
    }

    pub fn summary_message(summary: &str) -> Message {
        Message::text("system", &format!(
            "[Memory — Consolidated Context]\n\
             The following is a compressed summary of earlier conversation:\n\n{}",
            summary
        ))
    }

    pub fn consolidation_count(&self) -> usize { self.records.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_needs_consolidation() {
        let engine = ConsolidationEngine::new();
        assert!(engine.needs_consolidation(0.85, 0.80));
        assert!(!engine.needs_consolidation(0.75, 0.80));
    }

    #[test]
    fn test_record_and_count() {
        let mut engine = ConsolidationEngine::new();
        engine.record_consolidation(5, "Summary", 2000).unwrap();
        assert_eq!(engine.consolidation_count(), 1);
    }
}
