//! Tier 2: Consolidation Engine — context overflow → summarize → archive.

use crate::provider::Message;

pub struct ConsolidationEngine {
    count: usize,
}

impl ConsolidationEngine {
    pub fn new() -> Self {
        Self { count: 0 }
    }

    /// Check if working memory needs consolidation.
    pub fn needs_consolidation(&self, usage_pct: f32, threshold: f32) -> bool {
        usage_pct >= threshold
    }

    /// Split messages for consolidation: oldest 60% for summarization, keep freshest 40%.
    pub fn split_for_consolidation(&self, messages: &[Message]) -> (Vec<Message>, Vec<Message>) {
        let split_point = (messages.len() as f64 * 0.6) as usize;
        let to_summarize = messages[..split_point].to_vec();
        let to_keep = messages[split_point..].to_vec();
        (to_summarize, to_keep)
    }

    /// Record that a consolidation occurred.
    pub fn record_consolidation(&mut self) {
        self.count += 1;
        tracing::info!(count = self.count, "Consolidation recorded");
    }

    pub fn consolidation_count(&self) -> usize {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(content: &str) -> Message {
        Message { role: "user".to_string(), content: content.to_string(), images: Vec::new() }
    }

    #[test]
    fn test_needs_consolidation() {
        let engine = ConsolidationEngine::new();
        assert!(engine.needs_consolidation(0.85, 0.80));
        assert!(!engine.needs_consolidation(0.75, 0.80));
    }

    #[test]
    fn test_split_for_consolidation() {
        let engine = ConsolidationEngine::new();
        let messages: Vec<Message> = (0..10).map(|i| msg(&format!("msg{}", i))).collect();
        let (old, fresh) = engine.split_for_consolidation(&messages);
        assert_eq!(old.len(), 6);
        assert_eq!(fresh.len(), 4);
    }

    #[test]
    fn test_consolidation_count() {
        let mut engine = ConsolidationEngine::new();
        assert_eq!(engine.consolidation_count(), 0);
        engine.record_consolidation();
        assert_eq!(engine.consolidation_count(), 1);
    }
}
