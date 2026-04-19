// Ern-OS — Tier 6: Lessons — discovered behavioral rules

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Lesson {
    pub id: String,
    pub rule: String,
    pub source: String,
    pub confidence: f32,
    pub times_applied: usize,
}

pub struct LessonStore {
    lessons: Vec<Lesson>,
    file_path: Option<PathBuf>,
}

impl LessonStore {
    pub fn new() -> Self { Self { lessons: Vec::new(), file_path: None } }

    pub fn open(path: &Path) -> Result<Self> {
        tracing::info!(module = "lessons", fn_name = "open", "lessons::open called");
        let mut store = Self { lessons: Vec::new(), file_path: Some(path.to_path_buf()) };
        if path.exists() {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read lessons: {}", path.display()))?;
            store.lessons = serde_json::from_str(&content)?;
        }
        Ok(store)
    }

    fn persist(&self) -> Result<()> {
        if let Some(ref path) = self.file_path {
            if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
            std::fs::write(path, serde_json::to_string_pretty(&self.lessons)?)?;
        }
        Ok(())
    }

    pub fn add(&mut self, rule: &str, source: &str, confidence: f32) -> Result<()> {
        tracing::info!(module = "lessons", fn_name = "add", "lessons::add called");
        self.lessons.push(Lesson {
            id: uuid::Uuid::new_v4().to_string(),
            rule: rule.to_string(), source: source.to_string(),
            confidence, times_applied: 0,
        });
        self.persist()
    }

    pub fn remove(&mut self, id: &str) -> Result<()> {
        tracing::info!(module = "lessons", fn_name = "remove", "lessons::remove called");
        let before = self.lessons.len();
        self.lessons.retain(|l| l.id != id);
        if self.lessons.len() == before { anyhow::bail!("Lesson '{}' not found", id); }
        self.persist()
    }

    pub fn apply(&mut self, id: &str) -> Result<()> {
        if let Some(l) = self.lessons.iter_mut().find(|l| l.id == id) {
            l.times_applied += 1;
            self.persist()?;
        }
        Ok(())
    }

    pub fn high_confidence(&self, threshold: f32) -> Vec<&Lesson> {
        self.lessons.iter().filter(|l| l.confidence >= threshold).collect()
    }

    pub fn search(&self, query: &str, limit: usize) -> Vec<&Lesson> {
        let q = query.to_lowercase();
        let mut matches: Vec<&Lesson> = self.lessons.iter()
            .filter(|l| l.rule.to_lowercase().contains(&q))
            .collect();
        matches.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(limit);
        matches
    }

    pub fn all(&self) -> &[Lesson] { &self.lessons }
    pub fn count(&self) -> usize { self.lessons.len() }

    /// Add a lesson only if no semantically similar lesson exists.
    /// Returns true if added, false if deduplicated.
    pub fn add_if_new(&mut self, rule: &str, source: &str, confidence: f32) -> Result<bool> {
        let dominated = self.lessons.iter().any(|l| word_overlap_similarity(&l.rule, rule) > 0.8);
        if dominated {
            tracing::debug!(rule, "Insight deduplicated — similar lesson exists");
            return Ok(false);
        }
        self.add(rule, source, confidence)?;
        Ok(true)
    }

    /// Evict lowest-confidence lessons if count exceeds cap.
    /// Returns the number of evicted lessons.
    pub fn enforce_cap(&mut self, max_count: usize) -> Result<usize> {
        if self.lessons.len() <= max_count { return Ok(0); }
        self.lessons.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));
        let evicted = self.lessons.len() - max_count;
        self.lessons.truncate(max_count);
        tracing::info!(evicted, remaining = self.lessons.len(), "Lessons cap enforced");
        self.persist()?;
        Ok(evicted)
    }

    /// Decay confidence of unused lessons. Evict those below minimum.
    /// Returns the number of evicted lessons.
    pub fn decay_unused(&mut self, factor: f32, min_confidence: f32) -> Result<usize> {
        for lesson in &mut self.lessons {
            if lesson.times_applied == 0 {
                lesson.confidence *= factor;
            }
        }
        let before = self.lessons.len();
        self.lessons.retain(|l| l.confidence >= min_confidence);
        let evicted = before - self.lessons.len();
        if evicted > 0 { self.persist()?; }
        Ok(evicted)
    }
}

/// Word-overlap similarity between two strings (Jaccard on words).
fn word_overlap_similarity(a: &str, b: &str) -> f32 {
    let a_lower = a.to_lowercase();
    let b_lower = b.to_lowercase();
    let a_words: std::collections::HashSet<&str> = a_lower.split_whitespace().collect();
    let b_words: std::collections::HashSet<&str> = b_lower.split_whitespace().collect();
    if a_words.is_empty() && b_words.is_empty() { return 1.0; }
    let intersection = a_words.intersection(&b_words).count() as f32;
    let union = a_words.union(&b_words).count() as f32;
    if union == 0.0 { 0.0 } else { intersection / union }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_and_filter() {
        let mut store = LessonStore::new();
        store.add("Rule 1", "test", 0.9).unwrap();
        store.add("Rule 2", "test", 0.3).unwrap();
        assert_eq!(store.high_confidence(0.8).len(), 1);
    }

    #[test]
    fn test_search() {
        let mut store = LessonStore::new();
        store.add("Use Rust", "test", 0.9).unwrap();
        store.add("Use Python", "test", 0.8).unwrap();
        assert_eq!(store.search("rust", 10).len(), 1);
    }

    #[test]
    fn test_add_if_new_deduplicates() {
        let mut store = LessonStore::new();
        assert!(store.add_if_new("User prefers short answers", "test", 0.9).unwrap());
        assert!(!store.add_if_new("User prefers short answers", "test", 0.95).unwrap());
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn test_add_if_new_allows_different() {
        let mut store = LessonStore::new();
        store.add_if_new("User prefers short answers", "test", 0.9).unwrap();
        assert!(store.add_if_new("User works on a Rust AI engine", "test", 0.85).unwrap());
        assert_eq!(store.count(), 2);
    }

    #[test]
    fn test_enforce_cap_evicts_lowest() {
        let mut store = LessonStore::new();
        store.add("Low", "test", 0.2).unwrap();
        store.add("Medium", "test", 0.5).unwrap();
        store.add("High", "test", 0.9).unwrap();
        let evicted = store.enforce_cap(2).unwrap();
        assert_eq!(evicted, 1);
        assert_eq!(store.count(), 2);
        // The low-confidence lesson should be gone
        assert!(store.all().iter().all(|l| l.confidence >= 0.5));
    }

    #[test]
    fn test_enforce_cap_noop_under_limit() {
        let mut store = LessonStore::new();
        store.add("Only one", "test", 0.9).unwrap();
        assert_eq!(store.enforce_cap(10).unwrap(), 0);
    }

    #[test]
    fn test_word_overlap_identical() {
        assert!(word_overlap_similarity("user prefers short", "user prefers short") > 0.99);
    }

    #[test]
    fn test_word_overlap_different() {
        assert!(word_overlap_similarity("user prefers short", "engine uses Rust") < 0.2);
    }
}
