//! Tier 6: Lessons — discovered/extracted behavioral rules.

use serde::{Deserialize, Serialize};

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
}

impl LessonStore {
    pub fn new() -> Self { Self { lessons: Vec::new() } }

    pub fn add(&mut self, rule: &str, source: &str, confidence: f32) {
        self.lessons.push(Lesson {
            id: uuid::Uuid::new_v4().to_string(),
            rule: rule.to_string(),
            source: source.to_string(),
            confidence,
            times_applied: 0,
        });
    }

    pub fn apply(&mut self, id: &str) {
        if let Some(lesson) = self.lessons.iter_mut().find(|l| l.id == id) {
            lesson.times_applied += 1;
        }
    }

    pub fn high_confidence(&self, threshold: f32) -> Vec<&Lesson> {
        self.lessons.iter().filter(|l| l.confidence >= threshold).collect()
    }

    pub fn all(&self) -> &[Lesson] { &self.lessons }
    pub fn count(&self) -> usize { self.lessons.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_lesson() {
        let mut store = LessonStore::new();
        store.add("Always verify before responding", "observer", 0.9);
        assert_eq!(store.count(), 1);
    }

    #[test]
    fn test_high_confidence_filter() {
        let mut store = LessonStore::new();
        store.add("r1", "src", 0.9);
        store.add("r2", "src", 0.3);
        assert_eq!(store.high_confidence(0.8).len(), 1);
    }
}
