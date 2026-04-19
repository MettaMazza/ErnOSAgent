// Ern-OS — Embedding store — cosine similarity vector index

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredEmbedding {
    pub id: String,
    pub source_text: String,
    pub source_type: String,
    pub vector: Vec<f32>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

pub struct EmbeddingStore {
    embeddings: Vec<StoredEmbedding>,
    file_path: Option<PathBuf>,
    dimensions: usize,
}

impl EmbeddingStore {
    pub fn new() -> Self {
        Self { embeddings: Vec::new(), file_path: None, dimensions: 0 }
    }

    pub fn open(path: &Path) -> Result<Self> {
        tracing::info!(module = "embeddings", fn_name = "open", "embeddings::open called");
        let mut store = Self { embeddings: Vec::new(), file_path: Some(path.to_path_buf()), dimensions: 0 };
        if path.exists() {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read embeddings: {}", path.display()))?;
            store.embeddings = serde_json::from_str(&content)?;
            if let Some(first) = store.embeddings.first() {
                store.dimensions = first.vector.len();
            }
        }
        Ok(store)
    }

    fn persist(&self) -> Result<()> {
        if let Some(ref path) = self.file_path {
            if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
            std::fs::write(path, serde_json::to_string(&self.embeddings)?)?;
        }
        Ok(())
    }

    pub fn insert(&mut self, text: &str, source_type: &str, vector: Vec<f32>) -> Result<String> {
        tracing::info!(module = "embeddings", fn_name = "insert", "embeddings::insert called");
        if self.dimensions == 0 && !vector.is_empty() { self.dimensions = vector.len(); }
        let id = uuid::Uuid::new_v4().to_string();
        self.embeddings.push(StoredEmbedding {
            id: id.clone(), source_text: text.to_string(),
            source_type: source_type.to_string(), vector, created_at: chrono::Utc::now(),
        });
        self.persist()?;
        Ok(id)
    }

    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<(&StoredEmbedding, f32)> {
        tracing::info!(module = "embeddings", fn_name = "search", "embeddings::search called");
        let mut scored: Vec<_> = self.embeddings.iter()
            .map(|e| (e, cosine_similarity(query, &e.vector)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    pub fn count(&self) -> usize { self.embeddings.len() }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 { return 0.0; }
    dot / (na * nb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_insert_and_search() {
        let mut store = EmbeddingStore::new();
        store.insert("Rust", "test", vec![1.0, 0.0]).unwrap();
        store.insert("Python", "test", vec![0.0, 1.0]).unwrap();
        let results = store.search(&[1.0, 0.0], 1);
        assert_eq!(results[0].0.source_text, "Rust");
    }
}
