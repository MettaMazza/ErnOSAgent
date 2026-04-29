// Ern-OS — Document store — chunked document storage with neural embeddings
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Tier 8 memory: persistent document chunk storage with embedding-based retrieval.
//! Documents are split into paragaph-boundary chunks, embedded via `provider.embed()`,
//! and retrieved via cosine similarity for RAG-style question answering.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A stored document chunk with its neural embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: String,
    pub document_name: String,
    pub page: usize,
    pub chunk_index: usize,
    pub content: String,
    pub vector: Vec<f32>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

/// Persistent document chunk store with embedding-based semantic search.
pub struct DocumentStore {
    chunks: Vec<DocumentChunk>,
    file_path: Option<PathBuf>,
}

impl DocumentStore {
    pub fn new() -> Self {
        Self { chunks: Vec::new(), file_path: None }
    }

    /// Open from disk or create empty.
    pub fn open(path: &Path) -> Result<Self> {
        tracing::info!(path = %path.display(), "DocumentStore: opening");
        let mut store = Self { chunks: Vec::new(), file_path: Some(path.to_path_buf()) };
        if path.exists() {
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read document store: {}", path.display()))?;
            store.chunks = serde_json::from_str(&content)?;
            tracing::info!(chunks = store.chunks.len(), "DocumentStore: loaded from disk");
        }
        Ok(store)
    }

    /// Ingest document pages: split into chunks, embed each via provider.
    /// Returns the number of chunks stored.
    pub async fn ingest_document(
        &mut self,
        name: &str,
        pages: &[(usize, String)],
        provider: &dyn crate::provider::Provider,
        context_length: usize,
    ) -> Result<usize> {
        let chunk_size = chunk_size_chars(context_length);
        let mut count = 0;

        for (page, content) in pages {
            let chunks = chunk_page(content, chunk_size);
            for (idx, chunk_text) in chunks.iter().enumerate() {
                let vector = provider.embed(chunk_text).await
                    .with_context(|| format!("Embedding failed for {}:p{}:c{}", name, page, idx))?;

                self.chunks.push(DocumentChunk {
                    id: uuid::Uuid::new_v4().to_string(),
                    document_name: name.to_string(),
                    page: *page,
                    chunk_index: idx,
                    content: chunk_text.clone(),
                    vector,
                    created_at: chrono::Utc::now(),
                });
                count += 1;
            }
        }

        self.persist()?;
        tracing::info!(document = %name, chunks = count, "DocumentStore: ingested");
        Ok(count)
    }

    /// Semantic search: find top-k chunks by cosine similarity to query vector.
    pub fn search_by_vector(&self, query_vec: &[f32], top_k: usize) -> Vec<(&DocumentChunk, f32)> {
        let mut scored: Vec<_> = self.chunks.iter()
            .map(|c| (c, crate::memory::embeddings::cosine_similarity(query_vec, &c.vector)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    pub fn count(&self) -> usize { self.chunks.len() }

    pub fn document_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.chunks.iter()
            .map(|c| c.document_name.clone())
            .collect();
        names.sort();
        names.dedup();
        names
    }

    /// Remove all chunks for a specific document.
    pub fn remove_document(&mut self, name: &str) {
        let before = self.chunks.len();
        self.chunks.retain(|c| c.document_name != name);
        let removed = before - self.chunks.len();
        if removed > 0 {
            tracing::info!(document = %name, removed, "DocumentStore: document removed");
            let _ = self.persist();
        }
    }

    fn persist(&self) -> Result<()> {
        if let Some(ref path) = self.file_path {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(path, serde_json::to_string(&self.chunks)?)?;
        }
        Ok(())
    }
}

/// Split page text into chunks at paragraph boundaries.
/// Chunk size derived from context_length per §2.1.
fn chunk_page(content: &str, chunk_size: usize) -> Vec<String> {
    if content.len() <= chunk_size {
        return vec![content.to_string()];
    }

    let mut chunks = Vec::new();
    let mut current = String::new();

    for paragraph in content.split("\n\n") {
        if current.len() + paragraph.len() + 2 > chunk_size && !current.is_empty() {
            chunks.push(current.trim().to_string());
            current = String::new();
        }
        if !current.is_empty() {
            current.push_str("\n\n");
        }
        current.push_str(paragraph);
    }

    if !current.trim().is_empty() {
        chunks.push(current.trim().to_string());
    }

    chunks
}

/// Chunk size = context_length / 128 * 4 chars (derived from model, §2.1).
/// For a 262K context: (262144 / 128) * 4 = 8192 chars per chunk.
fn chunk_size_chars(context_length: usize) -> usize {
    (context_length / 128) * 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_page_small_content() {
        let chunks = chunk_page("Hello world", 1000);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Hello world");
    }

    #[test]
    fn test_chunk_page_splits_at_paragraphs() {
        let content = "Para one.\n\nPara two.\n\nPara three.";
        let chunks = chunk_page(content, 15);
        assert!(chunks.len() >= 2);
        // Each chunk should be a complete paragraph, not split mid-word
        for chunk in &chunks {
            assert!(!chunk.starts_with("\n\n"));
        }
    }

    #[test]
    fn test_chunk_size_scales_with_context() {
        let small = chunk_size_chars(32768);
        let large = chunk_size_chars(262144);
        assert!(large > small);
        assert!(small > 0);
    }

    #[test]
    fn test_chunk_size_262k() {
        assert_eq!(chunk_size_chars(262144), 8192);
    }

    #[test]
    fn test_document_store_new() {
        let store = DocumentStore::new();
        assert_eq!(store.count(), 0);
        assert!(store.document_names().is_empty());
    }

    #[test]
    fn test_search_by_vector_empty_store() {
        let store = DocumentStore::new();
        let results = store.search_by_vector(&[1.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_remove_document() {
        let mut store = DocumentStore::new();
        store.chunks.push(DocumentChunk {
            id: "1".into(),
            document_name: "book.md".into(),
            page: 1,
            chunk_index: 0,
            content: "test".into(),
            vector: vec![1.0],
            created_at: chrono::Utc::now(),
        });
        store.chunks.push(DocumentChunk {
            id: "2".into(),
            document_name: "other.md".into(),
            page: 1,
            chunk_index: 0,
            content: "other".into(),
            vector: vec![0.0],
            created_at: chrono::Utc::now(),
        });
        assert_eq!(store.count(), 2);
        store.remove_document("book.md");
        assert_eq!(store.count(), 1);
        assert_eq!(store.document_names(), vec!["other.md".to_string()]);
    }
}
