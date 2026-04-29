// Ern-OS — High-performance, model-neutral Rust AI agent engine
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! 7-Tier Cognitive Memory Architecture.
//!
//! Tier 1: Working context (inference/context — not stored here)
//! Tier 2: Consolidation — summarize overflow sessions
//! Tier 3: Timeline — verbatim session archive
//! Tier 4: Synaptic KG — in-memory graph with Hebbian plasticity
//! Tier 5: Scratchpad — pinned notes
//! Tier 6: Lessons — discovered rules
//! Tier 7: Procedures — reusable workflows

pub mod consolidation;
pub mod timeline;
pub mod embeddings;
pub mod scratchpad;
pub mod lessons;
pub mod procedures;
pub mod synaptic;
pub mod skills;
pub mod document_store;

use anyhow::{Context, Result};
use std::path::Path;

/// Orchestrates all 8 memory tiers with full disk persistence.
pub struct MemoryManager {
    pub consolidation: consolidation::ConsolidationEngine,
    pub timeline: timeline::TimelineStore,
    pub embeddings: embeddings::EmbeddingStore,
    pub scratchpad: scratchpad::ScratchpadStore,
    pub lessons: lessons::LessonStore,
    pub procedures: procedures::ProcedureStore,
    pub synaptic: synaptic::SynapticGraph,
    /// Tier 8: Document Knowledge — chunked documents with neural embeddings.
    pub documents: document_store::DocumentStore,
}

impl MemoryManager {
    /// Create a new memory manager with all tiers initialised.
    pub fn new(data_dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(data_dir)
            .with_context(|| format!("Failed to create data dir: {}", data_dir.display()))?;

        let timeline_dir = data_dir.join("timeline");
        let consolidation_path = data_dir.join("consolidation.json");
        let lessons_path = data_dir.join("lessons.json");
        let procedures_path = data_dir.join("procedures.json");
        let scratchpad_path = data_dir.join("scratchpad.json");
        let embeddings_path = data_dir.join("embeddings.json");
        let synaptic_dir = data_dir.join("synaptic");
        let documents_path = data_dir.join("documents.json");

        let consolidation = consolidation::ConsolidationEngine::open(&consolidation_path)
            .with_context(|| "Failed to open consolidation store")?;
        let timeline = timeline::TimelineStore::new(&timeline_dir)?;
        let lessons = lessons::LessonStore::open(&lessons_path)
            .with_context(|| "Failed to open lesson store")?;
        let procedures = procedures::ProcedureStore::open(&procedures_path)
            .with_context(|| "Failed to open procedure store")?;
        let scratchpad = scratchpad::ScratchpadStore::open(&scratchpad_path)
            .with_context(|| "Failed to open scratchpad store")?;
        let embeddings = embeddings::EmbeddingStore::open(&embeddings_path)
            .with_context(|| "Failed to open embedding store")?;
        let synaptic = synaptic::SynapticGraph::new(Some(synaptic_dir));
        let documents = document_store::DocumentStore::open(&documents_path)
            .with_context(|| "Failed to open document store")?;

        tracing::info!(
            timeline = timeline.entry_count(),
            lessons = lessons.count(),
            procedures = procedures.count(),
            scratchpad = scratchpad.count(),
            embeddings = embeddings.count(),
            documents = documents.count(),
            "Memory manager initialised — all persistent tiers loaded"
        );

        Ok(Self {
            consolidation,
            timeline,
            embeddings,
            scratchpad,
            lessons,
            procedures,
            synaptic,
            documents,
        })
    }

    /// Recall context as a formatted string for system prompt injection.
    /// Allocation: Scratchpad (30%) → Lessons (20%) → Documents (15%) → Skills (15%) → Timeline (10%) → KG (10%).
    /// `query_embedding` enables RAG retrieval from the document store when available.
    pub fn recall_context(&self, _query: &str, budget_tokens: usize, query_embedding: Option<&[f32]>) -> String {
        let total_chars = budget_tokens * 4; // ~4 chars per token
        let mut parts = Vec::new();
        if let Some(s) = self.recall_scratchpad(total_chars * 30 / 100) { parts.push(s); }
        if let Some(s) = self.recall_lessons(total_chars * 20 / 100) { parts.push(s); }
        if let Some(s) = self.recall_documents(total_chars * 15 / 100, query_embedding) { parts.push(s); }
        if let Some(s) = self.recall_procedures(total_chars * 15 / 100) { parts.push(s); }
        if let Some(s) = self.recall_timeline(total_chars * 10 / 100) { parts.push(s); }
        if let Some(s) = self.recall_knowledge_graph() { parts.push(s); }
        parts.join("\n")
    }

    fn recall_scratchpad(&self, budget: usize) -> Option<String> {
        let notes = self.scratchpad.all();
        if notes.is_empty() { return None; }
        let mut section = String::from("[Memory — Scratchpad]\n");
        for note in notes {
            let line = format!("• {}: {}\n", note.key, note.value);
            if section.len() + line.len() > budget { break; }
            section.push_str(&line);
        }
        Some(section)
    }

    fn recall_lessons(&self, budget: usize) -> Option<String> {
        let lessons = self.lessons.high_confidence(0.8);
        if lessons.is_empty() { return None; }
        let mut section = String::from("[Memory — Learned Lessons]\n");
        for l in &lessons {
            let line = format!("• {} (confidence: {:.0}%)\n", l.rule, l.confidence * 100.0);
            if section.len() + line.len() > budget { break; }
            section.push_str(&line);
        }
        Some(section)
    }

    fn recall_timeline(&self, budget: usize) -> Option<String> {
        let recent = self.timeline.recent(30);
        if recent.is_empty() { return None; }
        let mut section = String::from("[Memory — Recent Context]\n");
        for entry in recent {
            // Full content — no truncation. The model's context window is the only limit.
            let line = format!("• {}\n", entry.transcript);
            if section.len() + line.len() > budget { break; }
            section.push_str(&line);
        }
        Some(section)
    }

    fn recall_knowledge_graph(&self) -> Option<String> {
        let nodes = self.synaptic.recent_nodes(5);
        if nodes.is_empty() { return None; }
        let mut section = String::from("[Memory — Knowledge Graph]\n");
        for n in &nodes {
            section.push_str(&format!("• {} [{}]\n", n.id, n.layer));
        }
        Some(section)
    }

    fn recall_documents(&self, budget: usize, query_embedding: Option<&[f32]>) -> Option<String> {
        if self.documents.count() == 0 { return None; }

        let mut section = String::from("[Memory — Document Knowledge]\n");

        // If we have a query embedding, do real RAG retrieval
        if let Some(qvec) = query_embedding {
            let results = self.documents.search_by_vector(qvec, 8);
            if results.is_empty() {
                return None;
            }
            for (chunk, score) in results {
                let line = format!("• [{}:p{}] ({:.2}) {}\n",
                    chunk.document_name, chunk.page, score,
                    &chunk.content.chars().take(500).collect::<String>());
                if section.len() + line.len() > budget { break; }
                section.push_str(&line);
            }
        } else {
            // No embedding available — show document index as fallback
            let names = self.documents.document_names();
            section.push_str(&format!("Indexed documents: {}\n", names.join(", ")));
        }

        Some(section)
    }

    fn recall_procedures(&self, budget: usize) -> Option<String> {
        let procs = self.procedures.all();
        if procs.is_empty() { return None; }
        let mut section = String::from("[Memory — Known Skills]\n");
        for p in procs {
            let line = format!("• **{}** ({} steps, used {} times)\n",
                p.name, p.steps.len(), p.success_count);
            if section.len() + line.len() > budget { break; }
            section.push_str(&line);
        }
        Some(section)
    }



    /// Ingest a single message into timeline.
    /// Accepts an optional pre-computed embedding vector from `provider.embed()`.
    pub fn ingest_turn(&mut self, role: &str, content: &str, session_id: &str, embedding: Option<Vec<f32>>) {
        let transcript = format!("{}: {}", role, content);
        if let Err(e) = self.timeline.archive(session_id, &transcript) {
            tracing::warn!(error = %e, "Failed to archive turn");
        }

        // Ingest embedding for semantic search (when provided by caller)
        if let Some(vector) = embedding {
            if let Err(e) = self.embeddings.insert(content, role, vector) {
                tracing::warn!(error = %e, "Failed to insert embedding");
            }
        }
    }

    /// Memory system status summary.
    pub fn status_summary(&self) -> String {
        format!(
            "Consolidations: {} | Timeline: {} | Lessons: {} | Procedures: {} | Scratchpad: {} | Embeddings: {} | Documents: {}",
            self.consolidation.consolidation_count(),
            self.timeline.entry_count(),
            self.lessons.count(),
            self.procedures.count(),
            self.scratchpad.count(),
            self.embeddings.count(),
            self.documents.count(),
        )
    }

    /// Factory reset: wipe all tiers.
    pub fn clear(&mut self) {
        self.timeline.clear_entries();
        self.lessons = lessons::LessonStore::new();
        self.procedures = procedures::ProcedureStore::new();
        self.scratchpad = scratchpad::ScratchpadStore::new();
        self.embeddings = embeddings::EmbeddingStore::new();
        self.consolidation = consolidation::ConsolidationEngine::new();
        self.documents = document_store::DocumentStore::new();
        tracing::info!("MemoryManager cleared — all tiers wiped");
    }
}


