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

use anyhow::{Context, Result};
use std::path::Path;

/// Orchestrates all 7 memory tiers with full disk persistence.
pub struct MemoryManager {
    pub consolidation: consolidation::ConsolidationEngine,
    pub timeline: timeline::TimelineStore,
    pub embeddings: embeddings::EmbeddingStore,
    pub scratchpad: scratchpad::ScratchpadStore,
    pub lessons: lessons::LessonStore,
    pub procedures: procedures::ProcedureStore,
    pub synaptic: synaptic::SynapticGraph,
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

        tracing::info!(
            timeline = timeline.entry_count(),
            lessons = lessons.count(),
            procedures = procedures.count(),
            scratchpad = scratchpad.count(),
            embeddings = embeddings.count(),
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
        })
    }

    /// Recall context as a formatted string for system prompt injection.
    /// Allocation: Scratchpad (35%) → Lessons (25%) → Skills (15%) → Timeline (15%) → KG (10%).
    pub fn recall_context(&self, _query: &str, budget_tokens: usize) -> String {
        let total_chars = budget_tokens * 4; // ~4 chars per token
        let mut parts = Vec::new();
        if let Some(s) = self.recall_scratchpad(total_chars * 35 / 100) { parts.push(s); }
        if let Some(s) = self.recall_lessons(total_chars * 25 / 100) { parts.push(s); }
        if let Some(s) = self.recall_procedures(total_chars * 15 / 100) { parts.push(s); }
        if let Some(s) = self.recall_timeline(total_chars * 15 / 100) { parts.push(s); }
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
    pub fn ingest_turn(&mut self, role: &str, content: &str, session_id: &str) {
        let transcript = format!("{}: {}", role, content);
        if let Err(e) = self.timeline.archive(session_id, &transcript) {
            tracing::warn!(error = %e, "Failed to archive turn");
        }

        // Also ingest into embeddings for semantic search
        if content.len() > 20 { // Skip trivial messages
            let vector = text_to_vector(content);
            if let Err(e) = self.embeddings.insert(content, role, vector) {
                tracing::warn!(error = %e, "Failed to insert embedding");
            }
        }
    }

    /// Memory system status summary.
    pub fn status_summary(&self) -> String {
        format!(
            "Consolidations: {} | Timeline: {} | Lessons: {} | Procedures: {} | Scratchpad: {} | Embeddings: {}",
            self.consolidation.consolidation_count(),
            self.timeline.entry_count(),
            self.lessons.count(),
            self.procedures.count(),
            self.scratchpad.count(),
            self.embeddings.count(),
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
        tracing::info!("MemoryManager cleared — all tiers wiped");
    }
}

/// Lightweight text-to-vector using hash projection.
/// Not a neural embedding — this is a deterministic hash into a fixed-dimension
/// vector that enables basic similarity search without an embedding model.
/// When a proper embedding endpoint is available, replace this with real embeddings.
fn text_to_vector(text: &str) -> Vec<f32> {
    const DIMS: usize = 64;
    let mut vec = vec![0.0f32; DIMS];
    for word in text.split_whitespace() {
        let w = word.to_lowercase();
        let hash = w.bytes().fold(0u64, |acc, b| acc.wrapping_mul(31).wrapping_add(b as u64));
        let idx = (hash % DIMS as u64) as usize;
        vec[idx] += 1.0;
    }
    // Normalize
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut vec { *v /= norm; }
    }
    vec
}
