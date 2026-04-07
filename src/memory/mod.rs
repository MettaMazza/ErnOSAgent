//! 7-Tier Cognitive Memory Architecture.

pub mod consolidation;
pub mod timeline;
pub mod embeddings;
pub mod knowledge_graph;
pub mod scratchpad;
pub mod lessons;
pub mod procedures;

use anyhow::{Context, Result};
use std::path::Path;

/// Orchestrates all 7 memory tiers.
pub struct MemoryManager {
    pub consolidation: consolidation::ConsolidationEngine,
    pub timeline: timeline::TimelineStore,
    pub embeddings: embeddings::EmbeddingService,
    pub scratchpad: scratchpad::ScratchpadStore,
    pub lessons: lessons::LessonStore,
    pub procedures: procedures::ProcedureStore,
}

impl MemoryManager {
    /// Create a new memory manager with all tiers initialised.
    pub async fn new(
        data_dir: &Path,
        _neo4j_uri: &str,
        _neo4j_user: &str,
        _neo4j_pass: &str,
    ) -> Result<Self> {
        let timeline_dir = data_dir.join("timeline");
        std::fs::create_dir_all(&timeline_dir)
            .with_context(|| format!("Failed to create timeline dir: {}", timeline_dir.display()))?;

        Ok(Self {
            consolidation: consolidation::ConsolidationEngine::new(),
            timeline: timeline::TimelineStore::new(&timeline_dir)?,
            embeddings: embeddings::EmbeddingService::new(),
            scratchpad: scratchpad::ScratchpadStore::new(),
            lessons: lessons::LessonStore::new(),
            procedures: procedures::ProcedureStore::new(),
        })
    }

    /// Get a summary of the memory system state for the HUD prompt.
    pub fn status_summary(&self) -> String {
        format!(
            "Consolidations: {} | Timeline entries: {} | Lessons: {} | Procedures: {}",
            self.consolidation.consolidation_count(),
            self.timeline.entry_count(),
            self.lessons.count(),
            self.procedures.count(),
        )
    }
}
