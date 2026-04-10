// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Query operations — store, search, beliefs, recent, stats, export.

use super::{SynapticGraph, SynapticNode, SynapticEdge};

impl SynapticGraph {
    /// Store a concept → data entry. Appends to existing concepts.
    pub async fn store(&self, concept: &str, data: &str) {
        let key = concept.to_lowercase();
        let now = chrono::Utc::now().to_rfc3339();
        {
            let mut nodes = self.nodes.write().await;
            let entry = nodes.entry(key.clone()).or_insert_with(|| SynapticNode {
                concept: concept.to_string(),
                data: Vec::new(),
                created_at: now.clone(),
                updated_at: now.clone(),
                layer: "concepts".to_string(),
                origin: "local".to_string(),
            });
            if !entry.data.iter().any(|d| d == data) {
                entry.data.push(data.to_string());
                entry.updated_at = now;
            }
        }
        self.save_nodes().await;
        tracing::info!("[SYNAPTIC] Stored: '{}' → '{}'", concept, data);
    }

    /// Search for a concept. Three-pass: exact → prefix → substring.
    pub async fn search(&self, concept: &str) -> Vec<String> {
        let query = concept.to_lowercase();
        let nodes = self.nodes.read().await;

        if let Some(node) = nodes.get(&query) {
            return node
                .data
                .iter()
                .map(|d| format!("[{}] {}", node.concept, d))
                .collect();
        }

        let mut results = Vec::new();
        for node in nodes.values() {
            let key = node.concept.to_lowercase();
            if key.starts_with(&query) || key.contains(&query) || query.contains(&key) {
                for d in &node.data {
                    results.push(format!("[{}] {}", node.concept, d));
                }
            }
        }
        results
    }

    /// Retrieve the most recently updated nodes.
    pub async fn get_recent_nodes(&self, limit: usize) -> Vec<(String, String)> {
        let nodes = self.nodes.read().await;
        let mut sorted: Vec<&SynapticNode> = nodes.values().collect();
        sorted.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        sorted
            .into_iter()
            .take(limit)
            .map(|n| (n.concept.clone(), n.data.join("; ")))
            .collect()
    }

    /// Core beliefs — concepts with the most data entries.
    pub async fn get_beliefs(&self, limit: usize) -> Vec<String> {
        let nodes = self.nodes.read().await;
        let mut sorted: Vec<&SynapticNode> = nodes.values().collect();
        sorted.sort_by(|a, b| b.data.len().cmp(&a.data.len()));
        sorted
            .into_iter()
            .take(limit)
            .map(|n| format_belief(n))
            .collect()
    }

    /// Total counts for diagnostics.
    pub async fn stats(&self) -> (usize, usize) {
        let nodes = self.nodes.read().await;
        let edges = self.edges.read().await;
        (nodes.len(), edges.len())
    }

    /// Export the entire raw graph for dashboard rendering.
    pub async fn export_graph(&self) -> (Vec<SynapticNode>, Vec<SynapticEdge>) {
        let nodes = self.nodes.read().await;
        let edges = self.edges.read().await;
        (nodes.values().cloned().collect(), edges.clone())
    }
}

fn format_belief(n: &SynapticNode) -> String {
    let summary = if n.data.len() <= 3 {
        n.data.join("; ")
    } else {
        format!(
            "{} (+{} more)",
            n.data[..3].join("; "),
            n.data.len() - 3
        )
    };
    format!("{}: {}", n.concept, summary)
}
