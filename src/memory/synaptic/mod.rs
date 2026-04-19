// Ern-OS — Tier 4: Synaptic Knowledge Graph — in-memory Hebbian graph

pub mod query;
pub mod relationships;
pub mod plasticity;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticNode {
    pub id: String,
    pub data: HashMap<String, String>,
    pub layer: String,
    pub strength: f32,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub access_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SynapticEdge {
    pub source: String,
    pub target: String,
    pub edge_type: String,
    pub weight: f32,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

pub struct SynapticGraph {
    nodes: HashMap<String, SynapticNode>,
    edges: Vec<SynapticEdge>,
    persist_dir: Option<PathBuf>,
}

impl SynapticGraph {
    pub fn new(persist_dir: Option<PathBuf>) -> Self {
        let mut graph = Self {
            nodes: HashMap::new(), edges: Vec::new(), persist_dir,
        };
        let _ = graph.load();
        graph
    }

    pub fn upsert_node(&mut self, id: &str, data: HashMap<String, String>, layer: &str) -> Result<()> {
        let now = chrono::Utc::now();
        if let Some(node) = self.nodes.get_mut(id) {
            for (k, v) in &data { node.data.insert(k.clone(), v.clone()); }
            node.updated_at = now;
            node.access_count += 1;
            if !layer.is_empty() { node.layer = layer.to_string(); }
        } else {
            self.nodes.insert(id.to_string(), SynapticNode {
                id: id.to_string(), data, layer: layer.to_string(),
                strength: 1.0, created_at: now, updated_at: now, access_count: 1,
            });
        }
        self.persist()
    }

    pub fn add_edge(&mut self, source: &str, target: &str, edge_type: &str) -> Result<()> {
        if !self.nodes.contains_key(source) || !self.nodes.contains_key(target) {
            anyhow::bail!("Both source and target nodes must exist");
        }
        let exists = self.edges.iter().any(|e|
            e.source == source && e.target == target && e.edge_type == edge_type
        );
        if !exists {
            self.edges.push(SynapticEdge {
                source: source.to_string(), target: target.to_string(),
                edge_type: edge_type.to_string(), weight: 1.0,
                created_at: chrono::Utc::now(),
            });
        }
        self.persist()
    }

    pub fn get_node(&self, id: &str) -> Option<&SynapticNode> { self.nodes.get(id) }

    pub fn search_nodes(&self, query: &str, limit: usize) -> Vec<&SynapticNode> {
        let q = query.to_lowercase();
        let mut matches: Vec<&SynapticNode> = self.nodes.values()
            .filter(|n| {
                n.id.to_lowercase().contains(&q) ||
                n.data.values().any(|v| v.to_lowercase().contains(&q))
            })
            .collect();
        matches.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap_or(std::cmp::Ordering::Equal));
        matches.truncate(limit);
        matches
    }

    pub fn recent_nodes(&self, n: usize) -> Vec<&SynapticNode> {
        let mut nodes: Vec<&SynapticNode> = self.nodes.values().collect();
        nodes.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        nodes.truncate(n);
        nodes
    }

    pub fn co_activate(&mut self, a: &str, b: &str, delta: f32) {
        plasticity::strengthen_edge(&mut self.edges, a, b, delta);
        let _ = self.persist();
    }

    pub fn decay_all(&mut self, factor: f32) {
        plasticity::decay_all_edges(&mut self.edges, factor);
        let _ = self.persist();
    }

    pub fn layers(&self) -> Vec<String> {
        let mut ls: Vec<String> = self.nodes.values()
            .map(|n| n.layer.clone())
            .collect::<std::collections::HashSet<_>>()
            .into_iter().collect();
        ls.sort();
        ls
    }

    pub fn node_count(&self) -> usize { self.nodes.len() }
    pub fn edge_count(&self) -> usize { self.edges.len() }

    fn persist(&self) -> Result<()> {
        if let Some(ref dir) = self.persist_dir {
            std::fs::create_dir_all(dir)?;
            std::fs::write(dir.join("nodes.json"), serde_json::to_string(&self.nodes)?)?;
            std::fs::write(dir.join("edges.json"), serde_json::to_string(&self.edges)?)?;
        }
        Ok(())
    }

    fn load(&mut self) -> Result<()> {
        if let Some(ref dir) = self.persist_dir {
            let np = dir.join("nodes.json");
            if np.exists() {
                self.nodes = serde_json::from_str(&std::fs::read_to_string(&np)?)?;
            }
            let ep = dir.join("edges.json");
            if ep.exists() {
                self.edges = serde_json::from_str(&std::fs::read_to_string(&ep)?)?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upsert_and_get() {
        let mut g = SynapticGraph::new(None);
        let mut data = HashMap::new();
        data.insert("type".into(), "language".into());
        g.upsert_node("rust", data, "tech").unwrap();
        assert!(g.get_node("rust").is_some());
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_edge() {
        let mut g = SynapticGraph::new(None);
        g.upsert_node("a", HashMap::new(), "x").unwrap();
        g.upsert_node("b", HashMap::new(), "x").unwrap();
        g.add_edge("a", "b", "related_to").unwrap();
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_search() {
        let mut g = SynapticGraph::new(None);
        let mut data = HashMap::new();
        data.insert("desc".into(), "A systems programming language".into());
        g.upsert_node("rust", data, "tech").unwrap();
        assert_eq!(g.search_nodes("systems", 10).len(), 1);
    }
}
