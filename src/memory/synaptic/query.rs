// Ern-OS — Synaptic KG: Query helpers

use super::SynapticNode;
use std::collections::HashMap;

/// Find nodes connected to a given node via edges.
pub fn neighbors<'a>(
    node_id: &str,
    nodes: &'a HashMap<String, SynapticNode>,
    edges: &[super::SynapticEdge],
) -> Vec<&'a SynapticNode> {
    let connected_ids: Vec<&str> = edges.iter()
        .filter_map(|e| {
            if e.source == node_id { Some(e.target.as_str()) }
            else if e.target == node_id { Some(e.source.as_str()) }
            else { None }
        })
        .collect();

    connected_ids.iter()
        .filter_map(|id| nodes.get(*id))
        .collect()
}

/// Get all edges for a specific node.
pub fn edges_for<'a>(
    node_id: &str,
    edges: &'a [super::SynapticEdge],
) -> Vec<&'a super::SynapticEdge> {
    edges.iter()
        .filter(|e| e.source == node_id || e.target == node_id)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::*;

    #[test]
    fn test_neighbors() {
        let mut nodes = HashMap::new();
        let now = chrono::Utc::now();
        nodes.insert("a".into(), SynapticNode {
            id: "a".into(), data: HashMap::new(), layer: "x".into(),
            strength: 1.0, created_at: now, updated_at: now, access_count: 0,
        });
        nodes.insert("b".into(), SynapticNode {
            id: "b".into(), data: HashMap::new(), layer: "x".into(),
            strength: 1.0, created_at: now, updated_at: now, access_count: 0,
        });
        let edges = vec![SynapticEdge {
            source: "a".into(), target: "b".into(), edge_type: "link".into(),
            weight: 1.0, created_at: now,
        }];
        assert_eq!(neighbors("a", &nodes, &edges).len(), 1);
    }
}
