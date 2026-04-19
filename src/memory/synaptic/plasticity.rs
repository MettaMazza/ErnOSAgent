// Ern-OS — Synaptic KG: Hebbian plasticity

use super::SynapticEdge;

/// Strengthen the edge between two co-activated nodes (Hebbian learning).
pub fn strengthen_edge(edges: &mut [SynapticEdge], a: &str, b: &str, delta: f32) {
    for edge in edges.iter_mut() {
        if (edge.source == a && edge.target == b) ||
           (edge.source == b && edge.target == a) {
            edge.weight = (edge.weight + delta).min(10.0);
            return;
        }
    }
    // No existing edge — create one
    edges.to_vec().push(SynapticEdge {
        source: a.to_string(), target: b.to_string(),
        edge_type: "co_activated".to_string(),
        weight: delta.min(10.0),
        created_at: chrono::Utc::now(),
    });
}

/// Apply decay to all edges — simulates Hebbian weakening over time.
pub fn decay_all_edges(edges: &mut Vec<SynapticEdge>, factor: f32) {
    for edge in edges.iter_mut() {
        edge.weight *= factor;
    }
    // Remove edges below threshold
    edges.retain(|e| e.weight > 0.01);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strengthen() {
        let mut edges = vec![SynapticEdge {
            source: "a".into(), target: "b".into(), edge_type: "link".into(),
            weight: 1.0, created_at: chrono::Utc::now(),
        }];
        strengthen_edge(&mut edges, "a", "b", 0.5);
        assert!((edges[0].weight - 1.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_decay() {
        let mut edges = vec![SynapticEdge {
            source: "a".into(), target: "b".into(), edge_type: "link".into(),
            weight: 1.0, created_at: chrono::Utc::now(),
        }];
        decay_all_edges(&mut edges, 0.5);
        assert!((edges[0].weight - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_decay_removes_weak() {
        let mut edges = vec![SynapticEdge {
            source: "a".into(), target: "b".into(), edge_type: "link".into(),
            weight: 0.005, created_at: chrono::Utc::now(),
        }];
        decay_all_edges(&mut edges, 0.5);
        assert!(edges.is_empty());
    }
}
