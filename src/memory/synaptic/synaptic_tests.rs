// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Tests for the Synaptic Knowledge Graph.

use super::*;

#[tokio::test]
async fn test_store_and_search() {
    let graph = SynapticGraph::new(None);

    graph.store("Apple", "A red fruit").await;
    graph.store("Apple", "Grows on trees").await;
    graph.store("Banana", "A yellow fruit").await;

    let results = graph.search("Apple").await;
    assert_eq!(results.len(), 2);
    assert!(results[0].contains("A red fruit"));
    assert!(results[1].contains("Grows on trees"));

    let results = graph.search("app").await;
    assert_eq!(results.len(), 2);

    let results = graph.search("Watermelon").await;
    assert!(results.is_empty());
}

#[tokio::test]
async fn test_deduplicate_store() {
    let graph = SynapticGraph::new(None);
    graph.store("Apple", "A red fruit").await;
    graph.store("Apple", "A red fruit").await;
    let results = graph.search("Apple").await;
    assert_eq!(results.len(), 1);
}

#[tokio::test]
async fn test_beliefs() {
    let graph = SynapticGraph::new(None);
    graph.store("Apple", "fact 1").await;
    graph.store("Apple", "fact 2").await;
    graph.store("Apple", "fact 3").await;
    graph.store("Banana", "fact A").await;

    let beliefs = graph.get_beliefs(10).await;
    assert_eq!(beliefs.len(), 2);
    assert!(beliefs[0].starts_with("Apple"));
    assert!(beliefs[1].starts_with("Banana"));
}

#[tokio::test]
async fn test_recent_nodes() {
    let graph = SynapticGraph::new(None);
    graph.store("First", "data").await;
    graph.store("Second", "data").await;

    let recent = graph.get_recent_nodes(1).await;
    assert_eq!(recent.len(), 1);
    assert_eq!(recent[0].0, "Second");
}

#[tokio::test]
async fn test_relationships() {
    let graph = SynapticGraph::new(None);
    graph.store_relationship("Apple", "is_a", "Fruit").await;
    graph.store_relationship("Banana", "is_a", "Fruit").await;

    let rels = graph.get_recent_relationships(10).await;
    assert_eq!(rels.len(), 2);
    assert_eq!(rels[0].0, "Banana");
    assert_eq!(rels[0].1, "is_a");
    assert_eq!(rels[0].2, "Fruit");
}

#[tokio::test]
async fn test_deduplicate_edges() {
    let graph = SynapticGraph::new(None);
    graph.store_relationship("Apple", "is_a", "Fruit").await;
    graph.store_relationship("Apple", "is_a", "Fruit").await;
    let rels = graph.get_recent_relationships(10).await;
    assert_eq!(rels.len(), 1);
}

#[tokio::test]
async fn test_stats() {
    let graph = SynapticGraph::new(None);
    graph.store("A", "1").await;
    graph.store("B", "2").await;
    graph.store_relationship("A", "related", "B").await;
    let (nodes, edges) = graph.stats().await;
    assert_eq!(nodes, 2);
    assert_eq!(edges, 1);
}

#[tokio::test]
async fn test_persistence_with_tempdir() {
    let tmp = std::env::temp_dir().join(format!("ernosagent_synaptic_test_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);

    {
        let graph = SynapticGraph::new(Some(tmp.clone()));
        graph.store("Maria", "is the creator").await;
        graph.store("Maria", "believes in transparent AI").await;
        graph.store_relationship("Maria", "created", "ErnOS").await;
    }

    {
        let graph = SynapticGraph::new(Some(tmp.clone()));
        graph.load().await;

        let results = graph.search("Maria").await;
        assert_eq!(results.len(), 2);
        assert!(results[0].contains("creator"));

        let rels = graph.get_recent_relationships(10).await;
        assert_eq!(rels.len(), 1);
        assert_eq!(rels[0].0, "Maria");
    }

    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn test_default_layers_and_root_mesh() {
    let graph = SynapticGraph::new(None);
    graph.initialize().await;

    let layers = graph.list_layers().await;
    assert_eq!(layers.len(), 6);
    assert!(layers.iter().any(|l| l.name == "self"));
    assert!(layers.iter().any(|l| l.name == "people"));
    assert!(layers.iter().any(|l| l.name == "places"));
    assert!(layers.iter().any(|l| l.name == "concepts"));
    assert!(layers.iter().any(|l| l.name == "projects"));
    assert!(layers.iter().any(|l| l.name == "environment"));

    let (node_count, edge_count) = graph.stats().await;
    assert_eq!(node_count, 6);
    assert_eq!(edge_count, 15);

    let edges = graph.edges.read().await;
    for edge in edges.iter() {
        assert!(edge.permanent);
        assert_eq!(edge.weight, 1.0);
        assert_eq!(edge.relation, "root_mesh");
    }
}

#[tokio::test]
async fn test_hebbian_strengthening() {
    let graph = SynapticGraph::new(None);
    graph.store_relationship("A", "likes", "B").await;

    graph.strengthen_edge("A", "B").await;
    graph.strengthen_edge("A", "B").await;
    graph.strengthen_edge("A", "B").await;

    let edges = graph.edges.read().await;
    let edge = edges.iter().find(|e| e.from == "A" && e.to == "B").unwrap();
    assert!(edge.permanent);
    assert!(edge.weight > 0.6);
}

#[tokio::test]
async fn test_contradiction_detection() {
    let graph = SynapticGraph::new(None);
    graph.store_relationship("Earth", "IS_A", "Planet").await;

    let result = graph.check_contradiction("Earth", "IS_A", "Planet").await;
    assert!(result.is_none());

    let result = graph.check_contradiction("Earth", "IS_A", "Star").await;
    assert_eq!(result, Some("Planet".to_string()));
}

#[tokio::test]
async fn test_decay() {
    let graph = SynapticGraph::new(None);
    graph.store_relationship("A", "likes", "B").await;

    let (decayed, pruned, permanent) = graph.decay_all(0.5).await;
    assert_eq!(decayed, 1);
    assert_eq!(pruned, 0);
    assert_eq!(permanent, 0);

    let edges = graph.edges.read().await;
    let edge = edges.iter().find(|e| e.from == "A" && e.to == "B").unwrap();
    assert!((edge.weight - 0.25).abs() < 0.01);
}

#[tokio::test]
async fn test_co_activation() {
    let graph = SynapticGraph::new(None);
    let labels = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
    graph.co_activate(&labels).await;

    let edges = graph.edges.read().await;
    // 3 nodes → 3 pairs × 2 directions = 6 edges
    assert_eq!(edges.len(), 6);
}

#[tokio::test]
async fn test_shortcut_creation() {
    let graph = SynapticGraph::new(None);
    graph.create_shortcut("source", "target").await;

    let edges = graph.edges.read().await;
    assert_eq!(edges.len(), 1);
    let edge = &edges[0];
    assert_eq!(edge.from, "source");
    assert_eq!(edge.to, "target");
    assert_eq!(edge.relation, "shortcut");
    assert!((edge.weight - 0.3).abs() < 0.001);
    assert!(!edge.permanent);
}

#[tokio::test]
async fn test_shortcut_no_duplicate() {
    let graph = SynapticGraph::new(None);
    graph.create_shortcut("A", "B").await;
    graph.create_shortcut("A", "B").await;
    let edges = graph.edges.read().await;
    assert_eq!(edges.len(), 1, "Duplicate shortcut should not be added");
}

#[tokio::test]
async fn test_decay_preserves_permanent() {
    let graph = SynapticGraph::new(None);
    graph.store_relationship("P", "always_links", "Q").await;
    // Strengthen 3 times to make permanent
    graph.strengthen_edge("P", "Q").await;
    graph.strengthen_edge("P", "Q").await;
    graph.strengthen_edge("P", "Q").await;

    let (_decayed, _, permanent) = graph.decay_all(0.01).await;
    assert_eq!(permanent, 1, "Should have 1 permanent edge");
    // The original relationship edge may decay, the strengthened one is permanent
    // The original relationship edge may decay, the strengthened one is permanent
}

#[tokio::test]
async fn test_search_case_insensitive() {
    let graph = SynapticGraph::new(None);
    graph.store("Rust", "A systems language").await;
    let results = graph.search("rust").await;
    assert_eq!(results.len(), 1);
    assert!(results[0].contains("systems language"));
}
