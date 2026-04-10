// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Tests for the knowledge graph tier.

use super::*;
use chrono::Datelike;

#[test]
fn test_entity_struct() {
    let entity = Entity {
        id: "person:alice".to_string(),
        label: "Alice".to_string(),
        entity_type: "person".to_string(),
        properties: serde_json::json!({"role": "engineer"}),
        created_at: Utc::now(),
        last_accessed: Utc::now(),
        access_count: 5,
    };
    assert_eq!(entity.label, "Alice");
    assert_eq!(entity.access_count, 5);
}

#[test]
fn test_kg_relation_struct() {
    let rel = KgRelation {
        id: "r1".to_string(),
        source_id: "person:alice".to_string(),
        target_id: "concept:rust".to_string(),
        relation_type: "knows".to_string(),
        weight: 0.85,
        created_at: Utc::now(),
        last_reinforced: Utc::now(),
        reinforcement_count: 3,
    };
    assert_eq!(rel.relation_type, "knows");
    assert!(rel.weight > 0.8);
}

#[test]
fn test_recall_result_format() {
    let result = RecallResult {
        id: "concept:rust".to_string(),
        label: "Rust".to_string(),
        entity_type: "concept".to_string(),
        properties: serde_json::json!({"summary": "A systems programming language"}),
        weight: 0.75,
    };
    let formatted = result.format_for_context();
    assert!(formatted.contains("concept"));
    assert!(formatted.contains("Rust"));
    assert!(formatted.contains("0.750"));
}

#[test]
fn test_regex_escape() {
    assert_eq!(regex_escape("hello"), "hello");
    assert_eq!(regex_escape("a.b*c"), "a\\.b\\*c");
    assert_eq!(regex_escape("(test)"), "\\(test\\)");
}

#[test]
fn test_parse_datetime_valid() {
    let dt = parse_datetime("2026-01-15T10:30:00+00:00");
    assert_eq!(dt.year(), 2026);
}

#[test]
fn test_parse_datetime_invalid() {
    let dt = parse_datetime("not-a-date");
    // Should return now() on failure
    assert!(dt.year() >= 2026);
}

#[test]
fn test_entity_serialization() {
    let entity = Entity {
        id: "test:1".to_string(),
        label: "Test".to_string(),
        entity_type: "test".to_string(),
        properties: serde_json::json!({}),
        created_at: Utc::now(),
        last_accessed: Utc::now(),
        access_count: 0,
    };
    let json = serde_json::to_string(&entity).unwrap();
    let deserialized: Entity = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.id, "test:1");
}
