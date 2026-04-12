// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Tests for the synthetic distillation tool.

use super::*;

#[allow(dead_code)]
fn make_call(args: serde_json::Value) -> ToolCall {
    ToolCall {
        id: "t".to_string(),
        name: "distill_knowledge".to_string(),
        arguments: args,
    }
}

#[test]
fn test_build_distillation_prompt() {
    let prompt = build_distillation_prompt("Rust async programming", 3);
    assert!(prompt.contains("Rust async programming"));
    assert!(prompt.contains("3"));
    assert!(prompt.contains("JSON array"));
}

#[test]
fn test_parse_qa_pairs_valid() {
    let response = r#"[
        {"question": "What is ownership?", "answer": "Ownership is Rust's memory management system."},
        {"question": "What is borrowing?", "answer": "Borrowing lets you reference data without taking ownership."}
    ]"#;
    let pairs = parse_qa_pairs(response).unwrap();
    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs[0].0, "What is ownership?");
    assert!(pairs[1].1.contains("Borrowing"));
}

#[test]
fn test_parse_qa_pairs_with_code_fence() {
    let response = "Here are the Q&A pairs:\n```json\n[\n{\"question\": \"Q1?\", \"answer\": \"A1.\"}\n]\n```";
    let pairs = parse_qa_pairs(response).unwrap();
    assert_eq!(pairs.len(), 1);
}

#[test]
fn test_parse_qa_pairs_with_plain_fence() {
    let response = "```\n[{\"question\": \"Q?\", \"answer\": \"A.\"}]\n```";
    let pairs = parse_qa_pairs(response).unwrap();
    assert_eq!(pairs.len(), 1);
}

#[test]
fn test_parse_qa_pairs_embedded_in_text() {
    let response = "Here you go:\n[{\"question\": \"Q?\", \"answer\": \"A.\"}]\nDone!";
    let pairs = parse_qa_pairs(response).unwrap();
    assert_eq!(pairs.len(), 1);
}

#[test]
fn test_parse_qa_pairs_empty_fields_skipped() {
    let response = r#"[
        {"question": "", "answer": "valid answer"},
        {"question": "valid question", "answer": ""},
        {"question": "good q", "answer": "good a"}
    ]"#;
    let pairs = parse_qa_pairs(response).unwrap();
    assert_eq!(pairs.len(), 1);
    assert_eq!(pairs[0].0, "good q");
}

#[test]
fn test_parse_qa_pairs_no_json() {
    let response = "This is just plain text with no JSON.";
    assert!(parse_qa_pairs(response).is_err());
}

#[test]
fn test_parse_qa_pairs_invalid_json() {
    let response = "[{bad json here}]";
    assert!(parse_qa_pairs(response).is_err());
}

#[test]
fn test_extract_json_array_direct() {
    let result = extract_json_array("[1, 2, 3]").unwrap();
    assert_eq!(result, "[1, 2, 3]");
}

#[test]
fn test_extract_json_array_code_fence() {
    let text = "text\n```json\n[1]\n```\nmore text";
    let result = extract_json_array(text).unwrap();
    assert_eq!(result, "[1]");
}

#[test]
fn test_distill_config_defaults() {
    // Without env var set, training is disabled
    std::env::remove_var("ERNOS_TRAINING_ENABLED");
    let config = DistillConfig::from_env();
    assert!(!config.training_enabled);
}
