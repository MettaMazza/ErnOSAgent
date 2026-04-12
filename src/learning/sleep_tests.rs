// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Tests for the sleep cycle module.

use super::*;
use crate::learning::buffers::GoldenExample;

fn make_golden(response: &str, session: &str) -> GoldenExample {
    GoldenExample {
        system_prompt: "You are ErnOS.".to_string(),
        user_message: "Test question".to_string(),
        assistant_response: response.to_string(),
        session_id: session.to_string(),
        model_id: "test".to_string(),
        timestamp: chrono::Utc::now(),
    }
}

#[test]
fn test_quality_score_base() {
    let ex = make_golden("Short reply", "s1");
    let score = compute_quality_score(&ex, false);
    // First-pass bonus = 2.0, too short for goldilocks = 0
    assert!((score - 2.0).abs() < 0.01);
}

#[test]
fn test_quality_score_goldilocks() {
    let response = "a".repeat(500); // In the 200-2000 range
    let ex = make_golden(&response, "s1");
    let score = compute_quality_score(&ex, false);
    // First-pass 2.0 + goldilocks 1.0 = 3.0
    assert!((score - 3.0).abs() < 0.01);
}

#[test]
fn test_quality_score_verbose() {
    let response = "a".repeat(3000); // Over 2000
    let ex = make_golden(&response, "s1");
    let score = compute_quality_score(&ex, false);
    // First-pass 2.0 + verbose 0.5 = 2.5
    assert!((score - 2.5).abs() < 0.01);
}

#[test]
fn test_quality_score_recency() {
    let ex = make_golden("Short", "s1");
    let score_old = compute_quality_score(&ex, false);
    let score_new = compute_quality_score(&ex, true);
    assert!((score_new - score_old - 0.5).abs() < 0.01);
}

#[test]
fn test_quality_score_tool_usage() {
    let response = "I used the ✅ web_tool to search and got ✅ results.";
    let ex = make_golden(response, "s1");
    let score = compute_quality_score(&ex, false);
    // First-pass 2.0 + 2 tool refs * 0.5 = 3.0
    assert!((score - 3.0).abs() < 0.01);
}

#[test]
fn test_count_tool_references() {
    assert_eq!(count_tool_references("No tools here"), 0);
    assert_eq!(count_tool_references("✅ Done"), 1);
    assert_eq!(count_tool_references("✅ first ✅ second ❌ failed"), 3);
}

#[test]
fn test_sleep_status_display() {
    let ready = SleepStatus::Ready { golden_count: 10 };
    assert!(ready.to_string().contains("10 golden"));

    let disabled = SleepStatus::Disabled;
    assert_eq!(disabled.to_string(), "Disabled");
}

#[test]
fn test_sleep_config_missing_env() {
    // Without env vars set, config should fail
    std::env::remove_var("ERNOS_SLEEP_BATCH");
    let result = SleepConfig::from_env();
    assert!(result.is_err());
}
