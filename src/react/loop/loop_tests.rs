// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Tests for the ReAct loop.

use super::*;
use crate::observer::Verdict;

#[test]
fn test_react_event_variants() {
    let events = vec![
        ReactEvent::TurnStarted { turn: 1 },
        ReactEvent::Token("hello".to_string()),
        ReactEvent::Thinking("hmm".to_string()),
        ReactEvent::ToolExecuting { name: "search".to_string(), id: "t1".to_string(), arguments: "{}".to_string() },
        ReactEvent::AuditRunning,
        ReactEvent::AuditCompleted { verdict: Verdict::Allowed, reason: "none".to_string(), confidence: 0.95 },
        ReactEvent::ResponseReady { text: "hi".to_string() },
        ReactEvent::Error("oops".to_string()),
        ReactEvent::NeuralSnapshot(crate::interpretability::snapshot::simulate_snapshot(1, "test")),
    ];
    assert_eq!(events.len(), 9);
}

#[test]
fn test_react_config() {
    let config = ReactConfig {
        observer_enabled: true,
        observer_model: None,
        context_length: 262144,
    };
    assert!(config.observer_enabled);
}

#[test]
fn test_react_result() {
    let result = ReactResult {
        response: "Hello!".to_string(),
        turns: 2,
        tool_results: Vec::new(),
        audit_passes: 1,
        audit_rejections: 0,
    };
    assert_eq!(result.turns, 2);
    assert_eq!(result.audit_passes, 1);
}

#[test]
fn test_react_result_zero_turns() {
    let result = ReactResult {
        response: String::new(),
        turns: 0,
        tool_results: Vec::new(),
        audit_passes: 0,
        audit_rejections: 0,
    };
    assert!(result.response.is_empty());
    assert_eq!(result.turns, 0);
}

#[test]
fn test_react_config_observer_disabled() {
    let config = ReactConfig {
        observer_enabled: false,
        observer_model: Some("test".to_string()),
        context_length: 131072,
    };
    assert!(!config.observer_enabled);
    assert_eq!(config.observer_model.as_deref(), Some("test"));
}

#[test]
fn test_react_event_tool_completed() {
    use crate::tools::schema::ToolResult;
    let result = ToolResult {
        tool_call_id: "tc1".to_string(),
        name: "web_search".to_string(),
        output: "Found results".to_string(),
        success: true,
        error: None,
    };
    let event = ReactEvent::ToolCompleted {
        name: "web_search".to_string(),
        result: result.clone(),
    };
    if let ReactEvent::ToolCompleted { name, result: r } = event {
        assert_eq!(name, "web_search");
        assert!(r.success);
    }
}

#[test]
fn test_react_event_audit_rejected() {
    let event = ReactEvent::AuditCompleted {
        verdict: Verdict::Blocked,
        reason: "Safety violation".to_string(),
        confidence: 0.85,
    };
    if let ReactEvent::AuditCompleted { verdict, reason, confidence } = event {
        assert!(!verdict.is_allowed());
        assert!(reason.contains("Safety"));
        assert!(confidence > 0.0);
    }
}

#[test]
fn test_react_result_with_tools() {
    use crate::tools::schema::ToolResult;
    let result = ReactResult {
        response: "Done".to_string(),
        turns: 3,
        tool_results: vec![
            ToolResult {
                tool_call_id: "t1".to_string(),
                name: "search".to_string(),
                output: "ok".to_string(),
                success: true,
                error: None,
            },
            ToolResult {
                tool_call_id: "t2".to_string(),
                name: "write".to_string(),
                output: "".to_string(),
                success: false,
                error: Some("Permission denied".to_string()),
            },
        ],
        audit_passes: 2,
        audit_rejections: 1,
    };
    assert_eq!(result.tool_results.len(), 2);
    assert!(result.tool_results[0].success);
    assert!(!result.tool_results[1].success);
}

#[test]
fn test_tool_loop_signature_same_calls() {
    use crate::tools::schema::ToolCall;
    // Simulate the signature building logic from the ReAct loop
    fn build_sig(calls: &[ToolCall]) -> String {
        let mut parts: Vec<String> = calls.iter()
            .filter(|tc| !schema::is_reply_request(tc))
            .map(|tc| format!("{}:{}", tc.name, tc.arguments))
            .collect();
        parts.sort();
        parts.join("|")
    }

    let calls_a = vec![ToolCall {
        id: "1".to_string(),
        name: "memory_tool".to_string(),
        arguments: serde_json::json!({"action": "recall", "query": "spark"})
    }];
    let calls_b = vec![ToolCall {
        id: "2".to_string(),
        name: "memory_tool".to_string(),
        arguments: serde_json::json!({"action": "recall", "query": "spark"})
    }];
    // Same tool + same args = same signature (id is different but irrelevant)
    assert_eq!(build_sig(&calls_a), build_sig(&calls_b));
}

#[test]
fn test_tool_loop_signature_different_calls() {
    use crate::tools::schema::ToolCall;
    fn build_sig(calls: &[ToolCall]) -> String {
        let mut parts: Vec<String> = calls.iter()
            .filter(|tc| !schema::is_reply_request(tc))
            .map(|tc| format!("{}:{}", tc.name, tc.arguments))
            .collect();
        parts.sort();
        parts.join("|")
    }

    let calls_a = vec![ToolCall {
        id: "1".to_string(),
        name: "memory_tool".to_string(),
        arguments: serde_json::json!({"action": "recall", "query": "spark"})
    }];
    let calls_b = vec![ToolCall {
        id: "2".to_string(),
        name: "timeline_tool".to_string(),
        arguments: serde_json::json!({"action": "search", "query": "spark"})
    }];
    // Different tool = different signature
    assert_ne!(build_sig(&calls_a), build_sig(&calls_b));
}

#[test]
fn test_tool_loop_counter_logic() {
    // Simulate the counter tracking from the ReAct loop
    let mut last_sig: Option<String> = None;
    let mut count = 0_usize;

    // First call
    let sig1 = "memory_tool:{\"action\":\"recall\",\"query\":\"spark\"}".to_string();
    if Some(&sig1) == last_sig.as_ref() { count += 1; } else { last_sig = Some(sig1); count = 1; }
    assert_eq!(count, 1);

    // Second identical call
    let sig2 = "memory_tool:{\"action\":\"recall\",\"query\":\"spark\"}".to_string();
    if Some(&sig2) == last_sig.as_ref() { count += 1; } else { last_sig = Some(sig2); count = 1; }
    assert_eq!(count, 2);

    // Third identical call — triggers detection
    let sig3 = "memory_tool:{\"action\":\"recall\",\"query\":\"spark\"}".to_string();
    if Some(&sig3) == last_sig.as_ref() { count += 1; } else { last_sig = Some(sig3); count = 1; }
    assert_eq!(count, 3);
    assert!(count >= 3, "Should trigger tool-loop detection at count >= 3");

    // Different call resets
    let sig4 = "timeline_tool:{\"action\":\"search\",\"query\":\"spark\"}".to_string();
    if Some(&sig4) == last_sig.as_ref() { count += 1; } else { last_sig = Some(sig4); count = 1; }
    assert_eq!(count, 1, "Different tool should reset counter");
}

