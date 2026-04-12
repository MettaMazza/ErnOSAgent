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
        ReactEvent::ToolExecuting { name: "search".to_string(), id: "t1".to_string() },
        ReactEvent::AuditRunning,
        ReactEvent::AuditCompleted { verdict: Verdict::Allowed, reason: "none".to_string() },
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
    };
    if let ReactEvent::AuditCompleted { verdict, reason } = event {
        assert!(!verdict.is_allowed());
        assert!(reason.contains("Safety"));
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
