// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Tests for the Turing Grid tool.

use super::*;

fn make_call(action: &str, extra: serde_json::Value) -> ToolCall {
    let mut args = serde_json::json!({"action": action});
    if let serde_json::Value::Object(map) = extra {
        for (k, v) in map {
            args[k] = v;
        }
    }
    ToolCall {
        id: "test-turing".to_string(),
        name: "operate_turing_grid".to_string(),
        arguments: args,
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_move_and_read() {
    let state = TuringState::new_test();
    let call = make_call("move", serde_json::json!({"dx": 1, "dy": 2, "dz": 3}));
    let result = execute_turing_tool(&call, &state);
    assert!(result.success);
    assert!(result.output.contains("(1, 2, 3)"));

    let call = make_call("read", serde_json::json!({}));
    let result = execute_turing_tool(&call, &state);
    assert!(result.success);
    assert!(result.output.contains("empty"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_write_and_read() {
    let state = TuringState::new_test();
    let call = make_call(
        "write",
        serde_json::json!({"format": "text", "content": "hello grid"}),
    );
    let result = execute_turing_tool(&call, &state);
    assert!(result.success);
    assert!(result.output.contains("Successfully wrote"));

    let call = make_call("read", serde_json::json!({}));
    let result = execute_turing_tool(&call, &state);
    assert!(result.success);
    assert!(result.output.contains("hello grid"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_write_no_content() {
    let state = TuringState::new_test();
    let call = make_call("write", serde_json::json!({"format": "text"}));
    let result = execute_turing_tool(&call, &state);
    assert!(!result.success);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_index_empty() {
    let state = TuringState::new_test();
    let call = make_call("index", serde_json::json!({}));
    let result = execute_turing_tool(&call, &state);
    assert!(result.success);
    assert!(result.output.contains("empty"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_label_and_goto() {
    let state = TuringState::new_test();

    let call = make_call("move", serde_json::json!({"dx": 5, "dy": 5, "dz": 5}));
    execute_turing_tool(&call, &state);

    let call = make_call("label", serde_json::json!({"name": "home"}));
    let result = execute_turing_tool(&call, &state);
    assert!(result.success);
    assert!(result.output.contains("home"));
    assert!(result.output.contains("(5, 5, 5)"));

    let call = make_call("move", serde_json::json!({"dx": -5, "dy": -5, "dz": -5}));
    execute_turing_tool(&call, &state);

    let call = make_call("goto", serde_json::json!({"name": "home"}));
    let result = execute_turing_tool(&call, &state);
    assert!(result.success);
    assert!(result.output.contains("Jumped to label"));
    assert!(result.output.contains("(5, 5, 5)"));

    let call = make_call("goto", serde_json::json!({"name": "nonexistent"}));
    let result = execute_turing_tool(&call, &state);
    assert!(!result.success);
    assert!(result.output.contains("not found"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_history_and_undo() {
    let state = TuringState::new_test();

    let call = make_call(
        "write",
        serde_json::json!({"format": "text", "content": "v1"}),
    );
    execute_turing_tool(&call, &state);
    let call = make_call(
        "write",
        serde_json::json!({"format": "text", "content": "v2"}),
    );
    execute_turing_tool(&call, &state);

    let call = make_call("history", serde_json::json!({}));
    let result = execute_turing_tool(&call, &state);
    assert!(result.success);
    assert!(result.output.contains("1 entries"));
    assert!(result.output.contains("v1"));

    let call = make_call("undo", serde_json::json!({}));
    let result = execute_turing_tool(&call, &state);
    assert!(result.success);
    assert!(result.output.contains("Undo successful"));
    assert!(result.output.contains("v1"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_scan() {
    let state = TuringState::new_test();

    let call = make_call(
        "write",
        serde_json::json!({"format": "text", "content": "origin"}),
    );
    execute_turing_tool(&call, &state);

    let call = make_call("scan", serde_json::json!({"radius": 5}));
    let result = execute_turing_tool(&call, &state);
    assert!(result.success);
    assert!(result.output.contains("Radar Scan"));
    assert!(result.output.contains("(0, 0, 0)"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_unknown_action() {
    let state = TuringState::new_test();
    let call = make_call("invalid_action", serde_json::json!({}));
    let result = execute_turing_tool(&call, &state);
    assert!(!result.success);
    assert!(result.output.contains("Unknown action"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_execute_empty_cell() {
    let state = TuringState::new_test();
    let call = make_call("execute", serde_json::json!({}));
    let result = execute_turing_tool(&call, &state);
    assert!(!result.success);
    assert!(result.output.contains("empty"));
}
