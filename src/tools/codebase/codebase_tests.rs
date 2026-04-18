// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Tests for codebase tools.

use super::read;
use super::write;
use super::*;
use std::path::Path;
use tempfile::TempDir;

fn make_call(name: &str, args: serde_json::Value) -> crate::tools::schema::ToolCall {
    crate::tools::schema::ToolCall {
        id: "test-1".to_string(),
        name: name.to_string(),
        arguments: args,
    }
}

// ── codebase_read ──────────────────────────────────────────────

#[test]
fn read_existing_file() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(
        tmp.path().join("hello.rs"),
        "fn main() {\n    println!(\"hello\");\n}\n",
    )
    .unwrap();
    let _root = TestRoot::new(tmp);

    let call = make_call("codebase_read", serde_json::json!({"path": "hello.rs"}));
    let result = read::read_file(&call);
    assert!(result.success, "Expected success, got: {:?}", result.error);
    assert!(result.output.contains("fn main()"));
    assert!(result.output.contains("3 lines"));
}

#[test]
fn read_with_line_range() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(
        tmp.path().join("lines.txt"),
        "line1\nline2\nline3\nline4\nline5\n",
    )
    .unwrap();
    let _root = TestRoot::new(tmp);

    let call = make_call(
        "codebase_read",
        serde_json::json!({"path": "lines.txt", "start_line": 2, "end_line": 4}),
    );
    let result = read::read_file(&call);
    assert!(result.success);
    assert!(result.output.contains("line2"));
    assert!(result.output.contains("line4"));
    assert!(!result.output.contains("line1"));
    assert!(!result.output.contains("line5"));
}

#[test]
fn read_missing_file() {
    let tmp = TempDir::new().unwrap();
    let _root = TestRoot::new(tmp);

    let call = make_call(
        "codebase_read",
        serde_json::json!({"path": "nonexistent.rs"}),
    );
    let result = read::read_file(&call);
    assert!(!result.success);
    assert!(result.error.as_ref().unwrap().contains("not found"));
}

#[test]
fn read_containment_blocked() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("Dockerfile"), "FROM rust").unwrap();
    let _root = TestRoot::new(tmp);

    let call = make_call("codebase_read", serde_json::json!({"path": "Dockerfile"}));
    let result = read::read_file(&call);
    assert!(!result.success);
    assert!(result.error.as_ref().unwrap().contains("BLOCKED"));
}

#[test]
fn read_path_traversal_blocked() {
    let tmp = TempDir::new().unwrap();
    let _root = TestRoot::new(tmp);

    let call = make_call(
        "codebase_read",
        serde_json::json!({"path": "../../etc/passwd"}),
    );
    let result = read::read_file(&call);
    assert!(!result.success);
    assert!(result.error.as_ref().unwrap().contains("traversal"));
}

// ── codebase_write ─────────────────────────────────────────────

#[test]
fn write_new_file() {
    let tmp = TempDir::new().unwrap();
    let root = TestRoot::new(tmp);

    let call = make_call(
        "codebase_write",
        serde_json::json!({
            "path": "src/new_module.rs",
            "content": "pub fn hello() -> &'static str {\n    \"world\"\n}\n"
        }),
    );
    let result = write::write_file(&call);
    assert!(result.success, "Expected success, got: {:?}", result.error);
    assert!(result.output.contains("Created"));

    let written = std::fs::read_to_string(root.path().join("src/new_module.rs")).unwrap();
    assert!(written.contains("hello"));
}

#[test]
fn write_containment_blocked() {
    let tmp = TempDir::new().unwrap();
    let _root = TestRoot::new(tmp);

    let call = make_call(
        "codebase_write",
        serde_json::json!({
            "path": "Dockerfile",
            "content": "FROM evil"
        }),
    );
    let result = write::write_file(&call);
    assert!(!result.success);
    assert!(result.error.as_ref().unwrap().contains("BLOCKED"));
}

#[test]
fn write_git_internal_blocked() {
    let tmp = TempDir::new().unwrap();
    let _root = TestRoot::new(tmp);

    let call = make_call(
        "codebase_write",
        serde_json::json!({
            "path": ".git/config",
            "content": "bad"
        }),
    );
    let result = write::write_file(&call);
    assert!(!result.success);
    assert!(result.error.as_ref().unwrap().contains(".git/"));
}

// ── codebase_patch ─────────────────────────────────────────────

#[test]
fn patch_replaces_text() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("code.rs"), "fn old_name() {}\n").unwrap();
    let root = TestRoot::new(tmp);

    let call = make_call(
        "codebase_patch",
        serde_json::json!({
            "path": "code.rs",
            "search": "old_name",
            "replace": "new_name"
        }),
    );
    let result = write::patch_file(&call);
    assert!(result.success, "Expected success, got: {:?}", result.error);
    assert!(result.output.contains("1 occurrence"));

    let patched = std::fs::read_to_string(root.path().join("code.rs")).unwrap();
    assert!(patched.contains("new_name"));
    assert!(!patched.contains("old_name"));
}

#[test]
fn patch_search_not_found() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("code.rs"), "fn main() {}\n").unwrap();
    let _root = TestRoot::new(tmp);

    let call = make_call(
        "codebase_patch",
        serde_json::json!({
            "path": "code.rs",
            "search": "nonexistent_function",
            "replace": "replacement"
        }),
    );
    let result = write::patch_file(&call);
    assert!(!result.success);
    assert!(result.error.as_ref().unwrap().contains("not found"));
}

#[test]
fn patch_multiple_occurrences() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("code.rs"), "aaa bbb aaa ccc aaa\n").unwrap();
    let _root = TestRoot::new(tmp);

    let call = make_call(
        "codebase_patch",
        serde_json::json!({
            "path": "code.rs",
            "search": "aaa",
            "replace": "zzz"
        }),
    );
    let result = write::patch_file(&call);
    assert!(result.success);
    assert!(result.output.contains("3 occurrence"));
}

// ── Registration ───────────────────────────────────────────────

#[test]
fn register_all_tools() {
    let mut executor = crate::tools::executor::ToolExecutor::new();
    register_tools(&mut executor);
    assert!(executor.has_tool("codebase_read"));
    assert!(executor.has_tool("codebase_write"));
    assert!(executor.has_tool("codebase_patch"));
}

// ── Test helper: thread-local project root ─────────────────────

struct TestRoot {
    _tmp: TempDir,
}

impl TestRoot {
    fn new(tmp: TempDir) -> Self {
        super::set_project_root(tmp.path());
        Self { _tmp: tmp }
    }

    fn path(&self) -> &Path {
        self._tmp.path()
    }
}

impl Drop for TestRoot {
    fn drop(&mut self) {
        super::clear_project_root();
    }
}
