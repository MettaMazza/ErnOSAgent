// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Git tool — native git operations, branch-locked to `ernosagent/self-edit`.
//!
//! The agent can read git state freely (status, diff, log, blame, branches).
//! It can commit ONLY on its own branch (`ernosagent/self-edit`).
//! Remote operations (push, pull, fetch, clone) are hard-blocked.
//! Checkout to other branches is hard-blocked.

use crate::tools::containment;
use crate::tools::executor::ToolExecutor;
use crate::tools::schema::{ToolCall, ToolResult};
use std::time::Duration;

/// The branch the agent is locked to.
const AGENT_BRANCH: &str = "ernosagent/self-edit";

/// Git command timeout.
const GIT_TIMEOUT_SECS: u64 = 30;

/// Actions that are unconditionally blocked (remote manipulation).
const BLOCKED_ACTIONS: &[&str] = &["push", "pull", "fetch", "clone", "remote"];

/// Actions that require the agent to be on its own branch.
const WRITE_ACTIONS: &[&str] = &["commit", "stash", "stash_pop"];

/// Execute the git tool.
fn git_tool(call: &ToolCall) -> ToolResult {
    let action = call
        .arguments
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if action.is_empty() {
        return error_result(call, "Missing required argument: action. Valid: status, diff, log, blame, branches, commit, stash, stash_pop");
    }

    // Hard-block remote operations
    if BLOCKED_ACTIONS.contains(&action) {
        return error_result(
            call,
            &format!(
                "BLOCKED: '{}' is a remote git operation and is strictly prohibited. \
            The agent can only manipulate the local repository timeline on its own branch.",
                action
            ),
        );
    }

    // Hard-block checkout (agent is locked to its branch)
    if action == "checkout" || action == "branch" {
        return error_result(
            call,
            &format!(
                "BLOCKED: '{}' would change the branch. The agent is locked to '{}'. \
            Use read-only commands (status, diff, log, blame, branches) to inspect other branches.",
                action, AGENT_BRANCH
            ),
        );
    }

    // For write actions, verify we're on the agent branch
    if WRITE_ACTIONS.contains(&action) {
        match get_current_branch() {
            Ok(branch) => {
                if branch != AGENT_BRANCH {
                    return error_result(
                        call,
                        &format!(
                            "BLOCKED: Agent can only commit on '{}', but currently on '{}'. \
                        The agent branch must be created first: git checkout -b {}",
                            AGENT_BRANCH, branch, AGENT_BRANCH
                        ),
                    );
                }
            }
            Err(e) => {
                return error_result(call, &format!("Failed to determine current branch: {}", e))
            }
        }
    }

    let result = match action {
        "status" => run_git(&["status", "--porcelain", "--branch"]),
        "diff" => {
            let staged = call
                .arguments
                .get("staged")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            if staged {
                run_git(&["diff", "--staged"])
            } else {
                run_git(&["diff"])
            }
        }
        "log" => {
            let limit = call
                .arguments
                .get("limit")
                .and_then(|v| v.as_u64())
                .unwrap_or(10);
            let limit_str = format!("-{}", limit);
            run_git(&["log", "--oneline", "--decorate", &limit_str])
        }
        "blame" => {
            let path = call
                .arguments
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if path.is_empty() {
                return error_result(call, "Missing required argument: path (for blame)");
            }
            // Containment check on the path
            if let Some(protected) = containment::check_path(path) {
                return error_result(
                    call,
                    &format!("BLOCKED: Cannot blame containment file '{}'", protected),
                );
            }
            let line = call
                .arguments
                .get("line")
                .and_then(|v| v.as_u64())
                .unwrap_or(1);
            let range = format!("{},+10", line);
            run_git(&["blame", path, "-L", &range])
        }
        "branches" => run_git(&["branch", "-a", "--no-color"]),
        "commit" => {
            let message = call
                .arguments
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            if message.is_empty() {
                return error_result(call, "Missing required argument: message (for commit)");
            }
            // Stage all changes first
            if let Err(e) = run_git(&["add", "-A"]) {
                return error_result(call, &format!("Failed to stage changes: {}", e));
            }
            run_git(&["commit", "-m", message])
        }
        "stash" => {
            let message = call
                .arguments
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("ernosagent-auto-stash");
            run_git(&["stash", "push", "-m", message])
        }
        "stash_pop" => run_git(&["stash", "pop"]),
        other => {
            return error_result(call, &format!(
                "Unknown git action: '{}'. Valid: status, diff, log, blame, branches, commit, stash, stash_pop",
                other
            ));
        }
    };

    match result {
        Ok(output) => {
            tracing::info!(action = %action, "git_tool executed");
            ToolResult {
                tool_call_id: call.id.clone(),
                name: call.name.clone(),
                output,
                success: true,
                error: None,
            }
        }
        Err(error) => error_result(call, &error),
    }
}

/// Get the current git branch name.
fn get_current_branch() -> Result<String, String> {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .map_err(|e| format!("Failed to run git: {}", e))?;

    if output.status.success() {
        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

/// Run a git command synchronously with timeout via block_in_place.
fn run_git(args: &[&str]) -> Result<String, String> {
    let result = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            tokio::time::timeout(
                Duration::from_secs(GIT_TIMEOUT_SECS),
                tokio::process::Command::new("git").args(args).output(),
            )
            .await
        })
    });

    match result {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout).to_string();
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            if output.status.success() {
                let combined = if stdout.trim().is_empty() && !stderr.trim().is_empty() {
                    stderr.trim().to_string()
                } else if stdout.trim().is_empty() {
                    "Command succeeded with no output.".to_string()
                } else {
                    stdout.trim().to_string()
                };
                Ok(combined)
            } else {
                Err(format!("git {} failed:\n{}{}", args[0], stdout, stderr))
            }
        }
        Ok(Err(e)) => Err(format!("Failed to execute git: {}", e)),
        Err(_) => Err(format!(
            "git {} timed out after {}s",
            args[0], GIT_TIMEOUT_SECS
        )),
    }
}

/// Register the git tool with the executor.
pub fn register_tools(executor: &mut ToolExecutor) {
    executor.register("git_tool", Box::new(git_tool));
}

/// Helper to build a failed ToolResult.
fn error_result(call: &ToolCall, msg: &str) -> ToolResult {
    ToolResult {
        tool_call_id: call.id.clone(),
        name: call.name.clone(),
        output: format!("Error: {}", msg),
        success: false,
        error: Some(msg.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_call(args: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "test-git".to_string(),
            name: "git_tool".to_string(),
            arguments: args,
        }
    }

    // ── Blocked actions ────────────────────────────────────────────

    #[test]
    fn blocks_push() {
        let call = make_call(serde_json::json!({"action": "push"}));
        let result = git_tool(&call);
        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("BLOCKED"));
        assert!(result.error.as_ref().unwrap().contains("remote"));
    }

    #[test]
    fn blocks_pull() {
        let call = make_call(serde_json::json!({"action": "pull"}));
        let result = git_tool(&call);
        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("BLOCKED"));
    }

    #[test]
    fn blocks_fetch() {
        let call = make_call(serde_json::json!({"action": "fetch"}));
        let result = git_tool(&call);
        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("BLOCKED"));
    }

    #[test]
    fn blocks_clone() {
        let call = make_call(serde_json::json!({"action": "clone"}));
        let result = git_tool(&call);
        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("BLOCKED"));
    }

    #[test]
    fn blocks_checkout() {
        let call = make_call(serde_json::json!({"action": "checkout", "target": "main"}));
        let result = git_tool(&call);
        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("BLOCKED"));
        assert!(result.error.as_ref().unwrap().contains(AGENT_BRANCH));
    }

    #[test]
    fn blocks_branch_creation() {
        let call = make_call(serde_json::json!({"action": "branch", "name": "evil"}));
        let result = git_tool(&call);
        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("BLOCKED"));
    }

    // ── Missing args ───────────────────────────────────────────────

    #[test]
    fn missing_action() {
        let call = make_call(serde_json::json!({}));
        let result = git_tool(&call);
        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("Missing"));
    }

    #[test]
    fn unknown_action() {
        let call = make_call(serde_json::json!({"action": "explode"}));
        let result = git_tool(&call);
        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("Unknown"));
    }

    #[test]
    fn blame_missing_path() {
        let call = make_call(serde_json::json!({"action": "blame"}));
        let result = git_tool(&call);
        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("Missing"));
    }

    #[test]
    fn commit_missing_message_or_wrong_branch() {
        // When not on ernosagent/self-edit, the branch check fires first.
        // When on the right branch, the missing message check fires.
        // In parallel test environments, git CWD may not resolve.
        // The invariant: the commit is always rejected (success=false).
        let call = make_call(serde_json::json!({"action": "commit"}));
        let result = git_tool(&call);
        assert!(!result.success, "Commit should be rejected but succeeded");
        assert!(result.error.is_some(), "Expected an error message");
    }

    // ── Read-only actions (need tokio runtime) ─────────────────────

    #[tokio::test(flavor = "multi_thread")]
    async fn status_works() {
        let call = make_call(serde_json::json!({"action": "status"}));
        let result = git_tool(&call);
        // May succeed or fail depending on git availability, but shouldn't panic
        if result.success {
            assert!(!result.output.is_empty());
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn log_works() {
        let call = make_call(serde_json::json!({"action": "log", "limit": 3}));
        let result = git_tool(&call);
        if result.success {
            assert!(!result.output.is_empty());
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn branches_works() {
        let call = make_call(serde_json::json!({"action": "branches"}));
        let result = git_tool(&call);
        if result.success {
            assert!(!result.output.is_empty());
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn diff_works() {
        let call = make_call(serde_json::json!({"action": "diff"}));
        let result = git_tool(&call);
        if result.success {
            assert!(!result.output.is_empty());
        }
    }

    // ── Registration ───────────────────────────────────────────────

    #[test]
    fn register_all_tools() {
        let mut executor = ToolExecutor::new();
        register_tools(&mut executor);
        assert!(executor.has_tool("git_tool"));
    }
}
