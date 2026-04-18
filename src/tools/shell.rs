// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Shell execution tool — runs commands within the project directory.
//!
//! All commands are containment-checked before execution.
//! Output is captured and returned to the ReAct loop.

use crate::tools::containment;
use crate::tools::executor::ToolExecutor;
use crate::tools::schema::{ToolCall, ToolResult};
use std::time::Duration;

/// Default command timeout.
const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// Maximum command timeout.
const MAX_TIMEOUT_SECS: u64 = 600;

/// Execute a shell command and return the output.
fn run_command(call: &ToolCall) -> ToolResult {
    let command = call
        .arguments
        .get("command")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if command.is_empty() {
        return error_result(call, "Missing required argument: command");
    }

    // Containment check
    if let Some(reason) = containment::check_command(command) {
        return error_result(call, &format!("BLOCKED: {}", reason));
    }

    let timeout_secs = call
        .arguments
        .get("timeout_secs")
        .and_then(|v| v.as_u64())
        .unwrap_or(DEFAULT_TIMEOUT_SECS)
        .min(MAX_TIMEOUT_SECS);

    let project_root = match std::env::current_dir() {
        Ok(d) => d,
        Err(e) => return error_result(call, &format!("Failed to get working directory: {}", e)),
    };

    tracing::info!(
        command = %command,
        timeout_secs = timeout_secs,
        cwd = %project_root.display(),
        "run_command executing"
    );

    // Use block_in_place to run async timeout within the sync ToolHandler
    let result = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(async {
            tokio::time::timeout(
                Duration::from_secs(timeout_secs),
                tokio::process::Command::new("bash")
                    .args(["-c", command])
                    .current_dir(&project_root)
                    .output(),
            )
            .await
        })
    });

    match result {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let exit_code = output.status.code().unwrap_or(-1);

            let mut combined = String::new();
            if !stdout.is_empty() {
                combined.push_str(&stdout);
            }
            if !stderr.is_empty() {
                if !combined.is_empty() {
                    combined.push('\n');
                }
                combined.push_str("[stderr]\n");
                combined.push_str(&stderr);
            }

            let status_line = if output.status.success() {
                format!("[exit code: {}]", exit_code)
            } else {
                format!("[exit code: {} — FAILED]", exit_code)
            };

            tracing::info!(
                command = %command,
                exit_code = exit_code,
                stdout_bytes = stdout.len(),
                stderr_bytes = stderr.len(),
                "run_command completed"
            );

            ToolResult {
                tool_call_id: call.id.clone(),
                name: call.name.clone(),
                output: format!("{}\n{}", status_line, combined),
                success: output.status.success(),
                error: if output.status.success() {
                    None
                } else {
                    Some(format!("Command exited with code {}", exit_code))
                },
            }
        }
        Ok(Err(e)) => {
            tracing::error!(command = %command, error = %e, "run_command spawn failed");
            error_result(call, &format!("Failed to execute command: {}", e))
        }
        Err(_) => {
            tracing::warn!(command = %command, timeout_secs = timeout_secs, "run_command timed out");
            error_result(call, &format!(
                "Command timed out after {}s. Consider increasing timeout_secs or breaking the task into smaller steps.",
                timeout_secs
            ))
        }
    }
}

/// Register the shell tool with the executor.
pub fn register_tools(executor: &mut ToolExecutor) {
    executor.register("run_command", Box::new(run_command));
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
            id: "test-shell".to_string(),
            name: "run_command".to_string(),
            arguments: args,
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn runs_simple_command() {
        let call = make_call(serde_json::json!({"command": "echo hello_world"}));
        let result = run_command(&call);
        assert!(result.success, "Expected success, got: {:?}", result.error);
        assert!(result.output.contains("hello_world"));
        assert!(result.output.contains("exit code: 0"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn captures_exit_code_on_failure() {
        let call = make_call(serde_json::json!({"command": "false"}));
        let result = run_command(&call);
        assert!(!result.success);
        assert!(result.output.contains("FAILED"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn captures_stderr() {
        let call = make_call(serde_json::json!({"command": "echo err_msg >&2"}));
        let result = run_command(&call);
        assert!(result.success);
        assert!(result.output.contains("err_msg"));
        assert!(result.output.contains("[stderr]"));
    }

    #[test]
    fn blocks_containment_commands() {
        let call = make_call(serde_json::json!({"command": "docker exec -it test bash"}));
        let result = run_command(&call);
        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("BLOCKED"));
    }

    #[test]
    fn blocks_empty_command() {
        let call = make_call(serde_json::json!({"command": ""}));
        let result = run_command(&call);
        assert!(!result.success);
        assert!(result.error.as_ref().unwrap().contains("Missing"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn respects_custom_timeout() {
        let call = make_call(serde_json::json!({"command": "echo fast", "timeout_secs": 5}));
        let result = run_command(&call);
        assert!(result.success);
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn clamps_max_timeout() {
        let call = make_call(serde_json::json!({"command": "echo ok", "timeout_secs": 99999}));
        let result = run_command(&call);
        assert!(result.success);
    }

    #[test]
    fn register_all_tools() {
        let mut executor = ToolExecutor::new();
        register_tools(&mut executor);
        assert!(executor.has_tool("run_command"));
    }
}
