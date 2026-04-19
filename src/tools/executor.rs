// Ern-OS — Tool execution dispatcher
// This module provides standalone tool execution for shell and web search.
// Memory tools are dispatched directly through AppState in ws.rs.

use crate::tools::schema::{ToolCall, ToolResult};
use anyhow::Result;

/// Dispatch and execute a tool call (stateless tools only).
/// Memory tools are dispatched through ws.rs with AppState access.
pub async fn execute(tool_call: &ToolCall) -> Result<ToolResult> {
    tracing::info!(tool = "executor", "tool START");
    let args = tool_call.args();

    let result = match tool_call.name.as_str() {
        "run_bash_command" => {
            let cmd = args["command"].as_str().unwrap_or("");
            let wd = args["working_dir"].as_str();
            super::shell::run_command(cmd, wd).await
        }
        "web_search" => {
            let action = args["action"].as_str().unwrap_or("search");
            match action {
                "visit" => {
                    let url = args["url"].as_str().unwrap_or("");
                    super::web_search::visit(url).await
                }
                _ => {
                    let query = args["query"].as_str().unwrap_or("");
                    super::web_search::search(query).await
                }
            }
        }
        other => {
            Ok(format!("Tool '{}' requires state access — dispatch through ws handler", other))
        }
    };

    match result {
        Ok(output) => Ok(ToolResult {
            tool_call_id: tool_call.id.clone(),
            name: tool_call.name.clone(),
            output,
            success: true,
        }),
        Err(e) => Ok(ToolResult {
            tool_call_id: tool_call.id.clone(),
            name: tool_call.name.clone(),
            output: format!("Error: {}", e),
            success: false,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_execute_unknown_tool() {
        let tc = ToolCall {
            id: "1".into(), name: "unknown_tool".into(),
            arguments: "{}".into(),
        };
        let result = execute(&tc).await.unwrap();
        assert!(result.success);
    }
}
