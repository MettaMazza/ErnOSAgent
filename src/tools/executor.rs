//! Tool executor — dispatches tool calls to their implementations.

use crate::tools::schema::{ToolCall, ToolResult};
use std::collections::HashMap;

/// Type alias for tool handler functions.
pub type ToolHandler = Box<dyn Fn(&ToolCall) -> ToolResult + Send + Sync>;

/// Registry and dispatcher for tool implementations.
pub struct ToolExecutor {
    handlers: HashMap<String, ToolHandler>,
}

impl ToolExecutor {
    pub fn new() -> Self {
        Self {
            handlers: HashMap::new(),
        }
    }

    /// Register a tool handler.
    pub fn register(&mut self, name: &str, handler: ToolHandler) {
        tracing::debug!(tool = %name, "Registered tool handler");
        self.handlers.insert(name.to_string(), handler);
    }

    /// Execute a tool call. Returns ToolResult with success or error.
    pub fn execute(&self, call: &ToolCall) -> ToolResult {
        tracing::info!(
            tool = %call.name,
            call_id = %call.id,
            "Executing tool"
        );

        match self.handlers.get(&call.name) {
            Some(handler) => {
                let result = handler(call);
                tracing::info!(
                    tool = %call.name,
                    success = result.success,
                    "Tool execution complete"
                );
                result
            }
            None => {
                tracing::warn!(tool = %call.name, "Unknown tool called");
                ToolResult {
                    tool_call_id: call.id.clone(),
                    name: call.name.clone(),
                    output: String::new(),
                    success: false,
                    error: Some(format!(
                        "Tool '{}' is not registered. Available tools: {}",
                        call.name,
                        self.available_tools().join(", ")
                    )),
                }
            }
        }
    }

    /// List all registered tool names.
    pub fn available_tools(&self) -> Vec<String> {
        let mut tools: Vec<String> = self.handlers.keys().cloned().collect();
        tools.sort();
        tools
    }

    /// Check if a tool is registered.
    pub fn has_tool(&self, name: &str) -> bool {
        self.handlers.contains_key(name)
    }

    /// Format all tool results for the observer's tool context section.
    pub fn format_tool_context(results: &[ToolResult]) -> String {
        if results.is_empty() {
            return "[No tools were executed]".to_string();
        }

        let mut lines = vec!["[TOOL EXECUTION RESULTS]".to_string()];
        for result in results {
            lines.push(result.format_for_observer());
        }
        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_execute() {
        let mut executor = ToolExecutor::new();
        executor.register("echo", Box::new(|call: &ToolCall| {
            let text = call.arguments.get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("default");
            ToolResult {
                tool_call_id: call.id.clone(),
                name: call.name.clone(),
                output: text.to_string(),
                success: true,
                error: None,
            }
        }));

        let call = ToolCall {
            id: "tc1".to_string(),
            name: "echo".to_string(),
            arguments: serde_json::json!({"text": "hello"}),
        };

        let result = executor.execute(&call);
        assert!(result.success);
        assert_eq!(result.output, "hello");
    }

    #[test]
    fn test_execute_unknown_tool() {
        let executor = ToolExecutor::new();
        let call = ToolCall {
            id: "tc1".to_string(),
            name: "nonexistent".to_string(),
            arguments: serde_json::json!({}),
        };

        let result = executor.execute(&call);
        assert!(!result.success);
        assert!(result.error.unwrap().contains("not registered"));
    }

    #[test]
    fn test_available_tools() {
        let mut executor = ToolExecutor::new();
        executor.register("b_tool", Box::new(|call: &ToolCall| ToolResult {
            tool_call_id: call.id.clone(), name: call.name.clone(),
            output: String::new(), success: true, error: None,
        }));
        executor.register("a_tool", Box::new(|call: &ToolCall| ToolResult {
            tool_call_id: call.id.clone(), name: call.name.clone(),
            output: String::new(), success: true, error: None,
        }));

        let tools = executor.available_tools();
        assert_eq!(tools, vec!["a_tool", "b_tool"]); // sorted
    }

    #[test]
    fn test_format_tool_context_empty() {
        let formatted = ToolExecutor::format_tool_context(&[]);
        assert!(formatted.contains("No tools were executed"));
    }

    #[test]
    fn test_format_tool_context_with_results() {
        let results = vec![
            ToolResult {
                tool_call_id: "t1".to_string(), name: "search".to_string(),
                output: "Found 3".to_string(), success: true, error: None,
            },
            ToolResult {
                tool_call_id: "t2".to_string(), name: "read".to_string(),
                output: String::new(), success: false, error: Some("Not found".to_string()),
            },
        ];
        let formatted = ToolExecutor::format_tool_context(&results);
        assert!(formatted.contains("✅"));
        assert!(formatted.contains("❌"));
    }
}
