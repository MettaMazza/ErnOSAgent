// Ern-OS — Tool call types and schema definitions

use serde::{Deserialize, Serialize};

/// A tool call from the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// A tool execution result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub name: String,
    pub output: String,
    pub success: bool,
}

impl ToolCall {
    /// Parse the arguments as JSON.
    pub fn args(&self) -> serde_json::Value {
        serde_json::from_str(&self.arguments).unwrap_or(serde_json::Value::Null)
    }
}

/// Check if a tool call terminates the inference loop.
pub fn is_loop_terminator(name: &str) -> bool {
    matches!(name, "reply_request" | "refuse_request")
}

/// Extract reply text from a reply_request tool call.
pub fn extract_reply_text(tool_call: &ToolCall) -> Option<String> {
    let args = tool_call.args();
    args["message"].as_str().map(|s| s.to_string())
        .or_else(|| args["reply"].as_str().map(|s| s.to_string()))
        .or_else(|| args["text"].as_str().map(|s| s.to_string()))
}

/// The start_react_system tool schema — triggers Layer 2.
pub fn start_react_system_tool() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "start_react_system",
            "description": "Escalate to the ReAct reasoning loop for complex, multi-step, or tool-heavy tasks that require planning and iteration. Use this when the task cannot be completed in a single response. You must estimate the number of tool turns you will need.",
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "The specific objective to accomplish in the ReAct loop"
                    },
                    "plan": {
                        "type": "string",
                        "description": "Initial plan or approach for achieving the objective"
                    },
                    "planned_turns": {
                        "type": "integer",
                        "description": "How many tool-call turns you estimate needing. Be realistic — you can extend if needed."
                    }
                },
                "required": ["objective", "planned_turns"]
            }
        }
    })
}

/// The reply_request tool — used within ReAct to deliver final answer.
pub fn reply_request_tool() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "reply_request",
            "description": "Deliver the final response to the user. Use this when you have completed your task and are ready to reply.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The complete response to send to the user"
                    }
                },
                "required": ["message"]
            }
        }
    })
}

/// The refuse_request tool — used within ReAct to decline a request.
pub fn refuse_request_tool() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "refuse_request",
            "description": "Decline a request with an explanation. Use only when you genuinely cannot complete the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Clear explanation of why the request cannot be fulfilled"
                    }
                },
                "required": ["reason"]
            }
        }
    })
}

/// The extend_turns tool — used to request more turns in the ReAct loop.
pub fn extend_turns_tool() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "extend_turns",
            "description": "Request additional turns when your turn budget is exhausted. You must assess your progress, explain what you still need to do, and estimate additional turns. Only available when your budget is exhausted.",
            "parameters": {
                "type": "object",
                "properties": {
                    "progress_summary": {
                        "type": "string",
                        "description": "What you have accomplished so far and what information you have gathered"
                    },
                    "remaining_work": {
                        "type": "string",
                        "description": "What specific steps still need to be completed"
                    },
                    "additional_turns": {
                        "type": "integer",
                        "description": "How many more tool-call turns you estimate needing to finish"
                    }
                },
                "required": ["progress_summary", "remaining_work", "additional_turns"]
            }
        }
    })
}

/// The propose_plan tool schema — creates an implementation plan for user approval before execution.
pub fn propose_plan_tool() -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": "propose_plan",
            "description": "Create a detailed implementation plan for a complex task and present it to the user for approval before execution. Use this for significant work (code changes, architecture decisions, multi-file edits, research, system modifications) that the user should review before you proceed. The plan will be rendered as rich markdown. Only after the user approves will the ReAct loop execute it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short descriptive title for the plan"
                    },
                    "plan_markdown": {
                        "type": "string",
                        "description": "Full implementation plan in markdown. Include: goal summary, proposed changes (grouped by component/file), verification steps, and any open questions."
                    },
                    "estimated_turns": {
                        "type": "integer",
                        "description": "Estimated number of ReAct tool turns needed to execute this plan"
                    }
                },
                "required": ["title", "plan_markdown", "estimated_turns"]
            }
        }
    })
}

/// Build the full tool schema array for Layer 1 (Fast Reply).
pub fn layer1_tools() -> serde_json::Value {
    serde_json::json!([
        start_react_system_tool(),
        propose_plan_tool(),
        shell_tool_schema(),
        web_search_tool_schema(),
        file_read_tool_schema(),
        file_write_tool_schema(),
        codebase_search_tool_schema(),
        browser_tool_schema(),
        memory_tool_schema(),
        scratchpad_tool_schema(),
        timeline_tool_schema(),
        lessons_tool_schema(),
        create_artifact_tool_schema(),
        generate_image_tool_schema(),
        steering_tool_schema(),
        interpretability_tool_schema(),
        learning_tool_schema(),
        system_logs_tool_schema(),
    ])
}

/// Build the full tool schema array for Layer 2 (ReAct Loop).
pub fn layer2_tools() -> serde_json::Value {
    serde_json::json!([
        reply_request_tool(),
        refuse_request_tool(),
        extend_turns_tool(),
        shell_tool_schema(),
        web_search_tool_schema(),
        memory_tool_schema(),
        scratchpad_tool_schema(),
        synaptic_tool_schema(),
        timeline_tool_schema(),
        lessons_tool_schema(),
        self_skills_tool_schema(),
        learning_tool_schema(),
        steering_tool_schema(),
        interpretability_tool_schema(),
        codebase_search_tool_schema(),
        file_read_tool_schema(),
        file_write_tool_schema(),
        browser_tool_schema(),
        create_artifact_tool_schema(),
        generate_image_tool_schema(),
        spawn_sub_agent_tool_schema(),
        codebase_edit_tool_schema(),
        system_recompile_tool_schema(),
        checkpoint_tool_schema(),
        system_logs_tool_schema(),
    ])
}

// Tool schema definitions extracted to schema_definitions.rs for governance compliance.
use crate::tools::schema_definitions::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_terminator() {
        assert!(is_loop_terminator("reply_request"));
        assert!(is_loop_terminator("refuse_request"));
        assert!(!is_loop_terminator("shell"));
    }

    #[test]
    fn test_extract_reply() {
        let tc = ToolCall {
            id: "1".into(), name: "reply_request".into(),
            arguments: r#"{"message":"Hello!"}"#.into(),
        };
        assert_eq!(extract_reply_text(&tc), Some("Hello!".to_string()));
    }

    #[test]
    fn test_layer1_tools() {
        let tools = layer1_tools();
        let arr = tools.as_array().unwrap();
        assert_eq!(arr.len(), 18, "L1 should have 18 tools");
        assert!(arr.iter().any(|t| t["function"]["name"] == "start_react_system"));
        assert!(arr.iter().any(|t| t["function"]["name"] == "memory"));
        assert!(arr.iter().any(|t| t["function"]["name"] == "scratchpad"));
        assert!(arr.iter().any(|t| t["function"]["name"] == "generate_image"));
        assert!(arr.iter().any(|t| t["function"]["name"] == "system_logs"));
        // Self-coding tools must NOT be in L1 — require ReAct planning
        assert!(!arr.iter().any(|t| t["function"]["name"] == "codebase_edit"));
        assert!(!arr.iter().any(|t| t["function"]["name"] == "checkpoint"));
        assert!(!arr.iter().any(|t| t["function"]["name"] == "system_recompile"));
    }

    #[test]
    fn test_layer2_tools() {
        let tools = layer2_tools();
        let arr = tools.as_array().unwrap();
        assert!(arr.iter().any(|t| t["function"]["name"] == "reply_request"));
        assert!(arr.iter().any(|t| t["function"]["name"] == "memory"));
    }

    #[test]
    fn test_create_artifact_in_l2() {
        let tools = layer2_tools();
        let arr = tools.as_array().unwrap();
        assert!(arr.iter().any(|t| t["function"]["name"] == "create_artifact"));
    }

    #[test]
    fn test_self_coding_tools_in_l2() {
        let tools = layer2_tools();
        let arr = tools.as_array().unwrap();
        assert!(arr.iter().any(|t| t["function"]["name"] == "codebase_edit"));
        assert!(arr.iter().any(|t| t["function"]["name"] == "system_recompile"));
        assert!(arr.iter().any(|t| t["function"]["name"] == "checkpoint"));
    }
}
