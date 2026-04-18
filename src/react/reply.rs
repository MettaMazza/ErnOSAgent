// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! reply_request tool — the ONLY exit from the ReAct loop.

use crate::provider::ToolDefinition;
use crate::provider::ToolFunction;

/// Get the reply_request tool definition for inclusion in every inference call.
pub fn reply_request_tool() -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: ToolFunction {
            name: "reply_request".to_string(),
            description: "Deliver your final response to the user. This is the ONLY way \
                          to send a message to the user. You MUST call this tool when you \
                          are ready to respond. Raw text content is NOT delivered."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The complete response text to deliver to the user."
                    }
                },
                "required": ["message"]
            }),
        },
    }
}

/// Get the refuse_request tool definition — a loop terminator for refusals.
pub fn refuse_request_tool() -> ToolDefinition {
    ToolDefinition {
        tool_type: "function".to_string(),
        function: ToolFunction {
            name: "refuse_request".to_string(),
            description: "Refuse a request and deliver a refusal message to the user. \
                          Use this instead of reply_request when you are declining to engage — \
                          e.g. after the escalation ladder, when enforcing a boundary, or when \
                          a request violates your principles. The refusal is logged to a \
                          persistent audit trail. This terminates the turn."
                .to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The refusal message to deliver to the user."
                    },
                    "reason": {
                        "type": "string",
                        "description": "Internal reason for the refusal (logged, not shown to user)."
                    }
                },
                "required": ["message"]
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reply_request_tool_definition() {
        let tool = reply_request_tool();
        assert_eq!(tool.function.name, "reply_request");
        assert!(tool.function.description.contains("ONLY way"));
        assert!(tool.function.description.contains("MUST call"));
    }

    #[test]
    fn test_reply_request_parameters() {
        let tool = reply_request_tool();
        let params = &tool.function.parameters;
        assert!(params.get("properties").is_some());
        assert!(params["properties"]["message"]["type"].as_str() == Some("string"));
        assert!(params["required"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("message")));
    }

    #[test]
    fn test_refuse_request_tool_definition() {
        let tool = refuse_request_tool();
        assert_eq!(tool.function.name, "refuse_request");
        assert!(tool.function.description.contains("Refuse"));
        assert!(tool.function.description.contains("logged"));
    }

    #[test]
    fn test_refuse_request_parameters() {
        let tool = refuse_request_tool();
        let params = &tool.function.parameters;
        assert!(params["properties"]["message"]["type"].as_str() == Some("string"));
        assert!(params["properties"]["reason"]["type"].as_str() == Some("string"));
        assert!(params["required"]
            .as_array()
            .unwrap()
            .contains(&serde_json::json!("message")));
    }
}
