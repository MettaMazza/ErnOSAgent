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
        assert!(params["required"].as_array().unwrap().contains(&serde_json::json!("message")));
    }
}
