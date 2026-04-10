// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Tests for web route helper functions and serialization types.

#[cfg(test)]
mod tests {
    use crate::tools::schema::ToolCall;

    // Test the make_tool_call pattern used by all route handlers
    fn make_tool_call(name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall {
            id: format!("api_test_{}", name),
            name: name.to_string(),
            arguments: args,
        }
    }

    #[test]
    fn test_make_tool_call_memory() {
        let call = make_tool_call("memory_tool", serde_json::json!({
            "action": "recall",
            "query": "test",
            "limit": 10
        }));
        assert_eq!(call.name, "memory_tool");
        assert_eq!(call.arguments["action"], "recall");
        assert_eq!(call.arguments["limit"], 10);
    }

    #[test]
    fn test_make_tool_call_timeline_recent() {
        let call = make_tool_call("timeline_tool", serde_json::json!({
            "action": "recent",
            "limit": 50
        }));
        assert_eq!(call.name, "timeline_tool");
        assert_eq!(call.arguments["action"], "recent");
    }

    #[test]
    fn test_make_tool_call_timeline_search() {
        let call = make_tool_call("timeline_tool", serde_json::json!({
            "action": "search",
            "query": "test query"
        }));
        assert_eq!(call.arguments["query"], "test query");
    }

    #[test]
    fn test_make_tool_call_lessons() {
        let call = make_tool_call("lessons_tool", serde_json::json!({
            "action": "list",
            "limit": 100
        }));
        assert_eq!(call.name, "lessons_tool");
        assert_eq!(call.arguments["limit"], 100);
    }

    #[test]
    fn test_make_tool_call_reinforce() {
        let call = make_tool_call("lessons_tool", serde_json::json!({
            "action": "reinforce",
            "id": "lesson_123"
        }));
        assert_eq!(call.arguments["action"], "reinforce");
        assert_eq!(call.arguments["id"], "lesson_123");
    }

    #[test]
    fn test_make_tool_call_weaken() {
        let call = make_tool_call("lessons_tool", serde_json::json!({
            "action": "weaken",
            "id": "lesson_456"
        }));
        assert_eq!(call.arguments["action"], "weaken");
    }

    #[test]
    fn test_make_tool_call_scratchpad_read() {
        let call = make_tool_call("scratchpad_tool", serde_json::json!({
            "action": "read"
        }));
        assert_eq!(call.name, "scratchpad_tool");
        assert_eq!(call.arguments["action"], "read");
    }

    #[test]
    fn test_make_tool_call_scratchpad_write() {
        let call = make_tool_call("scratchpad_tool", serde_json::json!({
            "action": "write",
            "key": "notes",
            "content": "test content"
        }));
        assert_eq!(call.arguments["key"], "notes");
        assert_eq!(call.arguments["content"], "test content");
    }

    #[test]
    fn test_make_tool_call_reasoning() {
        let call = make_tool_call("reasoning_tool", serde_json::json!({
            "action": "review",
            "limit": 20
        }));
        assert_eq!(call.name, "reasoning_tool");
        assert_eq!(call.arguments["limit"], 20);
    }

    #[test]
    fn test_make_tool_call_reasoning_search() {
        let call = make_tool_call("reasoning_tool", serde_json::json!({
            "action": "search",
            "query": "logic"
        }));
        assert_eq!(call.arguments["action"], "search");
    }

    #[test]
    fn test_make_tool_call_reasoning_stats() {
        let call = make_tool_call("reasoning_tool", serde_json::json!({
            "action": "stats"
        }));
        assert_eq!(call.arguments["action"], "stats");
    }

    #[test]
    fn test_make_tool_call_checkpoint_list() {
        let call = make_tool_call("checkpoint_tool", serde_json::json!({
            "action": "list"
        }));
        assert_eq!(call.name, "checkpoint_tool");
    }

    #[test]
    fn test_make_tool_call_checkpoint_create() {
        let call = make_tool_call("checkpoint_tool", serde_json::json!({
            "action": "snapshot",
            "label": "before_deploy"
        }));
        assert_eq!(call.arguments["action"], "snapshot");
        assert_eq!(call.arguments["label"], "before_deploy");
    }

    #[test]
    fn test_make_tool_call_consolidate() {
        let call = make_tool_call("memory_tool", serde_json::json!({
            "action": "consolidate"
        }));
        assert_eq!(call.arguments["action"], "consolidate");
    }

    #[test]
    fn test_tool_call_id_format() {
        let call = make_tool_call("test", serde_json::json!({}));
        assert!(call.id.starts_with("api_test_"));
    }

    // Test serialization types used by routes
    #[test]
    fn test_tool_info_serialization() {
        #[derive(serde::Serialize)]
        struct ToolInfo { name: String, description: String }
        let info = ToolInfo {
            name: "web_search".to_string(),
            description: "Searches the web".to_string(),
        };
        let json = serde_json::to_string(&info).unwrap();
        assert!(json.contains("web_search"));
        assert!(json.contains("Searches the web"));
    }

    #[test]
    fn test_tool_history_entry_serialization() {
        #[derive(serde::Serialize)]
        struct ToolHistoryEntry {
            name: String,
            output_preview: String,
            success: bool,
        }
        let entry = ToolHistoryEntry {
            name: "download".to_string(),
            output_preview: "Downloaded 5MB".to_string(),
            success: true,
        };
        let json = serde_json::to_string(&entry).unwrap();
        assert!(json.contains("\"success\":true"));
    }

    #[test]
    fn test_search_result_serialization() {
        #[derive(serde::Serialize)]
        struct MemorySearchResult { query: String, results: String }
        let result = MemorySearchResult {
            query: "test".to_string(),
            results: "Found 3 matches".to_string(),
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("test"));
        assert!(json.contains("Found 3 matches"));
    }

    // Test ToolExecutor directly (the actual backend for routes)
    #[test]
    fn test_executor_unknown_tool() {
        let executor = crate::tools::executor::ToolExecutor::new();
        let call = make_tool_call("nonexistent_tool", serde_json::json!({}));
        let result = executor.execute(&call);
        assert!(!result.success);
        assert!(result.output.contains("not registered"));
    }

    #[test]
    fn test_executor_with_mock_handler() {
        let mut executor = crate::tools::executor::ToolExecutor::new();
        executor.register("mock_tool", Box::new(|call: &ToolCall| {
            crate::tools::schema::ToolResult {
                tool_call_id: call.id.clone(),
                name: call.name.clone(),
                output: format!("Executed with args: {}", call.arguments),
                success: true,
                error: None,
            }
        }));

        let call = make_tool_call("mock_tool", serde_json::json!({"key": "value"}));
        let result = executor.execute(&call);
        assert!(result.success);
        assert!(result.output.contains("key"));
    }

    #[test]
    fn test_executor_available_tools() {
        let mut executor = crate::tools::executor::ToolExecutor::new();
        executor.register("alpha", Box::new(|c: &ToolCall| crate::tools::schema::ToolResult {
            tool_call_id: c.id.clone(), name: c.name.clone(),
            output: String::new(), success: true, error: None,
        }));
        executor.register("beta", Box::new(|c: &ToolCall| crate::tools::schema::ToolResult {
            tool_call_id: c.id.clone(), name: c.name.clone(),
            output: String::new(), success: true, error: None,
        }));
        let tools = executor.available_tools();
        assert_eq!(tools, vec!["alpha", "beta"]);
    }
}
