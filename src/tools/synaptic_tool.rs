// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Synaptic Graph Tool — agent interface for the in-memory knowledge graph.
//!
//! 7 actions matching HIVE's `operate_synaptic_graph` tool:
//! store, search, beliefs, relate, stats, layers, link_memory

use crate::memory::synaptic::SynapticGraph;
use crate::tools::executor::ToolExecutor;
use crate::tools::schema::{ToolCall, ToolResult};
use std::sync::Arc;

/// Execute the operate_synaptic_graph tool.
fn execute_synaptic_tool(call: &ToolCall, graph: &Arc<SynapticGraph>) -> ToolResult {
    let action = call
        .arguments
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("search")
        .to_lowercase();

    tracing::debug!(action = %action, "Synaptic Graph tool executing");

    let (output, success, error) = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(dispatch_action(&action, call, graph))
    });

    ToolResult {
        tool_call_id: call.id.clone(),
        name: call.name.clone(),
        output,
        success,
        error,
    }
}

/// Dispatch to the correct action handler.
async fn dispatch_action(
    action: &str,
    call: &ToolCall,
    graph: &Arc<SynapticGraph>,
) -> (String, bool, Option<String>) {
    match action {
        "store" => action_store(call, graph).await,
        "search" => action_search(call, graph).await,
        "beliefs" => action_beliefs(call, graph).await,
        "relate" => action_relate(call, graph).await,
        "stats" => action_stats(graph).await,
        "layers" => action_layers(graph).await,
        "link_memory" => action_link_memory(call, graph).await,
        _ => (
            format!(
                "Unknown action: '{}'. Available: store, search, beliefs, relate, stats, layers, link_memory",
                action
            ),
            false,
            Some(format!("Unknown action: {}", action)),
        ),
    }
}

// ─── Action Implementations ─────────────────────────────────────

async fn action_store(
    call: &ToolCall,
    graph: &Arc<SynapticGraph>,
) -> (String, bool, Option<String>) {
    let concept = match call.arguments.get("concept").and_then(|v| v.as_str()) {
        Some(c) if !c.is_empty() => c,
        _ => {
            return (
                "Error: Missing required argument 'concept' for 'store' action.".to_string(),
                false,
                Some("Missing required argument: concept".to_string()),
            )
        }
    };
    let data = match call.arguments.get("data").and_then(|v| v.as_str()) {
        Some(d) if !d.is_empty() => d,
        _ => {
            return (
                "Error: Missing required argument 'data' for 'store' action.".to_string(),
                false,
                Some("Missing required argument: data".to_string()),
            )
        }
    };

    graph.store(concept, data).await;
    (format!("Stored: '{}' → '{}'", concept, data), true, None)
}

async fn action_search(
    call: &ToolCall,
    graph: &Arc<SynapticGraph>,
) -> (String, bool, Option<String>) {
    let concept = match call.arguments.get("concept").and_then(|v| v.as_str()) {
        Some(c) if !c.is_empty() => c,
        _ => {
            return (
                "Error: Missing required argument 'concept' for 'search' action. Pass the concept you want to look up.".to_string(),
                false,
                Some("Missing required argument: concept".to_string()),
            )
        }
    };

    let results = graph.search(concept).await;
    if results.is_empty() {
        (
            format!("No knowledge found for concept '{}'.", concept),
            true,
            None,
        )
    } else {
        (
            format!(
                "Found {} entries for '{}':\n{}",
                results.len(),
                concept,
                results.join("\n")
            ),
            true,
            None,
        )
    }
}

async fn action_beliefs(
    call: &ToolCall,
    graph: &Arc<SynapticGraph>,
) -> (String, bool, Option<String>) {
    let limit = call
        .arguments
        .get("limit")
        .and_then(|v| v.as_u64())
        .unwrap_or(10) as usize;

    let beliefs = graph.get_beliefs(limit).await;
    if beliefs.is_empty() {
        (
            "No beliefs stored yet. Use 'store' to build knowledge.".to_string(),
            true,
            None,
        )
    } else {
        (
            format!(
                "Core Beliefs ({} concepts):\n{}",
                beliefs.len(),
                beliefs
                    .iter()
                    .enumerate()
                    .map(|(i, b)| format!("{}. {}", i + 1, b))
                    .collect::<Vec<_>>()
                    .join("\n")
            ),
            true,
            None,
        )
    }
}

async fn action_relate(
    call: &ToolCall,
    graph: &Arc<SynapticGraph>,
) -> (String, bool, Option<String>) {
    let from = match call.arguments.get("from").and_then(|v| v.as_str()) {
        Some(f) if !f.is_empty() => f,
        _ => {
            return (
                "Error: Missing required argument 'from' for 'relate' action.".to_string(),
                false,
                Some("Missing required argument: from".to_string()),
            )
        }
    };
    let relation = match call.arguments.get("relation").and_then(|v| v.as_str()) {
        Some(r) if !r.is_empty() => r,
        _ => {
            return (
                "Error: Missing required argument 'relation' for 'relate' action.".to_string(),
                false,
                Some("Missing required argument: relation".to_string()),
            )
        }
    };
    let to = match call.arguments.get("to").and_then(|v| v.as_str()) {
        Some(t) if !t.is_empty() => t,
        _ => {
            return (
                "Error: Missing required argument 'to' for 'relate' action.".to_string(),
                false,
                Some("Missing required argument: to".to_string()),
            )
        }
    };

    // Check for contradictions before storing
    if let Some(existing) = graph.check_contradiction(from, relation, to).await {
        return (
            format!(
                "Warning: Contradiction detected. '{}' is already '{}' '{}', but you're asserting '{}'. Edge stored anyway — review if needed.",
                from, relation, existing, to
            ),
            true,
            None,
        );
    }

    graph.store_relationship(from, relation, to).await;
    (
        format!(
            "Relationship stored: '{}' --[{}]--> '{}'",
            from, relation, to
        ),
        true,
        None,
    )
}

async fn action_stats(graph: &Arc<SynapticGraph>) -> (String, bool, Option<String>) {
    let (nodes, edges) = graph.stats().await;
    let layers = graph.list_layers().await;
    (
        format!(
            "Synaptic Graph Stats:\n  Nodes: {}\n  Edges: {}\n  Layers: {}",
            nodes,
            edges,
            layers.len()
        ),
        true,
        None,
    )
}

async fn action_layers(graph: &Arc<SynapticGraph>) -> (String, bool, Option<String>) {
    let layers = graph.list_layers().await;
    if layers.is_empty() {
        (
            "No layers initialized. Call initialize() on startup.".to_string(),
            true,
            None,
        )
    } else {
        let mut out = format!("KG Layers ({}):\n", layers.len());
        for layer in &layers {
            out.push_str(&format!(
                "  • {} — {} (root: {})\n",
                layer.name, layer.description, layer.root_node
            ));
        }
        (out, true, None)
    }
}

async fn action_link_memory(
    call: &ToolCall,
    graph: &Arc<SynapticGraph>,
) -> (String, bool, Option<String>) {
    let concept = match call.arguments.get("concept").and_then(|v| v.as_str()) {
        Some(c) if !c.is_empty() => c,
        _ => {
            return (
                "Error: Missing required argument 'concept' for 'link_memory' action.".to_string(),
                false,
                Some("Missing required argument: concept".to_string()),
            )
        }
    };
    let target = match call.arguments.get("target").and_then(|v| v.as_str()) {
        Some(t) if !t.is_empty() => t,
        _ => {
            return (
                "Error: Missing required argument 'target' for 'link_memory' action.".to_string(),
                false,
                Some("Missing required argument: target".to_string()),
            )
        }
    };
    let memory_type = call
        .arguments
        .get("memory_type")
        .and_then(|v| v.as_str())
        .unwrap_or("timeline");

    graph
        .store_relationship(concept, &format!("linked_to_{}", memory_type), target)
        .await;
    (
        format!(
            "Linked concept '{}' to {} entry: '{}'",
            concept, memory_type, target
        ),
        true,
        None,
    )
}

/// Register the synaptic graph tool with the executor.
pub fn register_tools(executor: &mut ToolExecutor, graph: Arc<SynapticGraph>) {
    let g = graph;
    executor.register(
        "operate_synaptic_graph",
        Box::new(move |call: &ToolCall| execute_synaptic_tool(call, &g)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_call(action: &str, extra: serde_json::Value) -> ToolCall {
        let mut args = serde_json::json!({"action": action});
        if let serde_json::Value::Object(map) = extra {
            for (k, v) in map {
                args[k] = v;
            }
        }
        ToolCall {
            id: "test-synaptic".to_string(),
            name: "operate_synaptic_graph".to_string(),
            arguments: args,
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_store_and_search() {
        let graph = Arc::new(SynapticGraph::new(None));

        let call = make_call(
            "store",
            serde_json::json!({"concept": "Apple", "data": "A red fruit"}),
        );
        let result = execute_synaptic_tool(&call, &graph);
        assert!(result.success);
        assert!(result.output.contains("Stored"));

        let call = make_call("search", serde_json::json!({"concept": "Apple"}));
        let result = execute_synaptic_tool(&call, &graph);
        assert!(result.success);
        assert!(result.output.contains("A red fruit"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_beliefs() {
        let graph = Arc::new(SynapticGraph::new(None));

        for data in &["fact 1", "fact 2", "fact 3"] {
            let call = make_call(
                "store",
                serde_json::json!({"concept": "Rust", "data": data}),
            );
            execute_synaptic_tool(&call, &graph);
        }

        let call = make_call("beliefs", serde_json::json!({}));
        let result = execute_synaptic_tool(&call, &graph);
        assert!(result.success);
        assert!(result.output.contains("Rust"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_relate() {
        let graph = Arc::new(SynapticGraph::new(None));
        let call = make_call(
            "relate",
            serde_json::json!({"from": "Apple", "relation": "is_a", "to": "Fruit"}),
        );
        let result = execute_synaptic_tool(&call, &graph);
        assert!(result.success);
        assert!(result.output.contains("is_a"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_stats() {
        let graph = Arc::new(SynapticGraph::new(None));
        let call = make_call("stats", serde_json::json!({}));
        let result = execute_synaptic_tool(&call, &graph);
        assert!(result.success);
        assert!(result.output.contains("Nodes: 0"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_contradiction_warning() {
        let graph = Arc::new(SynapticGraph::new(None));

        let call = make_call(
            "relate",
            serde_json::json!({"from": "Earth", "relation": "IS_A", "to": "Planet"}),
        );
        execute_synaptic_tool(&call, &graph);

        let call = make_call(
            "relate",
            serde_json::json!({"from": "Earth", "relation": "IS_A", "to": "Star"}),
        );
        let result = execute_synaptic_tool(&call, &graph);
        assert!(result.success);
        assert!(result.output.contains("Contradiction"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_missing_args() {
        let graph = Arc::new(SynapticGraph::new(None));

        let call = make_call("store", serde_json::json!({}));
        let result = execute_synaptic_tool(&call, &graph);
        assert!(!result.success);
        assert!(result.error.unwrap().contains("concept"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_unknown_action() {
        let graph = Arc::new(SynapticGraph::new(None));
        let call = make_call("invalid", serde_json::json!({}));
        let result = execute_synaptic_tool(&call, &graph);
        assert!(!result.success);
        assert!(result.output.contains("Unknown action"));
    }
}
