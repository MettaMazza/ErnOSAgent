//! LLM-driven task planner — decomposes objectives into TaskDag structures.

use anyhow::{Context, Result};
use crate::provider::{Message, Provider};
use super::dag::{self, TaskDag, TaskNode};

/// Decompose a high-level objective into a TaskDag using the model.
pub async fn decompose_objective(
    provider: &dyn Provider,
    objective: &str,
    project_context: &str,
) -> Result<TaskDag> {
    let prompt = build_decomposition_prompt(objective, project_context);
    let messages = vec![
        Message::text("system", &prompt),
        Message::text("user", &format!("Decompose this objective into tasks: {}", objective)),
    ];

    let response = provider.chat_sync(&messages, None).await
        .context("Planner LLM call failed")?;

    let nodes = parse_task_nodes(&response)?;

    tracing::info!(
        objective = %objective,
        tasks = nodes.len(),
        "Objective decomposed into task DAG"
    );

    Ok(TaskDag::new(objective, nodes))
}

/// Re-plan a single failed task with error context.
pub async fn replan_task(
    provider: &dyn Provider,
    dag: &TaskDag,
    failed_task_id: &str,
    error_context: &str,
) -> Result<TaskNode> {
    let failed = dag.nodes.iter().find(|n| n.id == failed_task_id)
        .context("Failed task not found in DAG")?;

    let prompt = build_replan_prompt(failed, error_context, &dag.objective);
    let messages = vec![
        Message::text("system", &prompt),
        Message::text("user", &format!("Replan task '{}' after failure", failed.title)),
    ];

    let response = provider.chat_sync(&messages, None).await
        .context("Replan LLM call failed")?;

    let mut nodes = parse_task_nodes(&response)?;
    nodes.first().cloned().context("Replan produced no tasks")
}

/// Build the decomposition system prompt.
fn build_decomposition_prompt(objective: &str, context: &str) -> String {
    format!(
        "You are a task planner. Decompose the objective into concrete sub-tasks.\n\n\
         Output a JSON array of tasks. Each task has:\n\
         - \"id\": short kebab-case identifier\n\
         - \"title\": brief title\n\
         - \"description\": what to do (detailed instructions)\n\
         - \"tools\": array of tool names needed\n\
         - \"depends_on\": array of task IDs this depends on\n\n\
         Available tools: codebase_edit, codebase_create, codebase_search, \
         file_read, run_bash_command, web_search, reply_request\n\n\
         Project context:\n{}\n\n\
         Objective: {}\n\n\
         Output ONLY the JSON array, no markdown fences or explanation.",
        context, objective
    )
}

/// Build the replan prompt for a failed task.
fn build_replan_prompt(failed: &TaskNode, error: &str, objective: &str) -> String {
    format!(
        "A task failed during execution of objective: {}\n\n\
         Failed task: {} — {}\n\
         Error: {}\n\n\
         Create a revised version of this task that fixes the issue.\n\
         Output a JSON array with exactly 1 task object (same format as decomposition).\n\
         Output ONLY the JSON array.",
        objective, failed.title, failed.description, error
    )
}

/// Parse the LLM response into TaskNode objects.
fn parse_task_nodes(response: &str) -> Result<Vec<TaskNode>> {
    let cleaned = clean_json_response(response);

    let raw: Vec<serde_json::Value> = serde_json::from_str(&cleaned)
        .context("Failed to parse task JSON from planner")?;

    let nodes: Vec<TaskNode> = raw.iter()
        .map(|v| parse_single_node(v))
        .collect();

    if nodes.is_empty() {
        anyhow::bail!("Planner produced zero tasks");
    }

    Ok(nodes)
}

/// Parse a single JSON value into a TaskNode.
fn parse_single_node(v: &serde_json::Value) -> TaskNode {
    dag::task_node(
        v["id"].as_str().unwrap_or("task"),
        v["title"].as_str().unwrap_or("Untitled"),
        v["description"].as_str().unwrap_or(""),
        v["tools"].as_array()
            .map(|a| a.iter().filter_map(|t| t.as_str().map(String::from)).collect())
            .unwrap_or_default(),
        v["depends_on"].as_array()
            .map(|a| a.iter().filter_map(|d| d.as_str().map(String::from)).collect())
            .unwrap_or_default(),
    )
}

/// Strip markdown code fences from LLM JSON output.
fn clean_json_response(response: &str) -> String {
    let trimmed = response.trim();
    let without_fences = if trimmed.starts_with("```") {
        let start = trimmed.find('\n').unwrap_or(3);
        let end = trimmed.rfind("```").unwrap_or(trimmed.len());
        &trimmed[start..end]
    } else {
        trimmed
    };
    without_fences.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_task_nodes() {
        let json = r#"[
            {"id": "db", "title": "Database", "description": "Create schema", "tools": ["codebase_edit"], "depends_on": []},
            {"id": "api", "title": "API", "description": "Build routes", "tools": ["codebase_edit"], "depends_on": ["db"]}
        ]"#;
        let nodes = parse_task_nodes(json).unwrap();
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0].id, "db");
        assert_eq!(nodes[1].depends_on, vec!["db"]);
    }

    #[test]
    fn test_parse_with_code_fences() {
        let json = "```json\n[{\"id\":\"t1\",\"title\":\"Task 1\",\"description\":\"do it\",\"tools\":[],\"depends_on\":[]}]\n```";
        let nodes = parse_task_nodes(json).unwrap();
        assert_eq!(nodes.len(), 1);
    }

    #[test]
    fn test_parse_empty_fails() {
        let result = parse_task_nodes("[]");
        assert!(result.is_err());
    }

    #[test]
    fn test_clean_json_response() {
        assert_eq!(clean_json_response("```json\n[1,2]\n```"), "[1,2]");
        assert_eq!(clean_json_response("[1,2]"), "[1,2]");
    }

    #[test]
    fn test_build_decomposition_prompt() {
        let prompt = build_decomposition_prompt("Build blog", "Rust project");
        assert!(prompt.contains("Build blog"));
        assert!(prompt.contains("Rust project"));
        assert!(prompt.contains("codebase_edit"));
    }

    #[test]
    fn test_parse_single_node_defaults() {
        let v = serde_json::json!({"id": "x"});
        let node = parse_single_node(&v);
        assert_eq!(node.id, "x");
        assert_eq!(node.title, "Untitled");
    }
}
