//! DAG executor — runs ready tasks via sub-agents, tracks progress.

use anyhow::{Context, Result};
use crate::inference::sub_agent::{SubAgentConfig, SubAgentResult, run_sub_agent};
use crate::provider::Provider;
use crate::web::state::AppState;
use super::dag::{TaskDag, TaskStatus};
use super::planner;

/// Result of executing an entire DAG.
#[derive(Debug, Clone)]
pub struct DagExecutionResult {
    pub completed: usize,
    pub failed: usize,
    pub blocked: usize,
    pub total: usize,
    pub overall_success: bool,
    pub summary: String,
}

/// Execute a TaskDag by running ready tasks via sub-agents.
///
/// Loop: find ready tasks → spawn sub-agent for each → update DAG → repeat.
/// Bounded by DAG resolution (all tasks done/failed/blocked).
pub async fn execute_dag(
    provider: &dyn Provider,
    state: &AppState,
    dag: &mut TaskDag,
) -> Result<DagExecutionResult> {
    tracing::info!(
        objective = %dag.objective,
        tasks = dag.nodes.len(),
        "Starting DAG execution"
    );

    let mut iteration = 0usize;
    let max_iterations = dag.nodes.len() * 3; // safety: 3x task count

    while !dag.is_resolved() && iteration < max_iterations {
        iteration += 1;
        // Collect ready task info (owned) before mutating DAG
        let ready_info: Vec<(String, String, String, Vec<String>)> = dag.ready_tasks()
            .iter()
            .map(|t| (t.id.clone(), t.title.clone(), t.description.clone(), t.assigned_tools.clone()))
            .collect();

        if ready_info.is_empty() {
            tracing::warn!(iteration, "No ready tasks but DAG not resolved — possible deadlock");
            break;
        }

        for (task_id, task_title, task_desc, task_tools) in ready_info {
            tracing::info!(task = %task_title, id = %task_id, "Executing task");

            dag.start_task(&task_id);

            let config = SubAgentConfig {
                task: format!("Task: {}\n\nDescription: {}\n\nComplete this task thoroughly.", task_title, task_desc),
                allowed_tools: task_tools,
                max_turns: 15,
            };
            let result = run_sub_agent(provider, config, state).await;
            handle_task_result(provider, dag, &task_id, result).await;

            tracing::info!(progress = %dag.progress_summary(), "DAG progress");
        }
    }

    let result = build_execution_result(dag);
    tracing::info!(
        success = result.overall_success,
        completed = result.completed,
        failed = result.failed,
        "DAG execution complete"
    );

    Ok(result)
}


/// Handle the result of a sub-agent execution.
async fn handle_task_result(
    provider: &dyn Provider,
    dag: &mut TaskDag,
    task_id: &str,
    result: Result<SubAgentResult>,
) {
    match result {
        Ok(sub_result) if sub_result.success => {
            dag.complete_task(task_id, &sub_result.summary);
        }
        Ok(sub_result) => {
            tracing::warn!(task = %task_id, "Task failed — attempting replan");
            let replan_result = attempt_replan(provider, dag, task_id, &sub_result.summary).await;
            if !replan_result {
                dag.fail_task(task_id, &sub_result.summary);
            }
        }
        Err(e) => {
            dag.fail_task(task_id, &format!("Execution error: {}", e));
        }
    }
}

/// Attempt to replan a failed task. Returns true if replan succeeded.
async fn attempt_replan(
    provider: &dyn Provider,
    dag: &mut TaskDag,
    task_id: &str,
    error_context: &str,
) -> bool {
    match planner::replan_task(provider, dag, task_id, error_context).await {
        Ok(revised_task) => {
            if let Some(node) = dag.nodes.iter_mut().find(|n| n.id == task_id) {
                node.description = revised_task.description;
                node.assigned_tools = revised_task.assigned_tools;
                node.status = TaskStatus::Ready;
                node.error = None;
                node.started_at = None;
                tracing::info!(task = %task_id, "Task replanned successfully");
                true
            } else {
                false
            }
        }
        Err(e) => {
            tracing::warn!(task = %task_id, error = %e, "Replan failed");
            false
        }
    }
}

/// Build the final execution result from DAG state.
fn build_execution_result(dag: &TaskDag) -> DagExecutionResult {
    let total = dag.nodes.len();
    let completed = dag.nodes.iter().filter(|n| n.status == TaskStatus::Completed).count();
    let failed = dag.nodes.iter().filter(|n| n.status == TaskStatus::Failed).count();
    let blocked = dag.nodes.iter().filter(|n| n.status == TaskStatus::Blocked).count();

    DagExecutionResult {
        completed,
        failed,
        blocked,
        total,
        overall_success: failed == 0 && blocked == 0,
        summary: dag.progress_summary(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::planning::dag;

    #[test]
    fn test_build_execution_result_all_complete() {
        let mut task_dag = TaskDag::new("test", vec![
            dag::task_node("a", "A", "do A", vec![], vec![]),
        ]);
        task_dag.complete_task("a", "done");
        let result = build_execution_result(&task_dag);
        assert!(result.overall_success);
        assert_eq!(result.completed, 1);
        assert_eq!(result.failed, 0);
    }

    #[test]
    fn test_build_execution_result_with_failure() {
        let mut task_dag = TaskDag::new("test", vec![
            dag::task_node("a", "A", "do A", vec![], vec![]),
            dag::task_node("b", "B", "do B", vec![], vec!["a".into()]),
        ]);
        task_dag.fail_task("a", "broken");
        let result = build_execution_result(&task_dag);
        assert!(!result.overall_success);
        assert_eq!(result.failed, 1);
        assert_eq!(result.blocked, 1);
    }

    #[test]
    fn test_dag_execution_result_fields() {
        let result = DagExecutionResult {
            completed: 3, failed: 1, blocked: 0, total: 4,
            overall_success: false, summary: "test".into(),
        };
        assert_eq!(result.total, 4);
        assert!(!result.overall_success);
    }
}
