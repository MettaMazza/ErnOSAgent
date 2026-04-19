//! Task DAG — directed acyclic graph of sub-tasks with dependency tracking.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Status of a single task node.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    Pending,
    Ready,
    Running,
    Completed,
    Failed,
    Blocked,
}

/// A single task in the DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskNode {
    pub id: String,
    pub title: String,
    pub description: String,
    pub status: TaskStatus,
    pub assigned_tools: Vec<String>,
    pub depends_on: Vec<String>,
    pub result: Option<String>,
    pub error: Option<String>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// A directed acyclic graph of tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskDag {
    pub id: String,
    pub objective: String,
    pub nodes: Vec<TaskNode>,
    pub created_at: DateTime<Utc>,
}

impl TaskDag {
    /// Create a new DAG from an objective and a set of task nodes.
    pub fn new(objective: &str, nodes: Vec<TaskNode>) -> Self {
        let mut dag = Self {
            id: uuid::Uuid::new_v4().to_string(),
            objective: objective.to_string(),
            nodes,
            created_at: Utc::now(),
        };
        dag.update_ready_status();
        dag
    }

    /// Get all tasks whose dependencies are fully satisfied.
    pub fn ready_tasks(&self) -> Vec<&TaskNode> {
        self.nodes.iter()
            .filter(|n| n.status == TaskStatus::Ready)
            .collect()
    }

    /// Mark a task as running.
    pub fn start_task(&mut self, id: &str) {
        if let Some(node) = self.nodes.iter_mut().find(|n| n.id == id) {
            node.status = TaskStatus::Running;
            node.started_at = Some(Utc::now());
        }
    }

    /// Mark a task as completed with its result, then update deps.
    pub fn complete_task(&mut self, id: &str, result: &str) {
        if let Some(node) = self.nodes.iter_mut().find(|n| n.id == id) {
            node.status = TaskStatus::Completed;
            node.result = Some(result.to_string());
            node.completed_at = Some(Utc::now());
        }
        self.update_ready_status();
    }

    /// Mark a task as failed, then block dependents.
    pub fn fail_task(&mut self, id: &str, reason: &str) {
        if let Some(node) = self.nodes.iter_mut().find(|n| n.id == id) {
            node.status = TaskStatus::Failed;
            node.error = Some(reason.to_string());
            node.completed_at = Some(Utc::now());
        }
        self.cascade_block(id);
    }

    /// Check if the entire DAG is resolved (no pending/ready/running).
    pub fn is_resolved(&self) -> bool {
        self.nodes.iter().all(|n| matches!(
            n.status,
            TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Blocked
        ))
    }

    /// Progress summary string.
    pub fn progress_summary(&self) -> String {
        let total = self.nodes.len();
        let completed = self.count_status(TaskStatus::Completed);
        let failed = self.count_status(TaskStatus::Failed);
        let blocked = self.count_status(TaskStatus::Blocked);
        let running = self.count_status(TaskStatus::Running);
        let ready = self.count_status(TaskStatus::Ready);
        let pending = self.count_status(TaskStatus::Pending);

        format!(
            "[DAG: {}] {}/{} done, {} running, {} ready, {} pending, {} failed, {} blocked",
            self.objective, completed, total, running, ready, pending, failed, blocked
        )
    }

    /// Count nodes with a given status.
    fn count_status(&self, status: TaskStatus) -> usize {
        self.nodes.iter().filter(|n| n.status == status).count()
    }

    /// Update pending tasks to ready if all deps are completed.
    fn update_ready_status(&mut self) {
        let completed_ids: Vec<String> = self.nodes.iter()
            .filter(|n| n.status == TaskStatus::Completed)
            .map(|n| n.id.clone())
            .collect();

        for node in &mut self.nodes {
            if node.status == TaskStatus::Pending {
                let deps_met = node.depends_on.iter()
                    .all(|dep| completed_ids.contains(dep));
                if deps_met {
                    node.status = TaskStatus::Ready;
                }
            }
        }
    }

    /// Block all tasks that depend on a failed task (recursive).
    fn cascade_block(&mut self, failed_id: &str) {
        let dependents: Vec<String> = self.nodes.iter()
            .filter(|n| n.depends_on.contains(&failed_id.to_string()))
            .filter(|n| matches!(n.status, TaskStatus::Pending | TaskStatus::Ready))
            .map(|n| n.id.clone())
            .collect();

        for dep_id in &dependents {
            if let Some(node) = self.nodes.iter_mut().find(|n| n.id == *dep_id) {
                node.status = TaskStatus::Blocked;
                node.error = Some(format!("Blocked: dependency '{}' failed", failed_id));
            }
            self.cascade_block(dep_id);
        }
    }
}

/// Create a TaskNode with the given fields.
pub fn task_node(
    id: &str, title: &str, description: &str,
    tools: Vec<String>, depends_on: Vec<String>,
) -> TaskNode {
    TaskNode {
        id: id.to_string(),
        title: title.to_string(),
        description: description.to_string(),
        status: TaskStatus::Pending,
        assigned_tools: tools,
        depends_on,
        result: None,
        error: None,
        started_at: None,
        completed_at: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_dag() -> TaskDag {
        TaskDag::new("Build a blog", vec![
            task_node("db", "DB Schema", "Create tables", vec!["codebase_edit".into()], vec![]),
            task_node("api", "API Routes", "REST endpoints", vec!["codebase_edit".into()], vec!["db".into()]),
            task_node("ui", "Frontend", "HTML/CSS/JS", vec!["codebase_edit".into()], vec!["api".into()]),
        ])
    }

    #[test]
    fn test_initial_ready_no_deps() {
        let dag = sample_dag();
        let ready = dag.ready_tasks();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, "db");
    }

    #[test]
    fn test_complete_unlocks_dependents() {
        let mut dag = sample_dag();
        dag.complete_task("db", "Tables created");
        let ready = dag.ready_tasks();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].id, "api");
    }

    #[test]
    fn test_fail_blocks_dependents() {
        let mut dag = sample_dag();
        dag.fail_task("db", "syntax error");
        assert_eq!(dag.nodes[1].status, TaskStatus::Blocked); // api
        assert_eq!(dag.nodes[2].status, TaskStatus::Blocked); // ui
    }

    #[test]
    fn test_is_resolved_all_complete() {
        let mut dag = sample_dag();
        dag.complete_task("db", "done");
        dag.complete_task("api", "done");
        dag.complete_task("ui", "done");
        assert!(dag.is_resolved());
    }

    #[test]
    fn test_is_resolved_with_failures() {
        let mut dag = sample_dag();
        dag.fail_task("db", "err");
        assert!(dag.is_resolved()); // db=failed, api+ui=blocked
    }

    #[test]
    fn test_progress_summary() {
        let dag = sample_dag();
        let summary = dag.progress_summary();
        assert!(summary.contains("0/3 done"));
        assert!(summary.contains("1 ready"));
        assert!(summary.contains("2 pending"));
    }

    #[test]
    fn test_start_task() {
        let mut dag = sample_dag();
        dag.start_task("db");
        assert_eq!(dag.nodes[0].status, TaskStatus::Running);
        assert!(dag.nodes[0].started_at.is_some());
    }

    #[test]
    fn test_task_node_builder() {
        let node = task_node("t1", "Test", "Desc", vec!["tool".into()], vec!["dep".into()]);
        assert_eq!(node.id, "t1");
        assert_eq!(node.status, TaskStatus::Pending);
        assert_eq!(node.depends_on, vec!["dep"]);
    }
}
