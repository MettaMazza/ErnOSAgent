//! Plan persistence — stores pending implementation plans for user approval.

use std::path::PathBuf;

#[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
pub struct PendingPlan {
    pub title: String,
    pub plan_markdown: String,
    pub estimated_turns: usize,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub revision: usize,
}

fn plan_dir() -> PathBuf {
    let dir = PathBuf::from("data/plans");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn plan_path(session_id: &str) -> PathBuf {
    plan_dir().join(format!("{}.json", session_id))
}

/// Save a pending plan for a session.
pub fn save_pending_plan(
    session_id: &str,
    title: &str,
    plan_markdown: &str,
    estimated_turns: usize,
) -> PendingPlan {
    // Load existing to increment revision
    let revision = load_pending_plan(session_id)
        .map(|p| p.revision + 1)
        .unwrap_or(1);

    let plan = PendingPlan {
        title: title.to_string(),
        plan_markdown: plan_markdown.to_string(),
        estimated_turns,
        created_at: chrono::Utc::now(),
        revision,
    };

    let path = plan_path(session_id);
    if let Ok(json) = serde_json::to_string_pretty(&plan) {
        let _ = std::fs::write(&path, json);
        tracing::info!(
            session = %session_id, title = %title,
            revision, turns = estimated_turns,
            "Pending plan saved"
        );
    }

    plan
}

/// Load a pending plan for a session.
pub fn load_pending_plan(session_id: &str) -> Option<PendingPlan> {
    let path = plan_path(session_id);
    std::fs::read_to_string(&path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
}

/// Delete a pending plan (after execution or cancellation).
pub fn delete_pending_plan(session_id: &str) {
    let path = plan_path(session_id);
    let _ = std::fs::remove_file(&path);
    tracing::debug!(session = %session_id, "Pending plan deleted");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plan_save_load_delete() {
        let sid = "test_plan_session_001";
        let plan = save_pending_plan(sid, "Test Plan", "## Steps\n- Do things", 5);
        assert_eq!(plan.title, "Test Plan");
        assert_eq!(plan.revision, 1);
        assert_eq!(plan.estimated_turns, 5);

        let loaded = load_pending_plan(sid).unwrap();
        assert_eq!(loaded.title, "Test Plan");
        assert_eq!(loaded.plan_markdown, "## Steps\n- Do things");

        // Revision increments
        let plan2 = save_pending_plan(sid, "Test Plan v2", "## Revised", 8);
        assert_eq!(plan2.revision, 2);

        delete_pending_plan(sid);
        assert!(load_pending_plan(sid).is_none());
    }
}
