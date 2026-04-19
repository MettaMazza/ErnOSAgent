// Ern-OS — Self-Skills tool — manage reusable procedural skills

use crate::memory::procedures::{ProcedureStep, ProcedureStore};
use anyhow::Result;

pub async fn execute(args: &serde_json::Value, procedures: &mut ProcedureStore) -> Result<String> {
    tracing::info!(tool = "self_skills", "tool START");
    let action = args["action"].as_str().unwrap_or("");
    match action {
        "list" => list_skills(procedures),
        "view" => view_skill(args, procedures),
        "create" => create_skill(args, procedures),
        "refine" => refine_skill(args, procedures),
        "delete" => delete_skill(args, procedures),
        other => Ok(format!("Unknown self_skills action: {}", other)),
    }
}

fn list_skills(procedures: &ProcedureStore) -> Result<String> {
    let all = procedures.all();
    if all.is_empty() { return Ok("No skills learned yet.".to_string()); }
    let lines: Vec<String> = all.iter().map(|p| {
        format!("[{}] **{}** — {} ({} steps, used {} times)",
            &p.id[..8], p.name, p.description, p.steps.len(), p.success_count)
    }).collect();
    Ok(lines.join("\n"))
}

fn view_skill(args: &serde_json::Value, procedures: &ProcedureStore) -> Result<String> {
    let name = args["name"].as_str().unwrap_or("");
    match procedures.find_by_name(name) {
        Some(p) => {
            let steps: Vec<String> = p.steps.iter().enumerate().map(|(i, s)| {
                format!("  {}. [{}] {}", i + 1, s.tool, s.instruction)
            }).collect();
            Ok(format!("**{}**\n{}\n\nSteps:\n{}", p.name, p.description, steps.join("\n")))
        }
        None => Ok(format!("Skill '{}' not found", name)),
    }
}

fn create_skill(args: &serde_json::Value, procedures: &mut ProcedureStore) -> Result<String> {
    let name = args["name"].as_str().unwrap_or("unnamed");
    let desc = args["description"].as_str().unwrap_or("");
    let steps = parse_steps(args);
    match procedures.add_if_new(name, desc, steps)? {
        true => Ok(format!("Skill '{}' created successfully", name)),
        false => Ok(format!("Skill '{}' already exists — use refine to update", name)),
    }
}

/// Resolve a skill identifier — tries `id` first, then falls back to `name` lookup.
fn resolve_id(args: &serde_json::Value, procedures: &ProcedureStore) -> Option<String> {
    if let Some(id) = args["id"].as_str().filter(|s| !s.is_empty()) {
        return Some(id.to_string());
    }
    if let Some(name) = args["name"].as_str().filter(|s| !s.is_empty()) {
        return procedures.find_by_name(name).map(|p| p.id.clone());
    }
    None
}

fn refine_skill(args: &serde_json::Value, procedures: &mut ProcedureStore) -> Result<String> {
    let id = match resolve_id(args, procedures) {
        Some(id) => id,
        None => return Ok("Error: 'id' or 'name' required for refine. Use 'list' to see skills.".to_string()),
    };
    let steps = parse_steps(args);
    procedures.refine(&id, steps)?;
    Ok(format!("Skill '{}' refined", id))
}

fn delete_skill(args: &serde_json::Value, procedures: &mut ProcedureStore) -> Result<String> {
    let id = match resolve_id(args, procedures) {
        Some(id) => id,
        None => return Ok("Error: 'id' or 'name' required for delete. Use 'list' to see skills.".to_string()),
    };
    procedures.remove(&id)?;
    Ok(format!("Skill '{}' deleted", id))
}

fn parse_steps(args: &serde_json::Value) -> Vec<ProcedureStep> {
    args["steps"].as_array().map(|arr| {
        arr.iter().map(|s| ProcedureStep {
            tool: s["tool"].as_str().unwrap_or("").to_string(),
            purpose: s["purpose"].as_str().unwrap_or("").to_string(),
            instruction: s["instruction"].as_str().unwrap_or("").to_string(),
        }).collect()
    }).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_list_empty() {
        let mut store = ProcedureStore::new();
        let args = serde_json::json!({"action": "list"});
        let result = execute(&args, &mut store).await.unwrap();
        assert!(result.contains("No skills"));
    }

    #[tokio::test]
    async fn test_create_and_list() {
        let mut store = ProcedureStore::new();
        let args = serde_json::json!({
            "action": "create",
            "name": "Deploy",
            "description": "Deploy to production",
            "steps": [{"tool": "shell", "instruction": "cargo build --release"}]
        });
        execute(&args, &mut store).await.unwrap();
        let list_args = serde_json::json!({"action": "list"});
        let result = execute(&list_args, &mut store).await.unwrap();
        assert!(result.contains("Deploy"));
    }
}
