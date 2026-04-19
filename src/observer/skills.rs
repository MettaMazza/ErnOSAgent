// Ern-OS — Post-task skill synthesis
//! After successful multi-step tasks, reflects on the workflow
//! and synthesises reusable procedural skills (Hermes-inspired).

use crate::provider::{Message, Provider};
use anyhow::{Context, Result};
use serde::Deserialize;

/// A skill candidate extracted from a completed task.
#[derive(Debug, Clone, Deserialize)]
pub struct SkillCandidate {
    pub name: String,
    pub description: String,
    pub steps: Vec<SkillStep>,
    pub triggers: Vec<String>,
    pub confidence: f32,
}

/// A single step within a synthesised skill.
#[derive(Debug, Clone, Deserialize)]
pub struct SkillStep {
    pub tool: String,
    pub instruction: String,
}

/// Check if a task used enough tools to be worth synthesising.
pub fn is_skill_worthy(tool_count: usize) -> bool {
    tool_count >= 3
}

/// Synthesise a skill from a completed multi-step task.
pub async fn synthesise_skill(
    provider: &dyn Provider,
    user_query: &str,
    tool_history: &[(String, String)],
) -> Result<Option<SkillCandidate>> {
    if !is_skill_worthy(tool_history.len()) { return Ok(None); }

    let prompt = build_synthesis_prompt(user_query, tool_history);
    let messages = vec![
        Message::text("system", SYNTHESIS_SYSTEM_PROMPT),
        Message::text("user", &prompt),
    ];

    let response = provider
        .chat_sync(&messages, None)
        .await
        .context("Skill synthesis inference failed")?;

    parse_skill_candidate(&response)
}

/// Build the synthesis prompt from query and tool usage history.
fn build_synthesis_prompt(user_query: &str, tool_history: &[(String, String)]) -> String {
    let tools_used: String = tool_history.iter()
        .enumerate()
        .map(|(i, (name, result))| {
            format!("{}. **{}** → {}", i + 1, name, result)
        })
        .collect::<Vec<_>>()
        .join("\n");

    format!(
        "=== USER REQUEST ===\n{}\n\n=== TOOLS USED ===\n{}\n\n\
         If this is a reusable workflow, respond as JSON.\n\
         If NOT reusable (one-off task), respond with: null",
        user_query, tools_used
    )
}

/// Parse a skill candidate from model response.
fn parse_skill_candidate(response: &str) -> Result<Option<SkillCandidate>> {
    let trimmed = response.trim();
    if trimmed == "null" || trimmed.is_empty() { return Ok(None); }

    // Find JSON object in response
    let json_str = extract_json_object(trimmed)
        .context("No JSON object found in skill synthesis response")?;

    let candidate: SkillCandidate = serde_json::from_str(&json_str)
        .context("Failed to parse skill candidate JSON")?;

    if candidate.confidence < 0.6 { return Ok(None); }
    Ok(Some(candidate))
}

/// Extract a JSON object from text that may include markdown fences.
fn extract_json_object(text: &str) -> Option<String> {
    if let Ok(_) = serde_json::from_str::<SkillCandidate>(text) {
        return Some(text.to_string());
    }
    let start = text.find('{')?;
    let end = text.rfind('}')? + 1;
    if start < end { Some(text[start..end].to_string()) } else { None }
}

const SYNTHESIS_SYSTEM_PROMPT: &str = "\
You are a skill synthesis engine. Analyse completed multi-step tasks and \
extract reusable procedural skills. A skill is a repeatable workflow that \
could be applied to similar future tasks. \
Respond ONLY with JSON: {\"name\": \"...\", \"description\": \"...\", \
\"steps\": [{\"tool\": \"...\", \"instruction\": \"...\"}], \
\"triggers\": [\"keyword1\", \"keyword2\"], \"confidence\": 0.0-1.0} \
Or respond with: null (if the task was one-off and not reusable).";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_skill_worthy() {
        assert!(!is_skill_worthy(0));
        assert!(!is_skill_worthy(2));
        assert!(is_skill_worthy(3));
        assert!(is_skill_worthy(10));
    }

    #[test]
    fn test_build_synthesis_prompt() {
        let history = vec![
            ("shell".to_string(), "ok".to_string()),
            ("web_search".to_string(), "results".to_string()),
        ];
        let prompt = build_synthesis_prompt("deploy app", &history);
        assert!(prompt.contains("deploy app"));
        assert!(prompt.contains("shell"));
        assert!(prompt.contains("web_search"));
    }

    #[test]
    fn test_parse_skill_candidate_valid() {
        let json = r#"{"name": "deploy", "description": "deploy app", "steps": [{"tool": "shell", "instruction": "build"}], "triggers": ["deploy"], "confidence": 0.9}"#;
        let result = parse_skill_candidate(json).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "deploy");
    }

    #[test]
    fn test_parse_skill_candidate_null() {
        let result = parse_skill_candidate("null").unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_skill_candidate_low_confidence() {
        let json = r#"{"name": "x", "description": "x", "steps": [], "triggers": [], "confidence": 0.3}"#;
        let result = parse_skill_candidate(json).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_extract_json_object_embedded() {
        let text = "Here is the skill:\n{\"name\": \"test\", \"description\": \"t\", \"steps\": [], \"triggers\": [], \"confidence\": 0.8}\nDone.";
        let result = extract_json_object(text);
        assert!(result.is_some());
    }
}
