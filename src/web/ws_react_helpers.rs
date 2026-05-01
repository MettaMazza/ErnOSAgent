//! ReAct loop helper utilities — tool context building
//! and shared helpers used by the main ReAct loop handler in `ws_react.rs`.

use crate::provider::Message;
use crate::web::state::AppState;

/// Build a concise tool context string from the message history for the observer.
pub fn build_tool_context(messages: &[Message]) -> String {
    let mut entries: Vec<String> = Vec::new();
    for (i, msg) in messages.iter().enumerate() {
        if msg.role == "tool" {
            let tool_call_id = msg.tool_call_id.as_deref().unwrap_or("");
            let result_text = msg.text_content();
            let result_preview = truncate_preview(&result_text, 200);
            let tool_name = find_tool_name(messages, i, tool_call_id);
            entries.push(format!("[{}] {} → {}", entries.len() + 1, tool_name, result_preview));
        }
    }

    if entries.is_empty() {
        String::new()
    } else {
        format!("Tools executed this session ({} calls):\n{}", entries.len(), entries.join("\n"))
    }
}

/// Truncate a string to a preview length at a character boundary.
fn truncate_preview(text: &str, max_chars: usize) -> String {
    if text.len() <= max_chars {
        text.to_string()
    } else {
        let boundary = text.char_indices()
            .take_while(|(i, _)| *i <= max_chars)
            .last()
            .map(|(i, _)| i)
            .unwrap_or(0);
        format!("{}...", &text[..boundary])
    }
}

/// Find the tool name for a given tool_call_id by scanning preceding messages.
fn find_tool_name(messages: &[Message], current_idx: usize, tool_call_id: &str) -> String {
    for j in (0..current_idx).rev() {
        if messages[j].role == "assistant" {
            if let Some(tcs) = &messages[j].tool_calls {
                for tc in tcs {
                    if tc["id"].as_str() == Some(tool_call_id) {
                        return tc["function"]["name"]
                            .as_str()
                            .unwrap_or("unknown")
                            .to_string();
                    }
                }
            }
        }
    }
    "unknown".to_string()
}

/// Background skill synthesis after ReAct loop completion.
/// Extracts reusable procedural skills from the tool execution history.
pub fn spawn_skill_synthesis(state: &AppState, user_query: &str) {
    let provider = state.provider.clone();
    let memory = state.memory.clone();
    let query = user_query.to_string();

    tokio::spawn(async move {
        // Collect recent tool usage from memory for synthesis
        let tool_history: Vec<(String, String)> = {
            let mem = memory.read().await;
            mem.procedures.recent_tool_usage(10)
        };

        if !crate::observer::skills::is_skill_worthy(tool_history.len()) {
            return;
        }

        match crate::observer::skills::synthesise_skill(provider.as_ref(), &query, &tool_history).await {
            Ok(Some(skill)) => {
                tracing::info!(skill = %skill.name, confidence = skill.confidence, "Skill synthesised");
                let mut mem = memory.write().await;
                let _ = mem.procedures.record_skill(&skill.name, &skill.description);
            }
            Ok(None) => tracing::debug!("No reusable skill extracted"),
            Err(e) => tracing::warn!(error = %e, "Skill synthesis failed"),
        }
    });
}
