// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Performance Review Tool — self-introspection aggregating training and failure data.
//!
//! Ported from HIVENET's `performance_review.rs`. Gathers data from teacher
//! buffers, lessons, and memory stats, then returns a structured analysis
//! the ReAct loop can act on.

use crate::learning::buffers::TrainingBuffers;
use crate::tools::executor::ToolExecutor;
use crate::tools::schema::{ToolCall, ToolResult};
use std::collections::HashMap;
use std::sync::Arc;

/// Scope of the performance review.
#[derive(Debug, Clone, Copy)]
enum ReviewScope {
    Full,
    Failures,
    Successes,
}

impl ReviewScope {
    fn from_str(s: &str) -> Self {
        match s {
            "failures" => Self::Failures,
            "successes" => Self::Successes,
            _ => Self::Full,
        }
    }
}

/// Gather failure pattern data from the preference buffer.
fn gather_failure_patterns(buffers: &TrainingBuffers) -> String {
    let pairs = buffers.preference.read_all().unwrap_or_default();
    if pairs.is_empty() {
        return "No failure patterns recorded yet.".to_string();
    }

    let mut categories: HashMap<String, usize> = HashMap::new();
    for pair in &pairs {
        *categories.entry(pair.failure_category.clone()).or_insert(0) += 1;
    }

    let mut sorted: Vec<_> = categories.into_iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(&a.1));

    let mut report = format!("FAILURE PATTERNS ({} total preference pairs)\n", pairs.len());
    for (category, count) in &sorted {
        let pct = (*count as f64 / pairs.len() as f64) * 100.0;
        report.push_str(&format!("  {category}: {count} ({pct:.0}%)\n"));
    }

    // Include most recent failures for context
    report.push_str("\nMost recent failures:\n");
    for pair in pairs.iter().rev().take(3) {
        let preview = if pair.rejected_response.len() > 80 {
            let boundary = pair.rejected_response.char_indices()
                .take_while(|(i, _)| *i <= 80)
                .last()
                .map(|(i, _)| i)
                .unwrap_or(0);
            format!("{}...", &pair.rejected_response[..boundary])
        } else {
            pair.rejected_response.clone()
        };
        report.push_str(&format!(
            "  [{}] {}\n",
            pair.failure_category, preview
        ));
    }

    report
}

/// Gather success pattern data from the golden buffer.
fn gather_success_patterns(buffers: &TrainingBuffers) -> String {
    let examples = buffers.golden.read_all().unwrap_or_default();
    if examples.is_empty() {
        return "No golden examples recorded yet.".to_string();
    }

    let mut report = format!("SUCCESS PATTERNS ({} golden examples)\n", examples.len());

    // Count by session to show productivity
    let mut sessions: HashMap<String, usize> = HashMap::new();
    for ex in &examples {
        *sessions.entry(ex.session_id.clone()).or_insert(0) += 1;
    }
    report.push_str(&format!("  Across {} sessions\n", sessions.len()));

    // Show most recent golden examples
    report.push_str("\nMost recent golden interactions:\n");
    for ex in examples.iter().rev().take(3) {
        let user_preview = if ex.user_message.len() > 60 {
            let boundary = ex.user_message.char_indices()
                .take_while(|(i, _)| *i <= 60)
                .last()
                .map(|(i, _)| i)
                .unwrap_or(0);
            format!("{}...", &ex.user_message[..boundary])
        } else {
            ex.user_message.clone()
        };
        report.push_str(&format!("  User: {user_preview}\n"));
    }

    report
}

/// Gather lesson data.
fn gather_lessons() -> String {
    let path = lesson_store_path();
    if !path.exists() {
        return "No lessons stored yet.".to_string();
    }

    match std::fs::read_to_string(&path) {
        Ok(content) => {
            let lessons: Vec<serde_json::Value> = serde_json::from_str(&content)
                .unwrap_or_default();
            if lessons.is_empty() {
                return "No lessons stored yet.".to_string();
            }

            let mut report = format!("LEARNED LESSONS ({} total)\n", lessons.len());
            for lesson in lessons.iter().take(5) {
                let rule = lesson.get("rule").and_then(|v| v.as_str()).unwrap_or("(no rule)");
                let confidence = lesson.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.0);
                report.push_str(&format!("  [{:.0}%] {rule}\n", confidence * 100.0));
            }
            report
        }
        Err(_) => "Failed to read lesson store.".to_string(),
    }
}

/// Path to the lesson store file.
fn lesson_store_path() -> std::path::PathBuf {
    let dir = std::env::var("ERNOSAGENT_DATA_DIR").unwrap_or_else(|_| {
        dirs::home_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join(".ernosagent")
            .to_string_lossy()
            .to_string()
    });
    std::path::PathBuf::from(dir).join("lessons.json")
}

/// Format the complete performance report.
fn format_performance_report(buffers: &TrainingBuffers, scope: ReviewScope) -> String {
    let mut report = String::from("═══ PERFORMANCE REVIEW ═══\n\n");

    match scope {
        ReviewScope::Full => {
            report.push_str(&gather_failure_patterns(buffers));
            report.push('\n');
            report.push_str(&gather_success_patterns(buffers));
            report.push('\n');
            report.push_str(&gather_lessons());
            report.push('\n');
            report.push_str(&buffer_stats(buffers));
        }
        ReviewScope::Failures => {
            report.push_str(&gather_failure_patterns(buffers));
        }
        ReviewScope::Successes => {
            report.push_str(&gather_success_patterns(buffers));
        }
    }

    report
}

/// Buffer statistics for the training system.
fn buffer_stats(buffers: &TrainingBuffers) -> String {
    format!(
        "TRAINING DATA STATUS\n  Golden buffer: {} examples\n  Preference buffer: {} pairs\n",
        buffers.golden.count(),
        buffers.preference.count(),
    )
}

/// The tool handler.
fn performance_review_handler(
    buffers: Arc<TrainingBuffers>,
) -> impl Fn(&ToolCall) -> ToolResult + Send + Sync {
    move |call: &ToolCall| {
        let scope_str = call.arguments.get("scope")
            .and_then(|v| v.as_str())
            .unwrap_or("full");

        let scope = ReviewScope::from_str(scope_str);
        let report = format_performance_report(&buffers, scope);

        ToolResult {
            tool_call_id: call.id.clone(),
            name: call.name.clone(),
            output: report,
            success: true,
            error: None,
        }
    }
}

/// Register the performance review tool.
pub fn register_tools(executor: &mut ToolExecutor, buffers: Arc<TrainingBuffers>) {
    executor.register(
        "performance_review",
        Box::new(performance_review_handler(buffers)),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_review_scope_from_str() {
        assert!(matches!(ReviewScope::from_str("full"), ReviewScope::Full));
        assert!(matches!(ReviewScope::from_str("failures"), ReviewScope::Failures));
        assert!(matches!(ReviewScope::from_str("successes"), ReviewScope::Successes));
        assert!(matches!(ReviewScope::from_str("unknown"), ReviewScope::Full));
    }

    #[test]
    fn test_lesson_store_path() {
        let path = lesson_store_path();
        assert!(path.to_string_lossy().contains("lessons.json"));
    }
}
