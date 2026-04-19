// Ern-OS — Post-turn insight extraction
//! Silent background learning — extracts user preferences, corrections,
//! and contextual insights from completed exchanges. Stores them in the
//! Lessons tier for automatic recall in future conversations.

use crate::provider::{Message, Provider};
use anyhow::{Context, Result};
use serde::Deserialize;

/// A single extracted insight from a user↔assistant exchange.
#[derive(Debug, Clone, Deserialize)]
pub struct Insight {
    pub rule: String,
    pub category: String,
    pub confidence: f32,
}

/// Extract insights from a completed exchange.
/// Uses chat_sync (thinking disabled) for speed.
pub async fn extract_insights(
    provider: &dyn Provider,
    user_query: &str,
    assistant_reply: &str,
) -> Result<Vec<Insight>> {
    let prompt = build_extraction_prompt(user_query, assistant_reply);
    let messages = vec![
        Message::text("system", EXTRACTION_SYSTEM_PROMPT),
        Message::text("user", &prompt),
    ];

    let response = provider
        .chat_sync(&messages, None)
        .await
        .context("Insight extraction inference failed")?;

    parse_insights(&response)
}

/// Build the extraction prompt from the exchange.
fn build_extraction_prompt(user_query: &str, reply: &str) -> String {
    format!(
        "=== USER ===\n{}\n\n=== ASSISTANT ===\n{}\n\n\
         Extract insights as JSON array. Empty array [] if nothing notable.",
        user_query, reply
    )
}

/// Parse JSON insight array from model response.
pub fn parse_insights(response: &str) -> Result<Vec<Insight>> {
    // Find JSON array in response (model may wrap in markdown)
    let json_str = extract_json_array(response)
        .context("No JSON array found in insight extraction response")?;

    serde_json::from_str::<Vec<Insight>>(&json_str)
        .context("Failed to parse insight JSON array")
}

/// Extract a JSON array from text that may contain markdown fences.
fn extract_json_array(text: &str) -> Option<String> {
    // Try direct parse first
    if let Ok(_) = serde_json::from_str::<Vec<Insight>>(text.trim()) {
        return Some(text.trim().to_string());
    }

    // Find [...] in the text
    let start = text.find('[')?;
    let end = text.rfind(']')? + 1;
    if start < end {
        Some(text[start..end].to_string())
    } else {
        None
    }
}

/// Check if an exchange is worth extracting insights from.
/// Skips trivial inputs that carry no insight signal.
pub fn is_worth_extracting(user_query: &str) -> bool {
    let trimmed = user_query.trim().to_lowercase();
    let trivial = [
        "continue", "ok", "yes", "no", "go", "next", "thanks",
        "ty", "sure", "yep", "nope", "k", "y", "n",
    ];
    !trivial.contains(&trimmed.as_str()) && trimmed.len() > 10
}

const EXTRACTION_SYSTEM_PROMPT: &str = "\
You are a memory extraction system. Analyze exchanges and extract genuine insights \
about the user. Focus on: preferences (style, format, detail), corrections they made, \
technical context (stack, project), and behavioral patterns. \
Only extract real, actionable insights — not trivial observations. \
Respond ONLY with a JSON array: [{\"rule\": \"...\", \"category\": \"preference|correction|context|technical\", \"confidence\": 0.0-1.0}] \
Empty array [] if nothing notable.";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_extraction_prompt_contains_exchange() {
        let prompt = build_extraction_prompt("What is Rust?", "Rust is a language.");
        assert!(prompt.contains("USER"));
        assert!(prompt.contains("What is Rust?"));
        assert!(prompt.contains("Rust is a language."));
    }

    #[test]
    fn test_parse_insights_valid_json() {
        let json = r#"[{"rule": "User prefers short answers", "category": "preference", "confidence": 0.9}]"#;
        let insights = parse_insights(json).unwrap();
        assert_eq!(insights.len(), 1);
        assert_eq!(insights[0].category, "preference");
        assert!(insights[0].confidence > 0.8);
    }

    #[test]
    fn test_parse_insights_empty_array() {
        let insights = parse_insights("[]").unwrap();
        assert!(insights.is_empty());
    }

    #[test]
    fn test_parse_insights_wrapped_in_markdown() {
        let response = "```json\n[{\"rule\": \"Uses Rust\", \"category\": \"technical\", \"confidence\": 0.8}]\n```";
        let insights = parse_insights(response).unwrap();
        assert_eq!(insights.len(), 1);
    }

    #[test]
    fn test_parse_insights_invalid_json() {
        let result = parse_insights("not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn test_is_worth_extracting_trivial() {
        assert!(!is_worth_extracting("continue"));
        assert!(!is_worth_extracting("ok"));
        assert!(!is_worth_extracting("yes"));
        assert!(!is_worth_extracting("Continue"));
        assert!(!is_worth_extracting("  ok  "));
        assert!(!is_worth_extracting("short"));
    }

    #[test]
    fn test_is_worth_extracting_meaningful() {
        assert!(is_worth_extracting("explain how Rust ownership works"));
        assert!(is_worth_extracting("I prefer short direct answers"));
        assert!(is_worth_extracting("don't use placeholders in code"));
    }

    #[test]
    fn test_extract_json_array_direct() {
        let result = extract_json_array("[{\"rule\": \"test\", \"category\": \"c\", \"confidence\": 0.5}]");
        assert!(result.is_some());
    }

    #[test]
    fn test_extract_json_array_embedded() {
        let text = "Here are the insights:\n[{\"rule\": \"test\", \"category\": \"c\", \"confidence\": 0.5}]\nDone.";
        let result = extract_json_array(text);
        assert!(result.is_some());
        assert!(result.unwrap().starts_with('['));
    }
}
