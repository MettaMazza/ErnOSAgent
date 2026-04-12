// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Sleep Reflection — identity reflection generated during sleep cycles.
//!
//! During the sleep cycle, the agent reviews its best interactions and lessons,
//! then generates a self-analysis that is appended as a golden example.
//! This mirrors HIVE's `generate_identity_reflection()` pattern.

use crate::learning::buffers::{GoldenExample, TrainingBuffers};
use crate::provider::{Message, Provider};
use std::sync::Arc;

/// Attempt to generate an identity reflection from the top-ranked examples.
///
/// Returns true if a reflection was successfully generated and stored.
/// Failure is non-fatal — the sleep cycle continues regardless.
pub async fn try_generate_reflection(
    provider: &Arc<dyn Provider>,
    buffers: &Arc<TrainingBuffers>,
    ranked_examples: &[(GoldenExample, f64)],
) -> bool {
    match generate_reflection(provider, ranked_examples).await {
        Ok(reflection) => {
            store_reflection(buffers, &reflection);
            true
        }
        Err(e) => {
            tracing::warn!(error = %e, "Identity reflection failed — non-fatal");
            false
        }
    }
}

/// Generate an identity reflection by reviewing the best interactions.
async fn generate_reflection(
    provider: &Arc<dyn Provider>,
    ranked_examples: &[(GoldenExample, f64)],
) -> anyhow::Result<String> {
    if ranked_examples.is_empty() {
        anyhow::bail!("No examples to reflect on");
    }

    let prompt = build_reflection_prompt(ranked_examples);

    // Use the provider's model for reflection inference
    let models = provider.list_models().await?;
    let model_name = models.first()
        .map(|m| m.name.as_str())
        .unwrap_or("default");

    let messages = vec![
        Message {
            role: "user".to_string(),
            content: prompt,
            images: Vec::new(),
        },
    ];

    let response = provider.chat_sync(model_name, &messages, Some(0.3)).await?;

    tracing::info!(
        reflection_len = response.len(),
        examples_reviewed = ranked_examples.len(),
        "Identity reflection generated"
    );

    Ok(response)
}

/// Build the reflection prompt from the top-ranked examples.
fn build_reflection_prompt(examples: &[(GoldenExample, f64)]) -> String {
    let mut prompt = String::from(
        "You are ErnOS, an AI agent, performing a self-reflection during a sleep/training cycle.\n\n\
         Review the following interactions (your best recent work, ranked by quality):\n\n"
    );

    for (i, (ex, score)) in examples.iter().take(5).enumerate() {
        let user_preview = truncate_preview(&ex.user_message, 150);
        let response_preview = truncate_preview(&ex.assistant_response, 300);
        prompt.push_str(&format!(
            "--- Interaction {} (quality score: {score:.1}) ---\n\
             User: {user_preview}\n\
             Your response: {response_preview}\n\n",
            i + 1
        ));
    }

    prompt.push_str(
        "Based on these interactions, write a brief self-reflection:\n\
         1. What patterns in your strongest responses should you reinforce?\n\
         2. What cognitive strategies led to high-quality outputs?\n\
         3. What should you focus on improving?\n\n\
         Write your reflection as a natural self-assessment (2-3 paragraphs)."
    );

    prompt
}

/// Truncate text to a preview length, adding ellipsis.
fn truncate_preview(text: &str, max_len: usize) -> String {
    if text.len() <= max_len {
        text.to_string()
    } else {
        format!("{}...", &text[..max_len])
    }
}

/// Store the reflection as a golden example for future training.
fn store_reflection(buffers: &Arc<TrainingBuffers>, reflection: &str) {
    let system_prompt = "You are ErnOS, performing self-reflection during a sleep cycle.";
    let user_message = "Reflect on your recent performance and identify patterns for improvement.";

    if let Err(e) = buffers.golden.record(
        system_prompt,
        user_message,
        reflection,
        "sleep:reflection",
        "self",
    ) {
        tracing::warn!(error = %e, "Failed to store reflection as golden example");
    } else {
        tracing::info!("Identity reflection stored as golden example");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_preview_short() {
        assert_eq!(truncate_preview("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_preview_long() {
        let long = "a".repeat(200);
        let result = truncate_preview(&long, 50);
        assert_eq!(result.len(), 53); // 50 + "..."
        assert!(result.ends_with("..."));
    }

    #[test]
    fn test_build_reflection_prompt() {
        let examples = vec![(
            GoldenExample {
                system_prompt: "sys".to_string(),
                user_message: "What is Rust?".to_string(),
                assistant_response: "Rust is a systems programming language.".to_string(),
                session_id: "s1".to_string(),
                model_id: "test".to_string(),
                timestamp: chrono::Utc::now(),
            },
            3.5,
        )];
        let prompt = build_reflection_prompt(&examples);
        assert!(prompt.contains("self-reflection"));
        assert!(prompt.contains("What is Rust?"));
        assert!(prompt.contains("quality score: 3.5"));
    }
}
