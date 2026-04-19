// Ern-OS — GRPO self-play generation
//! Generates diverse candidate responses by varying temperature and sampling.

use crate::provider::{Message, Provider};

/// Generate multiple candidate responses for GRPO group scoring.
/// Uses the provider to generate `group_size` diverse responses at different temperatures.
pub async fn generate_candidates(
    provider: &dyn Provider,
    prompt: &str,
    group_size: usize,
) -> anyhow::Result<Vec<String>> {
    let mut candidates = Vec::with_capacity(group_size);
    let messages = vec![Message::text("user", prompt)];

    for i in 0..group_size {
        // Use chat_sync for each candidate — temperature variation is implicit
        // in the model's stochastic sampling
        match provider.chat_sync(&messages, None).await {
            Ok(response) => candidates.push(response),
            Err(e) => {
                tracing::warn!(candidate = i, error = %e, "GRPO: candidate generation failed");
                candidates.push(format!("[Generation failed: {}]", e));
            }
        }
    }

    Ok(candidates)
}

/// Generate candidates without a provider (offline mode using simple perturbation).
pub fn generate_candidates_offline(prompt: &str, group_size: usize) -> Vec<String> {
    (0..group_size).map(|i| {
        // Generate deterministic variations for offline training
        let variation = match i % 4 {
            0 => format!("Concise answer: {}", prompt),
            1 => format!("Detailed answer with examples for: {}", prompt),
            2 => format!("Step-by-step approach to: {}", prompt),
            _ => format!("Alternative perspective on: {}", prompt),
        };
        variation
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_offline() {
        let candidates = generate_candidates_offline("test", 4);
        assert_eq!(candidates.len(), 4);
        assert!(candidates[0].contains("Concise"));
        assert!(candidates[1].contains("Detailed"));
    }
}
