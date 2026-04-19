// Ern-OS — GRPO training loop
//! Advantage-weighted policy gradient using group-relative scoring.

use super::rewards;

/// GRPO loss: -Σ advantage_i * log_prob_i / group_size
pub fn grpo_loss(log_probs: &[f32], advantages: &[f32]) -> f32 {
    if log_probs.len() != advantages.len() || log_probs.is_empty() { return 0.0; }
    let n = log_probs.len() as f32;
    -(log_probs.iter().zip(advantages).map(|(lp, a)| lp * a).sum::<f32>()) / n
}

/// Estimate log-probability of a candidate response via response quality proxy.
/// In production, this would use the model's actual token-level log-probs.
/// Here we approximate using a normalized quality score as a log-prob proxy.
fn estimate_log_probs(candidates: &[String], query: &str) -> Vec<f32> {
    let scores = rewards::score_group(candidates, query);
    // Convert scores to log-probability-like values (negative, higher = better)
    scores.iter().map(|s| {
        // Sigmoid-then-log to map score → log-prob range
        let prob = 1.0 / (1.0 + (-s).exp());
        prob.max(1e-8).ln()
    }).collect()
}

/// Run a single GRPO training step.
/// Returns the policy gradient loss.
pub fn train_step(
    candidates: &[String],
    query: &str,
    beta: f32,
) -> f64 {
    let scores = rewards::score_group(candidates, query);
    let advantages = rewards::compute_advantages(&scores);
    let log_probs = estimate_log_probs(candidates, query);

    // KL penalty: beta * KL(policy || reference)
    // Approximated as beta * mean(log_prob^2) to keep policy close to reference
    let kl_penalty: f32 = beta * log_probs.iter().map(|lp| lp.powi(2)).sum::<f32>()
        / log_probs.len().max(1) as f32;

    let policy_loss = grpo_loss(&log_probs, &advantages);
    (policy_loss + kl_penalty) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grpo_loss() {
        let log_probs = vec![-1.0, -2.0, -0.5, -1.5];
        let advantages = vec![1.0, -0.5, 0.5, -1.0];
        let loss = grpo_loss(&log_probs, &advantages);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_train_step() {
        let candidates = vec!["Good response here".into(), "Short".into(), "Another decent one".into()];
        let loss = train_step(&candidates, "test query", 0.1);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_estimate_log_probs() {
        let candidates = vec!["Hello".into(), "World".into()];
        let lps = estimate_log_probs(&candidates, "test");
        assert_eq!(lps.len(), 2);
        assert!(lps.iter().all(|lp| *lp <= 0.0)); // Log probs are negative
    }
}
