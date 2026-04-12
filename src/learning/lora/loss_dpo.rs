// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! DPO loss — Direct Preference Optimization with KL-divergence constraint.
//!
//! DPO constrains the trained policy against a reference policy to prevent
//! catastrophic drift from base capabilities. Useful as a "safety brake"
//! when aggressive alignment is needed without losing general capability.
//!
//! Paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"

use anyhow::{Context, Result};
use candle_core::{DType, Tensor, D};

/// Compute the DPO loss for a preference pair.
///
/// L_DPO = -log σ(β(log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))
///
/// Requires log-probabilities from both the current model and a reference (base) model.
pub fn compute_dpo_loss(
    chosen_logits: &Tensor,
    rejected_logits: &Tensor,
    chosen_ref_logits: &Tensor,
    rejected_ref_logits: &Tensor,
    chosen_labels: &[i32],
    rejected_labels: &[i32],
    beta: f64,
) -> Result<Tensor> {
    // Current policy log-probs
    let chosen_logprob = sequence_logprob(chosen_logits, chosen_labels)?;
    let rejected_logprob = sequence_logprob(rejected_logits, rejected_labels)?;

    // Reference policy log-probs
    let chosen_ref_logprob = sequence_logprob(chosen_ref_logits, chosen_labels)?;
    let rejected_ref_logprob = sequence_logprob(rejected_ref_logits, rejected_labels)?;

    // Log-ratios: log(π_θ/π_ref)
    let chosen_ratio = (chosen_logprob - chosen_ref_logprob)?;
    let rejected_ratio = (rejected_logprob - rejected_ref_logprob)?;

    // β(chosen_ratio - rejected_ratio)
    let diff = (chosen_ratio - rejected_ratio)?;
    let scaled = (diff * beta)?;

    // -log σ(x) = log(1 + exp(-x))
    let neg_scaled = scaled.neg()?;
    let loss = (Tensor::ones_like(&neg_scaled)? + neg_scaled.exp()?)?.log()?;

    Ok(loss)
}

/// Sum of log-probabilities for non-ignored label positions.
fn sequence_logprob(logits: &Tensor, labels: &[i32]) -> Result<Tensor> {
    let (_, seq_len, _) = logits.dims3()
        .context("logits must be [1, seq_len, vocab_size]")?;

    let mut probs: Vec<Tensor> = Vec::new();

    for t in 0..seq_len.saturating_sub(1) {
        let label = labels[t + 1];
        if label < 0 {
            continue;
        }
        let logit = logits.get(0)?.get(t)?;
        let log_probs = candle_nn::ops::log_softmax(&logit.unsqueeze(0)?, D::Minus1)?;
        probs.push(log_probs.get(0)?.get(label as usize)?);
    }

    if probs.is_empty() {
        return Tensor::zeros((), DType::F32, logits.device()).context("zero logprob");
    }

    Tensor::stack(&probs, 0)?.sum(D::Minus1).context("sequence logprob sum failed")
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn make_logits(values: &[f32], seq_len: usize, vocab_size: usize) -> Tensor {
        Tensor::from_slice(values, (1, seq_len, vocab_size), &Device::Cpu).unwrap()
    }

    #[test]
    fn test_dpo_loss_chosen_better_than_ref() {
        // Current model strongly prefers chosen
        let chosen_logits = make_logits(&[5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0], 2, 4);
        let rejected_logits = make_logits(&[0.0, 5.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0], 2, 4);
        // Reference model is neutral
        let ref_logits = make_logits(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 2, 4);

        let labels = vec![-100i32, 0];

        let loss = compute_dpo_loss(
            &chosen_logits, &rejected_logits,
            &ref_logits, &ref_logits,
            &labels, &labels,
            0.1,
        ).unwrap();

        let val = loss.to_scalar::<f32>().unwrap();
        assert!(val.is_finite());
        assert!(val < 1.0, "DPO loss should be small when chosen dominates: {val}");
    }

    #[test]
    fn test_dpo_loss_beta_zero_gives_constant() {
        let chosen = make_logits(&[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0], 2, 4);
        let rejected = make_logits(&[0.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0], 2, 4);
        let ref_l = make_logits(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 2, 4);
        let labels = vec![-100i32, 0];

        let loss = compute_dpo_loss(
            &chosen, &rejected, &ref_l, &ref_l,
            &labels, &labels, 0.0,
        ).unwrap();

        let val = loss.to_scalar::<f32>().unwrap();
        // When beta=0, scaled=0, loss = log(1 + exp(0)) = log(2) ≈ 0.693
        assert!((val - 0.693).abs() < 0.01,
            "β=0 should give loss=log(2)≈0.693: {val}");
    }

    #[test]
    fn test_dpo_kl_constraint_symmetry() {
        // When current == reference, log-ratios cancel out
        let logits = make_logits(&[2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0], 2, 4);
        let labels = vec![-100i32, 0];

        let loss = compute_dpo_loss(
            &logits, &logits, &logits, &logits,
            &labels, &labels, 1.0,
        ).unwrap();

        let val = loss.to_scalar::<f32>().unwrap();
        // When chosen==rejected and current==ref, diff=0, loss=log(2)
        assert!((val - 0.693).abs() < 0.01,
            "Equal policies should give loss=log(2): {val}");
    }
}
