// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Loss functions and learning rate schedule.

use super::LoraConfig;
use anyhow::{Context, Result};
use candle_core::{DType, Tensor, D};

/// Cross-entropy loss for causal language modelling.
/// Ignores positions where labels == -100.
pub(crate) fn cross_entropy_loss(logits: &Tensor, labels: &[i32]) -> Result<Tensor> {
    let (_, seq_len, _) = logits
        .dims3()
        .context("logits must be [1, seq_len, vocab_size]")?;

    let mut token_losses: Vec<Tensor> = Vec::new();

    for t in 0..seq_len.saturating_sub(1) {
        let label = labels[t + 1];
        if label < 0 {
            continue;
        }
        let logit = logits.get(0)?.get(t)?;
        let log_probs = candle_nn::ops::log_softmax(&logit.unsqueeze(0)?, D::Minus1)?;
        let nll = log_probs.get(0)?.get(label as usize)?.neg()?;
        token_losses.push(nll);
    }

    if token_losses.is_empty() {
        return Tensor::zeros((), DType::F32, logits.device()).context("zero loss");
    }

    let stacked = Tensor::stack(&token_losses, 0)?;
    stacked.mean(D::Minus1).context("mean loss failed")
}

/// ORPO loss: SFT(chosen) + β × log-sigmoid(-log_odds_ratio).
pub fn compute_orpo_loss(
    chosen_logits: &Tensor,
    rejected_logits: &Tensor,
    chosen_labels: &[i32],
    rejected_labels: &[i32],
    chosen_sft_loss: &Tensor,
    beta: f64,
) -> Result<Tensor> {
    let chosen_logprob = sequence_logprob(chosen_logits, chosen_labels)?;
    let rejected_logprob = sequence_logprob(rejected_logits, rejected_labels)?;

    let log_odds = (chosen_logprob - rejected_logprob)?;
    let scaled = (log_odds * (-beta))?;
    let orpo_penalty = (Tensor::ones_like(&scaled)? + scaled.exp()?)?.log()?;

    (chosen_sft_loss + orpo_penalty).context("orpo loss sum failed")
}

/// Sum of log-probabilities for non-ignored label positions.
fn sequence_logprob(logits: &Tensor, labels: &[i32]) -> Result<Tensor> {
    let (_, seq_len, _) = logits.dims3()?;
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

    Tensor::stack(&probs, 0)?
        .sum(D::Minus1)
        .context("sequence logprob sum failed")
}

/// Linear warmup + cosine decay learning rate schedule.
pub(crate) fn learning_rate(iteration: usize, config: &LoraConfig) -> f64 {
    let base = config.learning_rate;
    if iteration <= config.warmup_steps {
        base * (iteration as f64 / config.warmup_steps.max(1) as f64)
    } else {
        let progress = (iteration - config.warmup_steps) as f64
            / (config.num_iterations - config.warmup_steps).max(1) as f64;
        let cosine = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
        base * cosine
    }
}
