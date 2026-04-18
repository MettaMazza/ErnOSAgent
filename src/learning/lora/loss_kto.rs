// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! KTO loss — Kahneman-Tversky Optimization (binary signal alignment).
//!
//! KTO uses binary desirable/undesirable labels per individual generation,
//! inspired by Prospect Theory. Losses are weighted more heavily than gains
//! (loss aversion). Does not require paired data — every single Observer
//! PASS/FAIL signal is training data.
//!
//! Paper: "KTO: Model Alignment as Prospect Theoretic Optimization"

use anyhow::{Context, Result};
use candle_core::{DType, Tensor, D};

/// Parameters for KTO loss computation.
#[derive(Debug, Clone)]
pub struct KtoParams {
    /// Reward scaling factor (higher = stronger signal).
    pub beta: f64,
    /// Weight for desirable examples (typically 1.0).
    pub lambda_d: f64,
    /// Weight for undesirable examples (typically > 1.0 for loss aversion).
    pub lambda_u: f64,
}

impl KtoParams {
    /// Load from environment variables per no-limits governance.
    pub fn from_env() -> Result<Self> {
        let beta = parse_env_f64("ERNOS_KTO_BETA")?;
        let lambda_d = parse_env_f64("ERNOS_KTO_LAMBDA_D")?;
        let lambda_u = parse_env_f64("ERNOS_KTO_LAMBDA_U")?;

        Ok(Self {
            beta,
            lambda_d,
            lambda_u,
        })
    }
}

/// Compute KTO loss for a single example.
///
/// If desirable:   L = λ_D × (1 - σ(β(r_θ - r_ref)))
/// If undesirable: L = λ_U × (1 - σ(β(r_ref - r_θ)))
///
/// r_θ = log π_θ(y|x) (sequence log-probability)
/// r_ref = KL reference baseline (estimated from batch)
pub fn compute_kto_loss(
    logits: &Tensor,
    labels: &[i32],
    is_desirable: bool,
    kl_reference: f64,
    params: &KtoParams,
) -> Result<Tensor> {
    let token_logprob = sequence_logprob_sum(logits, labels)?;
    let reward = token_logprob;

    let kl_ref_tensor = Tensor::new(kl_reference as f32, logits.device())?;

    if is_desirable {
        // σ(β(r_θ - r_ref))
        let diff = (reward - kl_ref_tensor)?;
        let scaled = (diff * params.beta)?;
        let sigmoid_val = sigmoid(&scaled)?;
        let one = Tensor::ones_like(&sigmoid_val)?;
        let loss = ((one - sigmoid_val)? * params.lambda_d)?;
        Ok(loss)
    } else {
        // σ(β(r_ref - r_θ))
        let diff = (kl_ref_tensor - reward)?;
        let scaled = (diff * params.beta)?;
        let sigmoid_val = sigmoid(&scaled)?;
        let one = Tensor::ones_like(&sigmoid_val)?;
        let loss = ((one - sigmoid_val)? * params.lambda_u)?;
        Ok(loss)
    }
}

/// Estimate KL reference from a batch of log-probabilities.
///
/// The KL reference is the mean log-probability across the batch,
/// providing a baseline for the reward signal.
pub fn estimate_kl_reference(batch_logprobs: &[f32]) -> f64 {
    if batch_logprobs.is_empty() {
        return 0.0;
    }
    let sum: f64 = batch_logprobs.iter().map(|x| *x as f64).sum();
    sum / batch_logprobs.len() as f64
}

/// Compute batched KTO loss over mixed desirable/undesirable examples.
///
/// Returns (total_loss, desirable_count, undesirable_count).
pub fn compute_kto_batch_loss(
    logits_batch: &[Tensor],
    labels_batch: &[Vec<i32>],
    desirable_flags: &[bool],
    kl_reference: f64,
    params: &KtoParams,
) -> Result<(Tensor, usize, usize)> {
    let mut losses: Vec<Tensor> = Vec::new();
    let mut d_count = 0usize;
    let mut u_count = 0usize;

    for (i, (logits, labels)) in logits_batch.iter().zip(labels_batch).enumerate() {
        let is_desirable = desirable_flags[i];
        let loss = compute_kto_loss(logits, labels, is_desirable, kl_reference, params)?;
        losses.push(loss);
        if is_desirable {
            d_count += 1;
        } else {
            u_count += 1;
        }
    }

    if losses.is_empty() {
        let zero = Tensor::zeros((), DType::F32, &candle_core::Device::Cpu)?;
        return Ok((zero, 0, 0));
    }

    let stacked = Tensor::stack(&losses, 0)?;
    let mean_loss = stacked.mean(D::Minus1).context("KTO batch mean failed")?;
    Ok((mean_loss, d_count, u_count))
}

/// Sigmoid activation: σ(x) = 1 / (1 + exp(-x))
fn sigmoid(x: &Tensor) -> Result<Tensor> {
    let neg_x = x.neg()?;
    let exp_neg = neg_x.exp()?;
    let one = Tensor::ones_like(&exp_neg)?;
    let denom = (one + exp_neg)?;
    let result = (Tensor::ones_like(&denom)? / denom)?;
    Ok(result)
}

/// Sum of log-probabilities for non-ignored label positions.
fn sequence_logprob_sum(logits: &Tensor, labels: &[i32]) -> Result<Tensor> {
    let (_, seq_len, _) = logits
        .dims3()
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

    Tensor::stack(&probs, 0)?
        .sum(D::Minus1)
        .context("sequence logprob sum failed")
}

/// Parse a required f64 from an env var.
fn parse_env_f64(key: &str) -> Result<f64> {
    std::env::var(key)
        .map_err(|_| anyhow::anyhow!("{key} env var not set — required for KTO"))
        .and_then(|v| {
            v.parse::<f64>()
                .map_err(|e| anyhow::anyhow!("{key} is not a valid float: {e}"))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn make_logits(values: &[f32], seq_len: usize, vocab_size: usize) -> Tensor {
        Tensor::from_slice(values, (1, seq_len, vocab_size), &Device::Cpu).unwrap()
    }

    #[test]
    fn test_kl_reference_estimation() {
        let logprobs = vec![-2.0f32, -3.0, -1.5, -2.5];
        let kl_ref = estimate_kl_reference(&logprobs);
        assert!((kl_ref - (-2.25)).abs() < 1e-6);
    }

    #[test]
    fn test_kl_reference_empty() {
        assert_eq!(estimate_kl_reference(&[]), 0.0);
    }

    #[test]
    fn test_kto_loss_desirable() {
        let logits = make_logits(&[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0], 2, 4);
        let labels = vec![-100i32, 0]; // label is 0 → high prob at position 0
        let params = KtoParams {
            beta: 1.0,
            lambda_d: 1.0,
            lambda_u: 1.0,
        };

        let loss = compute_kto_loss(&logits, &labels, true, -2.0, &params).unwrap();
        let val = loss.to_scalar::<f32>().unwrap();
        assert!(val.is_finite());
        assert!(val >= 0.0, "KTO desirable loss must be non-negative: {val}");
    }

    #[test]
    fn test_kto_loss_undesirable() {
        let logits = make_logits(&[0.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0], 2, 4);
        let labels = vec![-100i32, 0]; // label is 0 → low prob at position 0
        let params = KtoParams {
            beta: 1.0,
            lambda_d: 1.0,
            lambda_u: 1.5,
        };

        let loss = compute_kto_loss(&logits, &labels, false, -2.0, &params).unwrap();
        let val = loss.to_scalar::<f32>().unwrap();
        assert!(val.is_finite());
        assert!(
            val >= 0.0,
            "KTO undesirable loss must be non-negative: {val}"
        );
    }

    #[test]
    fn test_kto_loss_aversion_weighting() {
        let logits = make_logits(&[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0], 2, 4);
        let labels = vec![-100i32, 0];
        let params_equal = KtoParams {
            beta: 1.0,
            lambda_d: 1.0,
            lambda_u: 1.0,
        };
        let params_averse = KtoParams {
            beta: 1.0,
            lambda_d: 1.0,
            lambda_u: 2.0,
        };

        let loss_equal = compute_kto_loss(&logits, &labels, false, -1.0, &params_equal)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let loss_averse = compute_kto_loss(&logits, &labels, false, -1.0, &params_averse)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        // Higher lambda_u should produce higher undesirable loss
        assert!(
            loss_averse > loss_equal,
            "Loss aversion should increase loss: {loss_averse} > {loss_equal}"
        );
    }

    #[test]
    fn test_sigmoid_correctness() {
        let x = Tensor::new(0.0f32, &Device::Cpu).unwrap();
        let result = sigmoid(&x).unwrap().to_scalar::<f32>().unwrap();
        assert!((result - 0.5).abs() < 1e-5, "σ(0) should be 0.5: {result}");
    }

    #[test]
    fn test_kto_batch_loss() {
        let logits1 = make_logits(&[3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0], 2, 4);
        let logits2 = make_logits(&[0.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0], 2, 4);
        let labels1 = vec![-100i32, 0];
        let labels2 = vec![-100i32, 0];
        let params = KtoParams {
            beta: 1.0,
            lambda_d: 1.0,
            lambda_u: 1.5,
        };

        let (loss, d, u) = compute_kto_batch_loss(
            &[logits1, logits2],
            &[labels1, labels2],
            &[true, false],
            -2.0,
            &params,
        )
        .unwrap();

        assert_eq!(d, 1);
        assert_eq!(u, 1);
        assert!(loss.to_scalar::<f32>().unwrap().is_finite());
    }
}
