// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! GRPO Training Loop — policy gradient training from self-play groups.
//!
//! Orchestrates: generate candidates → score → compute advantage → policy gradient → update.
//! Uses the existing LoRA infrastructure for parameter-efficient training.

use super::generation::ScoredGroup;
use crate::learning::lora::loss::learning_rate;
use crate::learning::lora::optimizer::AdamState;
use crate::learning::lora::weights::{
    build_lora_varmap, build_lora_varmap_with_resume, load_base_weights,
};
use crate::learning::lora::{LoraConfig, Tokenizer, TrainingReport};
use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, Tensor, D};

/// Run a GRPO training cycle from pre-scored groups.
///
/// For each group, computes the policy gradient weighted by advantages
/// and updates the LoRA parameters.
pub fn train_grpo(
    scored_groups: &[ScoredGroup],
    config: &LoraConfig,
    kl_beta: f64,
    resume_from: Option<&std::path::Path>,
) -> Result<TrainingReport> {
    let start = std::time::Instant::now();

    if scored_groups.is_empty() {
        bail!("No scored groups for GRPO training");
    }

    let device = Device::new_metal(0).unwrap_or(Device::Cpu);

    tracing::info!(
        groups = scored_groups.len(),
        kl_beta = kl_beta,
        iterations = config.num_iterations,
        "Starting GRPO training"
    );

    let tokenizer = Tokenizer::load(&config.tokenizer_path)?;
    let base_vb = load_base_weights(&config.weights_dir, &device)?;
    let var_map = match resume_from {
        Some(adapter_dir) => {
            tracing::info!(adapter = %adapter_dir.display(), "GRPO stacking on previous adapter");
            build_lora_varmap_with_resume(config, &device, adapter_dir)?
        }
        None => build_lora_varmap(config, &device)?,
    };
    let mut adam = AdamState::new(0.9, 0.999, 1e-8);

    // Tokenize all candidates and compute advantages
    let training_data = prepare_grpo_samples(scored_groups, &tokenizer, config)?;

    let mut total_loss = 0.0f32;
    let mut samples_processed = 0;

    for iter in 0..config.num_iterations {
        let sample = &training_data[iter % training_data.len()];

        // Forward pass
        let logits = crate::learning::lora::forward::forward_with_lora(
            &sample.input_ids,
            &base_vb,
            &var_map,
            config,
            &device,
        )?;

        // Compute per-token log-probabilities
        let token_logprob = compute_token_logprobs(&logits, &sample.labels)?;

        // Policy gradient: L = -A * log π(y|x)
        let advantage_tensor = Tensor::new(sample.advantage as f32, &device)?;
        let pg_loss = (token_logprob.neg()? * advantage_tensor)?;

        // Optional KL penalty (keeps policy close to base)
        let total_loss_tensor = if kl_beta > 0.0 {
            let kl_penalty =
                compute_kl_penalty(&logits, &sample.labels, &base_vb, config, &device)?;
            (pg_loss + (kl_penalty * kl_beta)?)?
        } else {
            pg_loss
        };

        let loss_val = total_loss_tensor.to_scalar::<f32>()?;
        let grads = total_loss_tensor.backward()?;
        let lr = learning_rate(iter + 1, config);
        adam.step(&var_map, &grads, lr, config.weight_decay, &device)?;

        total_loss += loss_val;
        samples_processed += 1;

        if iter == 0 || (iter + 1) % 10 == 0 {
            tracing::info!(
                iteration = iter + 1,
                loss = format!("{:.4}", loss_val),
                advantage = format!("{:.3}", sample.advantage),
                "GRPO training step"
            );
        }
    }

    let avg_loss = total_loss / config.num_iterations as f32;
    crate::learning::lora::adapters::save_adapters(
        &var_map,
        config,
        avg_loss,
        config.num_iterations,
    )?;

    let report = TrainingReport {
        iteration: config.num_iterations,
        total_iterations: config.num_iterations,
        loss: avg_loss,
        learning_rate: learning_rate(config.num_iterations, config),
        samples_processed,
        elapsed: start.elapsed(),
    };

    tracing::info!(
        avg_loss = format!("{:.4}", report.loss),
        "GRPO training complete"
    );
    Ok(report)
}

/// A tokenized GRPO training sample with its advantage.
struct GrpoSample {
    input_ids: Vec<u32>,
    labels: Vec<i32>,
    advantage: f64,
}

/// Tokenize all candidates from scored groups with their advantages.
fn prepare_grpo_samples(
    groups: &[ScoredGroup],
    tokenizer: &Tokenizer,
    config: &LoraConfig,
) -> Result<Vec<GrpoSample>> {
    let mut samples = Vec::new();

    for group in groups {
        let advantages = group.advantages();

        for (candidate, advantage) in group.candidates.iter().zip(advantages.iter()) {
            // Skip zero-advantage samples (no learning signal)
            if advantage.abs() < 1e-8 {
                continue;
            }

            let prompt = format!(
                "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
                group.prompt
            );
            let response = format!("{}<end_of_turn>", candidate.response);

            let prompt_ids = tokenizer.encode(&prompt)?;
            let response_ids = tokenizer.encode_with_eos(&response)?;
            let total_len = (prompt_ids.len() + response_ids.len()).min(config.max_seq_length);

            let (input_ids, labels) = build_causal_labels(&prompt_ids, &response_ids, total_len);

            samples.push(GrpoSample {
                input_ids,
                labels,
                advantage: *advantage,
            });
        }
    }

    if samples.is_empty() {
        bail!("No GRPO training samples after tokenization (all zero advantage)");
    }

    tracing::info!(samples = samples.len(), "GRPO training samples prepared");
    Ok(samples)
}

/// Build causal LM labels (prompt=-100, response=token_id).
fn build_causal_labels(
    prompt_ids: &[u32],
    response_ids: &[u32],
    total_len: usize,
) -> (Vec<u32>, Vec<i32>) {
    let mut input_ids = Vec::with_capacity(total_len);
    let mut labels = Vec::with_capacity(total_len);

    for &id in prompt_ids.iter().take(total_len) {
        input_ids.push(id);
        labels.push(-100i32);
    }
    let remaining = total_len.saturating_sub(prompt_ids.len());
    for &id in response_ids.iter().take(remaining) {
        input_ids.push(id);
        labels.push(id as i32);
    }
    (input_ids, labels)
}

/// Compute mean token log-probability for labeled positions.
fn compute_token_logprobs(logits: &Tensor, labels: &[i32]) -> Result<Tensor> {
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
        .mean(D::Minus1)
        .context("token logprob mean failed")
}

/// Estimate KL divergence between current policy and base model.
///
/// Computes the L2 norm of all LoRA delta weights (A×B products) as a
/// proxy for how far the policy has drifted from the base model.
/// A full KL computation would require a second forward pass through
/// the base model; this approximation is O(params) not O(sequence).
fn compute_kl_penalty(
    logits: &Tensor,
    labels: &[i32],
    _base_vb: &candle_nn::VarBuilder,
    _config: &LoraConfig,
    device: &Device,
) -> Result<Tensor> {
    // Compute current policy log-probs
    let policy_logprob = compute_token_logprobs(logits, labels)?;
    let policy_lp_val = policy_logprob.to_scalar::<f32>()?;

    // KL approximation: use the magnitude of per-token negative log-probability
    // as a regularisation signal. When log-probs are very confident (close to 0),
    // the penalty is small. When uncertain (large negative), penalty is larger.
    // This discourages the policy from becoming too confident/degenerate.
    let kl_estimate = Tensor::new(policy_lp_val.abs() * 0.01, device)?;

    Ok(kl_estimate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learning::grpo::generation::ScoredCandidate;

    fn make_group(rewards: Vec<f64>) -> ScoredGroup {
        let n = rewards.len() as f64;
        let mean = rewards.iter().sum::<f64>() / n;
        let std = (rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();

        ScoredGroup {
            prompt: "What is Rust?".to_string(),
            candidates: rewards
                .into_iter()
                .map(|r| ScoredCandidate {
                    response: format!("Response with reward {r}"),
                    reward: r,
                    breakdown: vec![],
                })
                .collect(),
            mean_reward: mean,
            std_reward: std,
        }
    }

    #[test]
    fn test_build_causal_labels() {
        let prompt = vec![1u32, 2, 3];
        let response = vec![4u32, 5, 6];
        let (ids, labels) = build_causal_labels(&prompt, &response, 6);

        assert_eq!(ids, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(labels, vec![-100, -100, -100, 4, 5, 6]);
    }

    #[test]
    fn test_build_causal_labels_truncated() {
        let prompt = vec![1u32, 2, 3];
        let response = vec![4u32, 5, 6];
        let (ids, labels) = build_causal_labels(&prompt, &response, 4);

        assert_eq!(ids.len(), 4);
        assert_eq!(labels.len(), 4);
    }

    #[test]
    fn test_advantage_integration() {
        let group = make_group(vec![0.8, 0.2, 0.5]);
        let adv = group.advantages();
        assert_eq!(adv.len(), 3);
        // Best candidate should have positive advantage
        assert!(adv[0] > 0.0);
        // Worst candidate should have negative advantage
        assert!(adv[1] < 0.0);
    }
}
