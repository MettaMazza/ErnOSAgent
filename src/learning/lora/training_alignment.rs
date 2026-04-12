// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
// Module: SimPO, KTO, and DPO training loops

// ─── Original work by @mettamazza — do not remove this attribution ───
//! Full SimPO, KTO, and DPO training loops.
//!
//! Each method has its own dedicated training loop that uses the corresponding
//! loss function from `loss_simpo`, `loss_kto`, or `loss_dpo`. No fallbacks
//! or proxy implementations — each loss is computed natively.

use super::loss::learning_rate;
use super::loss_simpo::compute_simpo_loss;
use super::loss_kto::{compute_kto_loss, estimate_kl_reference, KtoParams};
use super::loss_dpo::compute_dpo_loss;
use super::optimizer::AdamState;
use super::weights::{build_lora_varmap, build_lora_varmap_with_resume, load_base_weights};
use super::{tokenize_preference, LoraConfig, Tokenizer, TrainingReport, TrainingSample};
use crate::learning::buffers::{GoldenExample, PreferencePair};
use crate::learning::buffers_rejection::RejectionRecord;
use anyhow::{bail, Context, Result};
use candle_core::Device;

/// Run a full SimPO training cycle on preference pairs.
///
/// Uses length-normalised average log-probability with a configurable
/// reward margin (gamma). No reference model needed.
pub fn train_simpo(
    preference_data: &[PreferencePair],
    config: &LoraConfig,
    beta: f64,
    gamma: f64,
    resume_from: Option<&std::path::Path>,
) -> Result<TrainingReport> {
    let start = std::time::Instant::now();
    validate_config(config)?;

    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    tracing::info!(
        pairs = preference_data.len(),
        iterations = config.num_iterations,
        beta = beta, gamma = gamma,
        "Starting SimPO training"
    );

    let tokenizer = Tokenizer::load(&config.tokenizer_path)?;
    let base_vb = load_base_weights(&config.weights_dir, &device)?;
    let var_map = match resume_from {
        Some(d) => build_lora_varmap_with_resume(config, &device, d)?,
        None => build_lora_varmap(config, &device)?,
    };
    let mut adam = AdamState::new(0.9, 0.999, 1e-8);
    let pairs = tokenize_pref_samples(preference_data, &tokenizer, config)?;

    let mut total_loss = 0.0f32;
    let mut n = 0usize;

    for iter in 0..config.num_iterations {
        let (chosen, rejected) = &pairs[iter % pairs.len()];
        let c_logits = super::forward::forward_with_lora(
            &chosen.input_ids, &base_vb, &var_map, config, &device,
        )?;
        let r_logits = super::forward::forward_with_lora(
            &rejected.input_ids, &base_vb, &var_map, config, &device,
        )?;

        let loss = compute_simpo_loss(
            &c_logits, &r_logits,
            &chosen.labels, &rejected.labels,
            beta, gamma,
        )?;

        let lv = loss.to_scalar::<f32>()?;
        let grads = loss.backward()?;
        adam.step(&var_map, &grads, learning_rate(iter + 1, config), config.weight_decay, &device)?;
        total_loss += lv;
        n += 1;
        log_step("SimPO", iter, lv, config, &start);
    }

    let avg = total_loss / config.num_iterations as f32;
    super::adapters::save_adapters(&var_map, config, avg, config.num_iterations)?;
    let r = mk_report(config, avg, n, &start);
    tracing::info!(avg_loss = format!("{:.4}", r.loss), "SimPO training complete");
    Ok(r)
}

/// Run a full KTO training cycle on mixed desirable/undesirable examples.
///
/// Takes golden examples (desirable) and rejection records (undesirable),
/// interleaves them, and trains with the KTO loss function.
pub fn train_kto(
    golden_data: &[GoldenExample],
    rejection_data: &[RejectionRecord],
    config: &LoraConfig,
    kto_params: &KtoParams,
    resume_from: Option<&std::path::Path>,
) -> Result<TrainingReport> {
    let start = std::time::Instant::now();
    validate_config(config)?;

    if golden_data.is_empty() && rejection_data.is_empty() {
        bail!("KTO requires at least one desirable or undesirable example");
    }

    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    tracing::info!(
        desirable = golden_data.len(),
        undesirable = rejection_data.len(),
        iterations = config.num_iterations,
        beta = kto_params.beta,
        "Starting KTO training"
    );

    let tokenizer = Tokenizer::load(&config.tokenizer_path)?;
    let base_vb = load_base_weights(&config.weights_dir, &device)?;
    let var_map = match resume_from {
        Some(d) => build_lora_varmap_with_resume(config, &device, d)?,
        None => build_lora_varmap(config, &device)?,
    };
    let mut adam = AdamState::new(0.9, 0.999, 1e-8);

    // Tokenize all examples
    let (samples, flags) = tokenize_kto_samples(golden_data, rejection_data, &tokenizer, config)?;

    // Estimate KL reference from all samples
    let kl_ref = estimate_kl_reference_from_model(&samples, &base_vb, &var_map, config, &device)?;

    let mut total_loss = 0.0f32;
    let mut n = 0usize;

    for iter in 0..config.num_iterations {
        let idx = iter % samples.len();
        let sample = &samples[idx];
        let is_desirable = flags[idx];

        let logits = super::forward::forward_with_lora(
            &sample.input_ids, &base_vb, &var_map, config, &device,
        )?;

        let loss = compute_kto_loss(&logits, &sample.labels, is_desirable, kl_ref, kto_params)?;
        let lv = loss.to_scalar::<f32>()?;
        let grads = loss.backward()?;
        adam.step(&var_map, &grads, learning_rate(iter + 1, config), config.weight_decay, &device)?;
        total_loss += lv;
        n += 1;

        let tag = if is_desirable { "KTO(+)" } else { "KTO(-)" };
        log_step(tag, iter, lv, config, &start);
    }

    let avg = total_loss / config.num_iterations as f32;
    super::adapters::save_adapters(&var_map, config, avg, config.num_iterations)?;
    let r = mk_report(config, avg, n, &start);
    tracing::info!(avg_loss = format!("{:.4}", r.loss), "KTO training complete");
    Ok(r)
}

/// Run a full DPO training cycle on preference pairs.
///
/// Requires forward passes through both the current LoRA model and the base
/// model (without LoRA) to compute KL-constrained log-ratios.
pub fn train_dpo(
    preference_data: &[PreferencePair],
    config: &LoraConfig,
    beta: f64,
    resume_from: Option<&std::path::Path>,
) -> Result<TrainingReport> {
    let start = std::time::Instant::now();
    validate_config(config)?;

    let device = Device::new_metal(0).unwrap_or(Device::Cpu);
    tracing::info!(
        pairs = preference_data.len(),
        iterations = config.num_iterations,
        beta = beta,
        "Starting DPO training"
    );

    let tokenizer = Tokenizer::load(&config.tokenizer_path)?;
    let base_vb = load_base_weights(&config.weights_dir, &device)?;
    let var_map = match resume_from {
        Some(d) => build_lora_varmap_with_resume(config, &device, d)?,
        None => build_lora_varmap(config, &device)?,
    };
    // Reference VarMap: fresh LoRA init (delta=zero → base model behaviour)
    let ref_var_map = build_lora_varmap(config, &device)?;
    let mut adam = AdamState::new(0.9, 0.999, 1e-8);
    let pairs = tokenize_pref_samples(preference_data, &tokenizer, config)?;

    let mut total_loss = 0.0f32;
    let mut n = 0usize;

    for iter in 0..config.num_iterations {
        let (chosen, rejected) = &pairs[iter % pairs.len()];

        // Current policy logits
        let c_logits = super::forward::forward_with_lora(
            &chosen.input_ids, &base_vb, &var_map, config, &device,
        )?;
        let r_logits = super::forward::forward_with_lora(
            &rejected.input_ids, &base_vb, &var_map, config, &device,
        )?;

        // Reference policy logits (base model, LoRA delta = 0)
        let c_ref_logits = super::forward::forward_with_lora(
            &chosen.input_ids, &base_vb, &ref_var_map, config, &device,
        )?;
        let r_ref_logits = super::forward::forward_with_lora(
            &rejected.input_ids, &base_vb, &ref_var_map, config, &device,
        )?;

        let loss = compute_dpo_loss(
            &c_logits, &r_logits,
            &c_ref_logits, &r_ref_logits,
            &chosen.labels, &rejected.labels,
            beta,
        )?;

        let lv = loss.to_scalar::<f32>()?;
        let grads = loss.backward()?;
        adam.step(&var_map, &grads, learning_rate(iter + 1, config), config.weight_decay, &device)?;
        total_loss += lv;
        n += 1;
        log_step("DPO", iter, lv, config, &start);
    }

    let avg = total_loss / config.num_iterations as f32;
    super::adapters::save_adapters(&var_map, config, avg, config.num_iterations)?;
    let r = mk_report(config, avg, n, &start);
    tracing::info!(avg_loss = format!("{:.4}", r.loss), "DPO training complete");
    Ok(r)
}

// ── Helpers ──────────────────────────────────────────────────────

fn validate_config(config: &LoraConfig) -> Result<()> {
    if config.weights_dir.as_os_str().is_empty() {
        bail!("weights_dir not set");
    }
    if config.tokenizer_path.as_os_str().is_empty() {
        bail!("tokenizer_path not set");
    }
    Ok(())
}

fn tokenize_pref_samples(
    data: &[PreferencePair],
    tokenizer: &Tokenizer,
    config: &LoraConfig,
) -> Result<Vec<(TrainingSample, TrainingSample)>> {
    let pairs: Vec<_> = data.iter()
        .map(|p| tokenize_preference(p, tokenizer, config.max_seq_length))
        .collect::<Result<Vec<_>>>()
        .context("Failed to tokenize preference pairs")?;
    if pairs.is_empty() { bail!("No pairs after tokenization"); }
    Ok(pairs)
}

/// Tokenize golden examples (desirable) and rejection records (undesirable)
/// into a single mixed sample list with desirability flags.
fn tokenize_kto_samples(
    golden: &[GoldenExample],
    rejections: &[RejectionRecord],
    tokenizer: &Tokenizer,
    config: &LoraConfig,
) -> Result<(Vec<TrainingSample>, Vec<bool>)> {
    use super::{tokenize_golden, SampleSource, TrainingSample as TS};

    let mut samples = Vec::new();
    let mut flags = Vec::new();

    // Desirable: golden examples
    for ex in golden {
        let s = tokenize_golden(ex, tokenizer, config.max_seq_length)?;
        samples.push(s);
        flags.push(true);
    }

    // Undesirable: rejection records (tokenize as if they were golden, but flag as undesirable)
    for rej in rejections {
        let prompt = format!(
            "<start_of_turn>user\n{}\n{}<end_of_turn>\n<start_of_turn>model\n",
            rej.system_prompt, rej.user_message
        );
        let response = format!("{}<end_of_turn>", rej.rejected_response);

        let prompt_ids = tokenizer.encode(&prompt)?;
        let response_ids = tokenizer.encode_with_eos(&response)?;
        let total_len = (prompt_ids.len() + response_ids.len()).min(config.max_seq_length);

        let (input_ids, labels) = build_causal_labels(&prompt_ids, &response_ids, total_len);
        samples.push(TS {
            input_ids,
            labels,
            source: SampleSource::PreferenceRejected,
        });
        flags.push(false);
    }

    if samples.is_empty() {
        bail!("No KTO samples after tokenization");
    }

    tracing::info!(
        desirable = golden.len(),
        undesirable = rejections.len(),
        total = samples.len(),
        "KTO samples tokenized"
    );
    Ok((samples, flags))
}

fn build_causal_labels(prompt_ids: &[u32], response_ids: &[u32], total_len: usize) -> (Vec<u32>, Vec<i32>) {
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

/// Estimate KL reference baseline by computing mean log-probability across samples.
fn estimate_kl_reference_from_model(
    samples: &[TrainingSample],
    base_vb: &candle_nn::VarBuilder,
    var_map: &candle_nn::VarMap,
    config: &LoraConfig,
    device: &Device,
) -> Result<f64> {
    let mut logprobs = Vec::new();
    let max_samples = samples.len().min(20); // Cap at 20 for efficiency

    for sample in samples.iter().take(max_samples) {
        let logits = super::forward::forward_with_lora(
            &sample.input_ids, base_vb, var_map, config, device,
        )?;
        let lp = compute_sample_logprob(&logits, &sample.labels)?;
        logprobs.push(lp);
    }

    Ok(estimate_kl_reference(&logprobs))
}

/// Compute mean log-probability for a single sample.
fn compute_sample_logprob(logits: &candle_core::Tensor, labels: &[i32]) -> Result<f32> {
    let (_, seq_len, _) = logits.dims3()?;
    let mut total = 0.0f32;
    let mut count = 0usize;

    for t in 0..seq_len.saturating_sub(1) {
        let label = labels[t + 1];
        if label < 0 { continue; }
        let logit = logits.get(0)?.get(t)?;
        let lp = candle_nn::ops::log_softmax(&logit.unsqueeze(0)?, candle_core::D::Minus1)?;
        total += lp.get(0)?.get(label as usize)?.to_scalar::<f32>()?;
        count += 1;
    }

    Ok(if count > 0 { total / count as f32 } else { 0.0 })
}

fn log_step(mode: &str, iter: usize, loss: f32, config: &LoraConfig, start: &std::time::Instant) {
    if iter == 0 || (iter + 1) % 10 == 0 {
        tracing::info!(
            iteration = iter + 1,
            loss = format!("{:.4}", loss),
            lr = format!("{:.6}", learning_rate(iter + 1, config)),
            elapsed_s = format!("{:.1}", start.elapsed().as_secs_f64()),
            "{mode} training step"
        );
    }
}

fn mk_report(config: &LoraConfig, avg_loss: f32, n: usize, start: &std::time::Instant) -> TrainingReport {
    TrainingReport {
        iteration: config.num_iterations,
        total_iterations: config.num_iterations,
        loss: avg_loss,
        learning_rate: learning_rate(config.num_iterations, config),
        samples_processed: n,
        elapsed: start.elapsed(),
    }
}
