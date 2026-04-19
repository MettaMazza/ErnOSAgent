// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! SAE Trainer — Metal-accelerated JumpReLU Sparse Autoencoder training.
//!
//! Trains on Gemma 4 27B residual stream activations using Candle with
//! Apple Silicon Metal backend. Produces SAELens-compatible safetensors.
//!
//! Architecture: JumpReLU SAE (Gemma Scope 2 standard)
//!   Encoder: h_i = max(0, W_enc_i · x + b_enc_i - θ_i) · H(W_enc_i · x + b_enc_i - θ_i)
//!   Decoder: x̂ = W_dec · h + b_dec
//!   Loss:    L = ||x - x̂||² + λ Σ|h_i|

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, Var};
use candle_nn::VarMap;
use std::collections::HashMap;
use std::path::PathBuf;

/// Training configuration for the SAE.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Number of learned features (expansion factor × model_dim)
    pub num_features: usize,
    /// Model dimension (auto-detected from first activation batch)
    pub model_dim: usize,
    /// L1 sparsity penalty coefficient
    pub l1_coefficient: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Weight decay for AdamW
    pub weight_decay: f64,
    /// Number of training steps
    pub num_steps: usize,
    /// Batch size per step
    pub batch_size: usize,
    /// Log progress every N steps
    pub log_interval: usize,
    /// Checkpoint every N steps
    pub checkpoint_interval: usize,
    /// Re-initialize dead features every N steps (0 = never)
    pub dead_feature_resample_interval: usize,
    /// JumpReLU initial threshold
    pub jump_threshold: f64,
    /// Checkpoint directory
    pub checkpoint_dir: PathBuf,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            num_features: 131_072,  // 128K — Gemma Scope standard
            model_dim: 0,          // auto-detected
            l1_coefficient: 5e-3,
            learning_rate: 3e-4,
            weight_decay: 0.0,
            num_steps: 100_000,
            batch_size: 4096,
            log_interval: 1000,
            checkpoint_interval: 5000,
            dead_feature_resample_interval: 25_000,
            jump_threshold: 0.001,
            checkpoint_dir: PathBuf::from(""),
        }
    }
}

/// Training stats for one step.
#[derive(Debug, Clone, Default)]
pub struct TrainStats {
    pub step: usize,
    pub reconstruction_loss: f64,
    pub l1_loss: f64,
    pub total_loss: f64,
    pub active_features: usize,
    pub dead_features: usize,
    /// Fraction of features that fired in this batch
    pub feature_density: f64,
}

/// Metal-accelerated SAE trainer using Candle.
pub struct SaeTrainer {
    /// Candle variable map holding all trainable parameters
    pub(crate) var_map: VarMap,
    /// Compute device (Metal GPU)
    pub(crate) device: Device,
    /// Training configuration
    pub config: TrainConfig,
    /// AdamW optimizer state (per-parameter moments)
    pub(crate) adam_m: HashMap<String, Tensor>,
    pub(crate) adam_v: HashMap<String, Tensor>,
    pub(crate) adam_step: usize,
    /// Feature usage counters for dead feature detection
    pub(crate) feature_usage: Vec<u64>,
    /// Current training step
    pub current_step: usize,
}

impl SaeTrainer {
    /// Initialize a new SAE trainer on Metal.
    ///
    /// Weights are initialized following Anthropic's Scaling Monosemanticity:
    /// - W_enc: Xavier uniform
    /// - W_dec columns: unit-norm random vectors
    /// - b_enc: zeros
    /// - b_dec: zeros
    pub fn new(config: TrainConfig) -> Result<Self> {
        // Use Metal GPU on macOS, fall back to CPU on Linux/Windows
        #[cfg(target_os = "macos")]
        let device = Device::new_metal(0)
            .context("Metal GPU initialization failed — falling back to CPU")
            .unwrap_or(Device::Cpu);

        #[cfg(not(target_os = "macos"))]
        let device = Device::Cpu;

        let device_name = if matches!(device, Device::Cpu) { "CPU" } else { "Metal" };

        tracing::info!(
            num_features = config.num_features,
            model_dim = config.model_dim,
            num_steps = config.num_steps,
            batch_size = config.batch_size,
            l1_coeff = config.l1_coefficient,
            lr = config.learning_rate,
            device = device_name,
            "Initializing SAE trainer"
        );

        let var_map = VarMap::new();
        let nf = config.num_features;
        let md = config.model_dim;
        let cpu = Device::Cpu;

        // Xavier uniform initialization scale
        let scale = (6.0 / (nf + md) as f64).sqrt();

        {
            let mut data = var_map.data().lock().expect("SAE var_map mutex poisoned");

            // W_enc: [num_features, model_dim] — Xavier uniform (init on CPU, train on Metal)
            let w_enc = Var::from_tensor(
                &(Tensor::randn(0.0f32, 1.0, (nf, md), &cpu)? * scale)?.to_device(&device)?
            )?;
            data.insert("W_enc".to_string(), w_enc);

            // b_enc: [num_features] — zeros
            let b_enc = Var::from_tensor(
                &Tensor::zeros(nf, DType::F32, &device)?
            )?;
            data.insert("b_enc".to_string(), b_enc);

            // W_dec: [model_dim, num_features] — unit-norm columns
            let w_dec_raw = Tensor::randn(0.0f32, 1.0, (md, nf), &cpu)?.to_device(&device)?;
            // Normalize each column to unit norm
            let norms = w_dec_raw.sqr()?.sum(0)?.sqrt()?; // [nf]
            let norms_expanded = norms.unsqueeze(0)?.broadcast_as((md, nf))?; // [md, nf]
            let w_dec_normed = w_dec_raw.div(&norms_expanded)?;
            let w_dec = Var::from_tensor(&w_dec_normed)?;
            data.insert("W_dec".to_string(), w_dec);

            // b_dec: [model_dim] — zeros
            let b_dec = Var::from_tensor(
                &Tensor::zeros(md, DType::F32, &device)?
            )?;
            data.insert("b_dec".to_string(), b_dec);
        }

        let feature_usage = vec![0u64; nf];

        Ok(Self {
            var_map,
            device,
            config,
            adam_m: HashMap::new(),
            adam_v: HashMap::new(),
            adam_step: 0,
            feature_usage,
            current_step: 0,
        })
    }

    /// Run one training step on a batch of activation vectors.
    ///
    /// Forward:  h = JumpReLU(W_enc @ x + b_enc)
    ///           x̂ = W_dec @ h + b_dec
    /// Loss:     L = ||x - x̂||² + λ||h||₁
    /// Backward: Candle autograd
    /// Update:   AdamW
    pub fn train_step(&mut self, activations: &[Vec<f32>]) -> Result<TrainStats> {
        let batch_size = activations.len();
        let model_dim = self.config.model_dim;

        // Build batch tensor [batch_size, model_dim]
        let flat: Vec<f32> = activations.iter().flatten().copied().collect();
        let x = Tensor::from_slice(&flat, (batch_size, model_dim), &self.device)?;

        // Get trainable vars
        let vars = self.var_map.data().lock().expect("SAE var_map mutex poisoned");
        let w_enc = vars.get("W_enc").context("Missing W_enc")?.as_tensor();
        let b_enc = vars.get("b_enc").context("Missing b_enc")?.as_tensor();
        let w_dec = vars.get("W_dec").context("Missing W_dec")?.as_tensor();
        let b_dec = vars.get("b_dec").context("Missing b_dec")?.as_tensor();

        // Forward: pre_act = x @ W_enc^T + b_enc  [batch, num_features]
        let pre_act = x.matmul(&w_enc.t()?)?.broadcast_add(b_enc)?;

        // JumpReLU: h = max(0, pre_act) * (pre_act > threshold)
        let threshold = self.config.jump_threshold as f32;
        let h = pre_act.relu()?;
        let mask = pre_act.ge(threshold)?.to_dtype(DType::F32)?;
        let h = h.mul(&mask)?;

        // Track feature usage (which features fired)
        let h_sum = h.sum(0)?; // [num_features]
        let h_sum_cpu = h_sum.to_vec1::<f32>()?;
        for (i, &val) in h_sum_cpu.iter().enumerate() {
            if val > 0.0 && i < self.feature_usage.len() {
                self.feature_usage[i] += 1;
            }
        }

        // Reconstruction: x̂ = h @ W_dec^T + b_dec  [batch, model_dim]
        let x_hat = h.matmul(&w_dec.t()?)?.broadcast_add(b_dec)?;

        // Reconstruction loss: ||x - x̂||² / batch_size (stay in f32 for Metal)
        let residual = (&x - &x_hat)?;
        let recon_loss = residual.sqr()?.sum_all()?
            .affine(1.0 / batch_size as f64, 0.0)?;

        // L1 sparsity loss: λ * Σ|h| / batch_size
        let l1_loss = h.abs()?.sum_all()?
            .affine(self.config.l1_coefficient / batch_size as f64, 0.0)?;

        // Total loss
        let total_loss = (&recon_loss + &l1_loss)?;

        // Extract scalar values before backward
        let recon_val = recon_loss.to_scalar::<f32>()? as f64;
        let l1_val = l1_loss.to_scalar::<f32>()? as f64;
        let total_val = total_loss.to_scalar::<f32>()? as f64;

        // Backward pass — compute gradients
        let grads = total_loss.backward()?;

        // Drop vars lock before AdamW update
        drop(vars);

        // AdamW update
        self.adamw_step(&grads)?;

        // Normalize W_dec columns to unit norm after update
        self.normalize_decoder()?;

        // Stats
        let active = self.feature_usage.iter().filter(|&&c| c > 0).count();
        let dead = self.config.num_features - active;
        let density = active as f64 / self.config.num_features as f64;

        self.current_step += 1;

        Ok(TrainStats {
            step: self.current_step,
            reconstruction_loss: recon_val,
            l1_loss: l1_val,
            total_loss: total_val,
            active_features: active,
            dead_features: dead,
            feature_density: density,
        })
    }

    /// AdamW update step for all trainable variables.
    fn adamw_step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        self.adam_step += 1;
        let beta1 = 0.9;
        let beta2 = 0.999;
        let epsilon = 1e-8;
        let lr = self.config.learning_rate;
        let wd = self.config.weight_decay;

        let vars = self.var_map.data().lock().expect("SAE var_map mutex poisoned");
        for (name, var) in vars.iter() {
            let tensor = var.as_tensor();
            let grad = match grads.get(tensor) {
                Some(g) => g.to_dtype(DType::F32)?,
                None => continue,
            };

            let param = tensor.to_dtype(DType::F32)?;

            // Weight decay
            let grad = if wd > 0.0 {
                (&grad + (&param * wd)?)?
            } else {
                grad
            };

            // First moment
            let m = self.adam_m
                .entry(name.clone())
                .or_insert_with(|| Tensor::zeros_like(&grad).expect("m init"));
            *m = (m.affine(beta1, 0.0)? + grad.affine(1.0 - beta1, 0.0)?)?;

            // Second moment
            let v = self.adam_v
                .entry(name.clone())
                .or_insert_with(|| Tensor::zeros_like(&grad).expect("v init"));
            *v = (v.affine(beta2, 0.0)? + grad.sqr()?.affine(1.0 - beta2, 0.0)?)?;

            // Bias correction
            let m_hat = m.affine(1.0 / (1.0 - beta1.powi(self.adam_step as i32)), 0.0)?;
            let v_hat = v.affine(1.0 / (1.0 - beta2.powi(self.adam_step as i32)), 0.0)?;

            // Update
            let update = (&m_hat / &(v_hat.sqrt()? + epsilon)?)?;
            let new_param = (&param - update.affine(lr, 0.0)?)?;
            var.set(&new_param.to_dtype(tensor.dtype())?)?;
        }

        Ok(())
    }

    /// Normalize W_dec columns to unit norm (prevents decoder column drift).
    fn normalize_decoder(&self) -> Result<()> {
        let vars = self.var_map.data().lock().expect("SAE var_map mutex poisoned");
        let w_dec_var = vars.get("W_dec").context("Missing W_dec")?;
        let w_dec = w_dec_var.as_tensor();

        // w_dec shape: [model_dim, num_features]
        let norms = w_dec.sqr()?.sum(0)?.sqrt()?; // [num_features]
        let norms_clamped = norms.clamp(1e-8, f64::INFINITY)?;
        let md = self.config.model_dim;
        let nf = self.config.num_features;
        let norms_expanded = norms_clamped.unsqueeze(0)?.broadcast_as((md, nf))?;
        let normed = w_dec.div(&norms_expanded)?;
        w_dec_var.set(&normed)?;

        Ok(())
    }

    /// Re-initialize dead features (features that never fire).
    ///
    /// Dead features get their encoder direction set to a random activation
    /// from the batch, and decoder column re-normalized. Resets usage counters.
    pub fn resample_dead_features(&mut self, activations: &[Vec<f32>]) -> Result<usize> {
        let dead_indices: Vec<usize> = self.feature_usage
            .iter()
            .enumerate()
            .filter(|(_, &count)| count == 0)
            .map(|(i, _)| i)
            .collect();

        if dead_indices.is_empty() {
            return Ok(0);
        }

        let num_dead = dead_indices.len();
        tracing::info!(
            dead_count = num_dead,
            total_features = self.config.num_features,
            "Resampling dead features"
        );

        let md = self.config.model_dim;
        let vars = self.var_map.data().lock().expect("SAE var_map mutex poisoned");
        let w_enc_var = vars.get("W_enc").context("Missing W_enc")?;
        let b_enc_var = vars.get("b_enc").context("Missing b_enc")?;
        let w_dec_var = vars.get("W_dec").context("Missing W_dec")?;

        let mut w_enc_data = w_enc_var.as_tensor().to_vec2::<f32>()?;
        let mut b_enc_data = b_enc_var.as_tensor().to_vec1::<f32>()?;
        let mut w_dec_data = w_dec_var.as_tensor().to_vec2::<f32>()?;

        for (resample_idx, &feat_idx) in dead_indices.iter().enumerate() {
            // Pick a random activation as the new encoder direction
            let act_idx = resample_idx % activations.len();
            let activation = &activations[act_idx];

            // Set encoder row to normalized activation
            let norm: f32 = activation.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-8 {
                for j in 0..md {
                    w_enc_data[feat_idx][j] = activation[j] / norm * 0.1;
                }
            }

            // Reset encoder bias
            b_enc_data[feat_idx] = 0.0;

            // Reset decoder column to match encoder direction
            for j in 0..md {
                w_dec_data[j][feat_idx] = w_enc_data[feat_idx][j];
            }
        }

        // Write back
        w_enc_var.set(&Tensor::new(w_enc_data, &self.device)?)?;
        b_enc_var.set(&Tensor::new(b_enc_data, &self.device)?)?;
        w_dec_var.set(&Tensor::new(w_dec_data, &self.device)?)?;

        // Reset counters
        self.feature_usage = vec![0u64; self.config.num_features];

        // Reset Adam state for resampled features
        self.adam_m.clear();
        self.adam_v.clear();

        Ok(num_dead)
    }

    // Checkpoint, load, export, save_safetensors → trainer_persist.rs
}

include!("trainer_tests.rs");
