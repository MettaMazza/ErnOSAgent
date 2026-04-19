//! SAE persistence — checkpoint save/load, export, and safetensors output.

use crate::interpretability::sae::{SaeArchitecture, SparseAutoencoder};
use crate::interpretability::trainer::SaeTrainer;
use anyhow::{Context, Result};
use candle_core::Tensor;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

impl SaeTrainer {
    /// Save a training checkpoint (weights + optimizer state).
    pub fn checkpoint(&self) -> Result<PathBuf> {
        let dir = &self.config.checkpoint_dir;
        std::fs::create_dir_all(dir)?;
        let path = dir.join(format!("sae_step_{:06}.safetensors", self.current_step));

        let vars = self.var_map.data().lock().expect("SAE var_map mutex poisoned");
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for (name, var) in vars.iter() {
            tensors.insert(name.clone(), var.as_tensor().clone());
        }
        drop(vars);

        candle_core::safetensors::save(&tensors, &path)?;
        tracing::info!(step = self.current_step, path = %path.display(), "SAE checkpoint saved");
        Ok(path)
    }

    /// Load from a checkpoint to resume training.
    pub fn load_checkpoint(&mut self, path: &Path) -> Result<()> {
        let tensors = candle_core::safetensors::load(path, &self.device)?;
        let vars = self.var_map.data().lock().expect("SAE var_map mutex poisoned");
        for (name, var) in vars.iter() {
            if let Some(loaded) = tensors.get(name) { var.set(loaded)?; }
        }

        if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
            if let Some(step_str) = stem.strip_prefix("sae_step_") {
                if let Ok(step) = step_str.parse::<usize>() {
                    self.current_step = step;
                    self.adam_step = step;
                }
            }
        }

        tracing::info!(step = self.current_step, path = %path.display(), "SAE checkpoint loaded");
        Ok(())
    }

    /// Export trained weights to an inference-ready SparseAutoencoder.
    pub fn export_sae(&self) -> Result<SparseAutoencoder> {
        let vars = self.var_map.data().lock().expect("SAE var_map mutex poisoned");
        let w_enc = vars.get("W_enc").context("Missing W_enc")?
            .as_tensor().flatten_all()?.to_vec1::<f32>()?;
        let b_enc = vars.get("b_enc").context("Missing b_enc")?
            .as_tensor().to_vec1::<f32>()?;
        let w_dec = vars.get("W_dec").context("Missing W_dec")?
            .as_tensor().flatten_all()?.to_vec1::<f32>()?;
        let b_dec = vars.get("b_dec").context("Missing b_dec")?
            .as_tensor().to_vec1::<f32>()?;

        Ok(SparseAutoencoder::new(
            w_enc, b_enc, w_dec, b_dec,
            self.config.num_features, self.config.model_dim,
            SaeArchitecture::JumpReLU { threshold: self.config.jump_threshold as f32 },
        ))
    }

    /// Save final trained SAE as SAELens-compatible safetensors.
    pub fn save_safetensors(&self, path: &Path) -> Result<()> {
        std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))?;

        let vars = self.var_map.data().lock().expect("SAE var_map mutex poisoned");
        let mut tensors: HashMap<String, Tensor> = HashMap::new();
        for (name, var) in vars.iter() {
            tensors.insert(name.clone(), var.as_tensor().clone());
        }
        drop(vars);

        candle_core::safetensors::save(&tensors, path)?;
        tracing::info!(
            num_features = self.config.num_features, model_dim = self.config.model_dim,
            steps_trained = self.current_step, path = %path.display(),
            "SAE weights saved to safetensors"
        );
        Ok(())
    }
}

/// Estimate training time for the given configuration.
pub fn estimate_training_time(config: &super::trainer::TrainConfig) -> std::time::Duration {
    let ms_per_step = if config.num_features <= 32768 { 0.5 }
        else if config.num_features <= 131072 { 5.0 }
        else { 20.0 };
    std::time::Duration::from_secs_f64(ms_per_step * config.num_steps as f64 / 1000.0)
}
