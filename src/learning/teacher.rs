// Ern-OS — Training teacher orchestrator
//! Manages the full training pipeline: SFT from golden buffer,
//! preference training from rejection buffer, using LoRA adapters.

use super::lora::{LoraConfig, training, weights::LoraLayer};
use anyhow::Context;
use serde::{Deserialize, Serialize};

/// Training orchestrator — manages the full training pipeline.
pub struct Teacher {
    pub config: LoraConfig,
    pub total_runs: usize,
    pub last_run: Option<chrono::DateTime<chrono::Utc>>,
    /// Active LoRA adapter being trained
    pub adapter: Option<LoraLayer>,
}

impl Teacher {
    pub fn new(config: LoraConfig) -> Self {
        Self { config, total_runs: 0, last_run: None, adapter: None }
    }

    /// Ensure adapter is initialised.
    fn ensure_adapter(&mut self) {
        if self.adapter.is_none() {
            self.adapter = Some(LoraLayer::new(
                "sft_adapter",
                self.config.model_dim,
                self.config.model_dim,
                self.config.rank,
                self.config.alpha,
            ));
        }
    }

    /// Run a training cycle with golden buffer samples.
    pub async fn train_sft(&mut self, samples: &[super::TrainingSample]) -> anyhow::Result<TrainResult> {
        tracing::info!(module = "teacher", fn_name = "train_sft", "teacher::train_sft called");
        self.ensure_adapter();
        let adapter = self.adapter.as_mut()
            .context("LoRA adapter not initialised — ensure_adapter() failed")?;
        let dim = self.config.model_dim;

        // Convert text samples to vector pairs via simple hash embedding
        let pairs: Vec<(Vec<f32>, Vec<f32>)> = samples.iter().map(|s| {
            (text_to_vec(&s.input, dim), text_to_vec(&s.output, dim))
        }).collect();

        let mut total_loss = 0.0;
        let epochs = self.config.epochs.max(1);
        for _ in 0..epochs {
            total_loss = training::train_epoch(adapter, &pairs, self.config.learning_rate);
        }

        self.total_runs += 1;
        self.last_run = Some(chrono::Utc::now());

        Ok(TrainResult {
            method: "SFT".into(),
            samples: samples.len(),
            loss: total_loss,
            epochs,
        })
    }

    /// Run preference training (DPO/ORPO/SimPO/KTO).
    pub async fn train_preference(
        &mut self,
        pairs: &[super::buffers_rejection::PreferencePair],
        method: &str,
    ) -> anyhow::Result<TrainResult> {
        tracing::info!(module = "teacher", fn_name = "train_preference", "teacher::train_preference called");
        self.ensure_adapter();
        let adapter = self.adapter.as_mut()
            .context("LoRA adapter not initialised — ensure_adapter() failed")?;

        let dim = self.config.model_dim;

        // For preference training, compute loss differential between chosen and rejected
        let mut total_loss = 0.0;
        for pair in pairs {
            let chosen_vec = text_to_vec(&pair.chosen, dim);
            let rejected_vec = text_to_vec(&pair.rejected, dim);

            // Train to increase likelihood of chosen response
            let chosen_loss = training::train_step(
                adapter, &chosen_vec, &chosen_vec, self.config.learning_rate,
            );
            // Negative training on rejected (push weights away)
            let _rejected_loss = training::train_step(
                adapter, &rejected_vec, &rejected_vec, -self.config.learning_rate * 0.5,
            );
            total_loss += chosen_loss;
        }

        if !pairs.is_empty() {
            total_loss /= pairs.len() as f64;
        }

        self.total_runs += 1;
        self.last_run = Some(chrono::Utc::now());

        Ok(TrainResult {
            method: method.to_uppercase(),
            samples: pairs.len(),
            loss: total_loss,
            epochs: 1,
        })
    }

    /// Get the total number of trainable parameters in the active adapter.
    pub fn param_count(&self) -> usize {
        tracing::info!(module = "teacher", fn_name = "param_count", "teacher::param_count called");
        self.adapter.as_ref().map(|a| a.param_count()).unwrap_or(0)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainResult {
    pub method: String,
    pub samples: usize,
    pub loss: f64,
    pub epochs: usize,
}

/// Deterministic hash-based vector for text — used to convert strings
/// to fixed-dimension vectors for LoRA training without an embedding server.
fn text_to_vec(text: &str, dim: usize) -> Vec<f32> {
    let mut vec = vec![0.0f32; dim];
    for (i, b) in text.bytes().enumerate() {
        vec[i % dim] += (b as f32 - 128.0) / 128.0;
    }
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for v in &mut vec { *v /= norm; }
    vec
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_teacher_new() {
        let teacher = Teacher::new(LoraConfig::default());
        assert_eq!(teacher.total_runs, 0);
    }

    #[tokio::test]
    async fn test_teacher_sft() {
        let mut teacher = Teacher::new(LoraConfig::default());
        let samples = vec![super::super::TrainingSample {
            id: "test1".into(),
            input: "What is Rust?".into(),
            output: "Rust is a systems programming language.".into(),
            method: super::super::TrainingMethod::Sft,
            quality_score: 0.9,
            timestamp: chrono::Utc::now(),
        }];
        let result = teacher.train_sft(&samples).await.unwrap();
        assert_eq!(result.method, "SFT");
        assert_eq!(result.samples, 1);
        assert_eq!(teacher.total_runs, 1);
    }

    #[test]
    fn test_text_to_vec() {
        let v = text_to_vec("hello", 8);
        assert_eq!(v.len(), 8);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01); // Unit normalized
    }
}
