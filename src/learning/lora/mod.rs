// Ern-OS — LoRA training engine module declarations
pub mod weights;
pub mod forward;
pub mod loss;
pub mod loss_simpo;
pub mod loss_kto;
pub mod loss_dpo;
pub mod training;
pub mod training_alignment;
pub mod optimizer;
pub mod ewc;
pub mod adapters;

use serde::{Deserialize, Serialize};

/// LoRA configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f32,
    pub dropout: f32,
    pub model_dim: usize,
    pub target_modules: Vec<String>,
    pub learning_rate: f64,
    pub epochs: usize,
    pub batch_size: usize,
}

impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 16,
            alpha: 32.0,
            dropout: 0.1,
            model_dim: 64,
            target_modules: vec![
                "q_proj".to_string(),
                "v_proj".to_string(),
            ],
            learning_rate: 2e-5,
            epochs: 3,
            batch_size: 4,
        }
    }
}

/// Training progress report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingReport {
    pub epoch: usize,
    pub total_epochs: usize,
    pub loss: f64,
    pub learning_rate: f64,
    pub samples_processed: usize,
    pub elapsed_seconds: f64,
    pub eta_seconds: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = LoraConfig::default();
        assert_eq!(config.rank, 16);
        assert_eq!(config.target_modules.len(), 2);
    }
}
