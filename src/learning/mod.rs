// Ern-OS — Self-learning pipeline module
//! Recursive self-improvement via LoRA training, GRPO, sleep consolidation.

pub mod buffers;
pub mod buffers_rejection;
pub mod observer_buffer;
pub mod lora;
pub mod grpo;
pub mod teacher;
pub mod manifest;
pub mod distill;
pub mod sleep;

use serde::{Deserialize, Serialize};

/// A training sample for the learning pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSample {
    pub id: String,
    pub input: String,
    pub output: String,
    pub method: TrainingMethod,
    pub quality_score: f32,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Supported training methods.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrainingMethod {
    /// Supervised fine-tuning
    Sft,
    /// Odds Ratio Preference Optimization
    Orpo,
    /// Simple Preference Optimization
    SimPO,
    /// Kahneman-Tversky Optimization
    Kto,
    /// Direct Preference Optimization
    Dpo,
    /// Group Relative Policy Optimization
    Grpo,
}

impl TrainingMethod {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Sft => "sft",
            Self::Orpo => "orpo",
            Self::SimPO => "simpo",
            Self::Kto => "kto",
            Self::Dpo => "dpo",
            Self::Grpo => "grpo",
        }
    }
}

/// Training pipeline status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStatus {
    pub golden_buffer_size: usize,
    pub rejection_buffer_size: usize,
    pub preference_pairs: usize,
    pub active_adapters: Vec<String>,
    pub last_training: Option<chrono::DateTime<chrono::Utc>>,
    pub total_training_runs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_method() {
        assert_eq!(TrainingMethod::Orpo.as_str(), "orpo");
        assert_eq!(TrainingMethod::Grpo.as_str(), "grpo");
    }
}
