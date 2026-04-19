// Ern-OS — Mechanistic Interpretability Module
//! SAE-based feature extraction, neural snapshots, and steering bridge.

pub mod sae;
pub mod features;
pub mod snapshot;
pub mod live;
pub mod steering_bridge;
pub mod extractor;
pub mod collector;
pub mod corpus;
pub mod trainer;
pub mod trainer_persist;
pub mod train_runner;
pub mod probe;
pub mod divergence;

use serde::{Deserialize, Serialize};

/// A labeled SAE feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabeledFeature {
    pub index: usize,
    pub label: String,
    pub category: String,
    pub baseline_activation: f32,
}

/// A neural state snapshot — activations at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralSnapshot {
    pub id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub top_features: Vec<(usize, f32)>,
    pub context_summary: String,
    pub divergence_from_baseline: f32,
}

/// Feature activation with label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureActivation {
    pub feature_index: usize,
    pub label: String,
    pub activation: f32,
    pub baseline: f32,
    pub delta: f32,
}

impl FeatureActivation {
    /// Is this feature significantly above baseline?
    pub fn is_elevated(&self) -> bool {
        self.delta > 0.5
    }

    /// Is this feature suppressed below baseline?
    pub fn is_suppressed(&self) -> bool {
        self.delta < -0.5
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_elevated() {
        let fa = FeatureActivation {
            feature_index: 0, label: "test".into(),
            activation: 2.0, baseline: 1.0, delta: 1.0,
        };
        assert!(fa.is_elevated());
        assert!(!fa.is_suppressed());
    }
}
