// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Steering Bridge — convert SAE feature directions into control vectors.
//!
//! Each SAE decoder column is a direction vector in residual stream space.
//! This module extracts those columns and generates GGUF-compatible control
//! vector files that can be loaded by llama.cpp's `--control-vector` flag.

use crate::interpretability::features::FeatureDictionary;
use crate::interpretability::sae::SparseAutoencoder;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A feature that can be steered via its SAE decoder direction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SteerableFeature {
    pub index: usize,
    pub name: String,
    pub category: String,
    pub is_safety: bool,
    /// Current steering scale (0.0 = neutral, positive = amplify, negative = suppress)
    pub scale: f64,
    /// Whether this feature has an active control vector loaded
    pub active: bool,
}

/// Active feature steering state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSteeringState {
    /// Currently steered features
    pub active_features: Vec<SteerableFeature>,
    /// Directory where generated GGUF vectors are stored
    pub vectors_dir: PathBuf,
}

impl Default for FeatureSteeringState {
    fn default() -> Self {
        Self {
            active_features: Vec::new(),
            vectors_dir: PathBuf::from(""),
        }
    }
}

impl FeatureSteeringState {
    pub fn new(vectors_dir: PathBuf) -> Self {
        Self {
            active_features: Vec::new(),
            vectors_dir,
        }
    }

    /// List all features available for steering.
    pub fn list_steerable(dictionary: &FeatureDictionary) -> Vec<SteerableFeature> {
        let features: Vec<SteerableFeature> = dictionary
            .labels
            .iter()
            .map(|(&index, label)| {
                let category = match &label.category {
                    crate::interpretability::features::FeatureCategory::Cognitive =>
                        "cognitive".to_string(),
                    crate::interpretability::features::FeatureCategory::Safety(st) =>
                        format!("safety:{:?}", st).to_lowercase(),
                    crate::interpretability::features::FeatureCategory::Linguistic =>
                        "linguistic".to_string(),
                    crate::interpretability::features::FeatureCategory::Semantic =>
                        "semantic".to_string(),
                    crate::interpretability::features::FeatureCategory::Meta =>
                        "meta".to_string(),
                    crate::interpretability::features::FeatureCategory::Emotion(v) =>
                        format!("emotion:{:?}", v).to_lowercase(),
                    crate::interpretability::features::FeatureCategory::Unknown =>
                        "unknown".to_string(),
                };

                SteerableFeature {
                    index,
                    name: label.name.clone(),
                    category,
                    is_safety: dictionary.is_safety_feature(index),
                    scale: 0.0,
                    active: false,
                }
            })
            .collect();

        let safety_count = features.iter().filter(|f| f.is_safety).count();
        tracing::info!(
            total = features.len(),
            safety = safety_count,
            cognitive = features.iter().filter(|f| f.category == "cognitive").count(),
            "Listed steerable features"
        );

        features
    }

    /// Extract a feature direction from the SAE decoder as a raw float vector.
    ///
    /// This is the foundation for control vector generation — each decoded
    /// column represents the "direction" of a monosemantic feature in residual
    /// stream space.
    pub fn extract_direction(
        sae: &SparseAutoencoder,
        feature_index: usize,
    ) -> Vec<f32> {
        let direction = sae.decode_feature(feature_index);
        let l2_norm: f32 = direction.iter().map(|x| x * x).sum::<f32>().sqrt();
        tracing::info!(
            feature_index = feature_index,
            dim = direction.len(),
            l2_norm = format!("{:.4}", l2_norm),
            "Extracted SAE decoder direction vector"
        );
        direction
    }

    /// Save a feature direction as a raw binary file (for future GGUF conversion).
    ///
    /// The raw direction file can be converted to GGUF using the llama.cpp
    /// `convert-control-vector.py` script, or via the `gguf-rs` crate.
    pub fn save_direction_raw(
        direction: &[f32],
        output_dir: &Path,
        name: &str,
    ) -> Result<PathBuf> {
        std::fs::create_dir_all(output_dir)
            .with_context(|| format!("Failed to create vectors dir: {}", output_dir.display()))?;

        let filename = format!("sae_feature_{}.bin", name);
        let path = output_dir.join(&filename);

        let bytes: Vec<u8> = direction
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();

        std::fs::write(&path, &bytes)
            .with_context(|| format!("Failed to write direction: {}", path.display()))?;

        tracing::info!(
            name = name,
            dim = direction.len(),
            path = %path.display(),
            "Saved SAE feature direction"
        );

        Ok(path)
    }

    /// Apply a feature steering change.
    pub fn set_feature(&mut self, index: usize, name: String, category: String, scale: f64) {
        let action = if scale == 0.0 { "remove" } else if scale > 0.0 { "amplify" } else { "suppress" };
        tracing::info!(
            feature_index = index,
            feature_name = name.as_str(),
            category = category.as_str(),
            scale = scale,
            action = action,
            active_before = self.active_features.len(),
            "Feature steering change"
        );

        if let Some(existing) = self.active_features.iter_mut().find(|f| f.index == index) {
            if scale == 0.0 {
                existing.active = false;
                existing.scale = 0.0;
            } else {
                existing.scale = scale;
                existing.active = true;
            }
        } else if scale != 0.0 {
            self.active_features.push(SteerableFeature {
                index,
                name,
                category,
                is_safety: false,
                scale,
                active: true,
            });
        }

        // Clean up inactive
        self.active_features.retain(|f| f.active);

        tracing::info!(
            active_after = self.active_features.len(),
            summary = self.summary().as_str(),
            "Feature steering state updated"
        );
    }

    /// Clear all feature steering.
    pub fn clear(&mut self) {
        let cleared = self.active_features.len();
        self.active_features.clear();
        tracing::info!(
            cleared_count = cleared,
            "All feature steering cleared"
        );
    }

    /// Get a summary of active feature steering.
    pub fn summary(&self) -> String {
        if self.active_features.is_empty() {
            "No feature steering active".to_string()
        } else {
            self.active_features
                .iter()
                .map(|f| {
                    let dir = if f.scale > 0.0 { "↑" } else { "↓" };
                    format!("{}{} {:.1}", dir, f.name, f.scale.abs())
                })
                .collect::<Vec<_>>()
                .join(", ")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_steerable() {
        let dict = FeatureDictionary::demo();
        let features = FeatureSteeringState::list_steerable(&dict);
        assert!(!features.is_empty());
        assert!(features.len() >= 20);
        assert!(features.iter().any(|f| f.name == "Reasoning Chain"));
    }

    #[test]
    fn test_set_and_clear() {
        let mut state = FeatureSteeringState::default();
        state.set_feature(0, "Reasoning".into(), "cognitive".into(), 1.5);
        assert_eq!(state.active_features.len(), 1);
        assert_eq!(state.active_features[0].scale, 1.5);

        state.set_feature(0, "Reasoning".into(), "cognitive".into(), 0.0);
        assert!(state.active_features.is_empty());
    }

    #[test]
    fn test_summary() {
        let mut state = FeatureSteeringState::default();
        assert_eq!(state.summary(), "No feature steering active");

        state.set_feature(0, "Reasoning".into(), "cognitive".into(), 2.0);
        state.set_feature(7, "Sycophancy".into(), "safety".into(), -1.5);
        let summary = state.summary();
        assert!(summary.contains("↑Reasoning 2.0"));
        assert!(summary.contains("↓Sycophancy 1.5"));
    }
}
