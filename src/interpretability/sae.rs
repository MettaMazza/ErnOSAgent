// Ern-OS — Sparse Autoencoder — decompose dense activations into sparse interpretable features.
// Ported from ErnOSAgent with full parity.
//
// Supports ReLU, JumpReLU, and TopK architectures.
// Compatible with SAELens/Gemma Scope weight formats (safetensors).
//
// Mathematical basis (from Anthropic's Scaling Monosemanticity):
//   Encoder: f_i(x) = ReLU(W_enc_i · x + b_enc_i)
//   Decoder: x̂ = b_dec + Σ f_i(x) · W_dec_·,i

use serde::{Deserialize, Serialize};

/// SAE activation function architecture.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SaeArchitecture {
    /// Standard ReLU: f(x) = max(0, x)
    ReLU,
    /// JumpReLU: f(x) = x · H(x - θ) where H is Heaviside step (Gemma Scope default)
    JumpReLU { threshold: f32 },
    /// TopK: keep only the K largest activations
    TopK { k: usize },
}

/// A single feature activation: index + strength.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureActivation {
    pub index: usize,
    pub activation: f32,
    pub label: Option<String>,
}

/// Sparse Autoencoder weights and inference.
#[derive(Debug)]
pub struct SparseAutoencoder {
    /// Encoder weights: num_features × model_dim
    pub(crate) w_enc: Vec<f32>,
    /// Encoder bias: num_features
    pub(crate) b_enc: Vec<f32>,
    /// Decoder weights: model_dim × num_features (column-major for fast feature lookup)
    pub(crate) w_dec: Vec<f32>,
    /// Decoder bias: model_dim
    pub(crate) b_dec: Vec<f32>,
    /// Architecture variant
    pub(crate) architecture: SaeArchitecture,
    /// Number of learned features
    pub num_features: usize,
    /// Model dimension (residual stream width)
    pub model_dim: usize,
}

/// Exported SAE weights for serialization.
pub struct SaeWeights {
    pub w_enc: Vec<f32>,
    pub b_enc: Vec<f32>,
    pub w_dec: Vec<f32>,
    pub b_dec: Vec<f32>,
}

impl SparseAutoencoder {
    /// Create a new SAE with the given weights.
    pub fn new(
        w_enc: Vec<f32>,
        b_enc: Vec<f32>,
        w_dec: Vec<f32>,
        b_dec: Vec<f32>,
        num_features: usize,
        model_dim: usize,
        architecture: SaeArchitecture,
    ) -> Self {
        Self {
            w_enc,
            b_enc,
            w_dec,
            b_dec,
            architecture,
            num_features,
            model_dim,
        }
    }

    /// Create a demonstration SAE with random-ish weights for dashboard development.
    pub fn demo(model_dim: usize, num_features: usize) -> Self {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut w_enc = vec![0.0f32; num_features * model_dim];
        let b_enc = vec![-0.5f32; num_features];
        let mut w_dec = vec![0.0f32; model_dim * num_features];
        let b_dec = vec![0.0f32; model_dim];

        for i in 0..w_enc.len() {
            let mut h = DefaultHasher::new();
            i.hash(&mut h);
            let val = (h.finish() % 10000) as f32 / 10000.0 - 0.5;
            w_enc[i] = val * 0.1;
        }
        for i in 0..w_dec.len() {
            let mut h = DefaultHasher::new();
            (i + 999999).hash(&mut h);
            let val = (h.finish() % 10000) as f32 / 10000.0 - 0.5;
            w_dec[i] = val * 0.1;
        }

        Self::new(
            w_enc,
            b_enc,
            w_dec,
            b_dec,
            num_features,
            model_dim,
            SaeArchitecture::ReLU,
        )
    }

    /// Encode activations into sparse feature vector.
    /// Returns the top-k most active features.
    pub fn encode(&self, activations: &[f32], top_k: usize) -> Vec<FeatureActivation> {
        assert_eq!(
            activations.len(),
            self.model_dim,
            "Activation dim mismatch: got {}, expected {}",
            activations.len(),
            self.model_dim,
        );

        let start = std::time::Instant::now();

        // Compute f_i(x) = activation_fn(W_enc_i · x + b_enc_i)
        let mut feature_acts: Vec<(usize, f32)> = Vec::new();

        for i in 0..self.num_features {
            let row_start = i * self.model_dim;
            let mut dot = self.b_enc[i];
            for j in 0..self.model_dim {
                dot += self.w_enc[row_start + j] * activations[j];
            }

            let act = match self.architecture {
                SaeArchitecture::ReLU => dot.max(0.0),
                SaeArchitecture::JumpReLU { threshold } => {
                    if dot > threshold {
                        dot
                    } else {
                        0.0
                    }
                }
                SaeArchitecture::TopK { .. } => dot, // filtering done below
            };

            if act > 0.0 {
                feature_acts.push((i, act));
            }
        }

        // For TopK, keep only the K largest
        if let SaeArchitecture::TopK { k } = self.architecture {
            feature_acts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            feature_acts.truncate(k);
        }

        // Sort by activation strength and take top_k
        feature_acts.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        feature_acts.truncate(top_k);

        let result: Vec<FeatureActivation> = feature_acts
            .into_iter()
            .map(|(index, activation)| FeatureActivation {
                index,
                activation,
                label: None,
            })
            .collect();

        tracing::debug!(
            top_k = top_k,
            returned = result.len(),
            architecture = format!("{:?}", self.architecture),
            top_activation = result.first().map(|f| f.activation).unwrap_or(0.0),
            elapsed_us = start.elapsed().as_micros(),
            "SAE encode complete"
        );

        result
    }

    /// Decode sparse features back to activation space (for steering vectors).
    pub fn decode_feature(&self, feature_index: usize) -> Vec<f32> {
        assert!(feature_index < self.num_features);
        let mut direction = vec![0.0f32; self.model_dim];
        for j in 0..self.model_dim {
            direction[j] = self.w_dec[j * self.num_features + feature_index];
        }
        direction
    }

    /// Export internal weights for serialization (safetensors, etc.).
    pub fn export_weights(&self) -> SaeWeights {
        SaeWeights {
            w_enc: self.w_enc.clone(),
            b_enc: self.b_enc.clone(),
            w_dec: self.w_dec.clone(),
            b_dec: self.b_dec.clone(),
        }
    }

    /// Load SAE weights from a safetensors file (SAELens / Gemma Scope format).
    ///
    /// Expected tensor keys: `W_enc`, `b_enc`, `W_dec`, `b_dec`.
    /// Dimensions are auto-derived from the `W_enc` shape: `[num_features, model_dim]`.
    pub fn load_safetensors(path: &std::path::Path) -> anyhow::Result<Self> {
        use anyhow::Context;

        let data = std::fs::read(path)
            .with_context(|| format!("Failed to read SAE weights: {}", path.display()))?;

        if data.len() < 8 {
            anyhow::bail!("SAE weights file too small");
        }

        let header_bytes: [u8; 8] = data[0..8].try_into()
            .context("Failed to read safetensors header length")?;
        let header_len = u64::from_le_bytes(header_bytes) as usize;
        if data.len() < 8 + header_len {
            anyhow::bail!("SAE weights file truncated");
        }

        let header_str = std::str::from_utf8(&data[8..8 + header_len])
            .context("Invalid UTF-8 in safetensors header")?;
        let header: serde_json::Value = serde_json::from_str(header_str)
            .context("Invalid JSON in safetensors header")?;

        let tensor_start = 8 + header_len;
        let mut w_enc = Vec::new();
        let mut b_enc = Vec::new();
        let mut w_dec = Vec::new();
        let mut b_dec = Vec::new();
        let (mut num_features, mut model_dim) = (0usize, 0usize);

        for (name, info) in header.as_object().unwrap_or(&serde_json::Map::new()) {
            if name == "__metadata__" { continue; }
            let dtype = info["dtype"].as_str().unwrap_or("F32");
            if dtype != "F32" { continue; }

            if let Some(offsets) = info["data_offsets"].as_array() {
                let s = offsets[0].as_u64().unwrap_or(0) as usize + tensor_start;
                let e = offsets[1].as_u64().unwrap_or(0) as usize + tensor_start;
                if e > data.len() { continue; }
                let floats: Vec<f32> = data[s..e].chunks_exact(4)
                    .filter_map(|c| c.try_into().ok().map(|arr: [u8; 4]| f32::from_le_bytes(arr)))
                    .collect();

                match name.as_str() {
                    "W_enc" => {
                        if let Some(shape) = info["shape"].as_array() {
                            if shape.len() == 2 {
                                num_features = shape[0].as_u64().unwrap_or(0) as usize;
                                model_dim = shape[1].as_u64().unwrap_or(0) as usize;
                            }
                        }
                        w_enc = floats;
                    }
                    "b_enc" => b_enc = floats,
                    "W_dec" => w_dec = floats,
                    "b_dec" => b_dec = floats,
                    _ => {}
                }
            }
        }

        if num_features == 0 || model_dim == 0 {
            anyhow::bail!("Could not determine SAE dimensions from safetensors");
        }

        tracing::info!(
            num_features = num_features,
            model_dim = model_dim,
            path = %path.display(),
            "Loaded SAE weights from safetensors"
        );

        Ok(Self::new(
            w_enc,
            b_enc,
            w_dec,
            b_dec,
            num_features,
            model_dim,
            SaeArchitecture::JumpReLU { threshold: 0.001 },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demo_sae_encode() {
        let sae = SparseAutoencoder::demo(64, 256);
        let activations = vec![0.1f32; 64];
        let features = sae.encode(&activations, 10);
        assert!(features.len() <= 10);
        for w in features.windows(2) {
            assert!(w[0].activation >= w[1].activation);
        }
    }

    #[test]
    fn test_decode_feature_dimension() {
        let sae = SparseAutoencoder::demo(64, 256);
        let direction = sae.decode_feature(0);
        assert_eq!(direction.len(), 64);
    }
}
