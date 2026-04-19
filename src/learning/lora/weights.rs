// Ern-OS — LoRA weight management
//! Low-rank adapter weight pairs (A, B matrices).

use serde::{Deserialize, Serialize};

/// A single LoRA adapter layer — W = W0 + (B @ A) * (alpha/rank)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoraLayer {
    pub name: String,
    pub rank: usize,
    pub alpha: f32,
    /// A matrix: (rank x in_dim) — initialized small random
    pub a_weights: Vec<Vec<f32>>,
    /// B matrix: (out_dim x rank) — initialized zeros
    pub b_weights: Vec<Vec<f32>>,
}

impl LoraLayer {
    pub fn new(name: &str, in_dim: usize, out_dim: usize, rank: usize, alpha: f32) -> Self {
        // Kaiming init for A, zeros for B
        let scale = (2.0 / in_dim as f32).sqrt();
        let a_weights: Vec<Vec<f32>> = (0..rank)
            .map(|i| (0..in_dim).map(|j| {
                // Deterministic pseudo-random for reproducibility
                let seed = (i * in_dim + j) as f32;
                (seed.sin() * 43758.5453).fract() * scale
            }).collect())
            .collect();
        let b_weights = vec![vec![0.0; rank]; out_dim];

        Self { name: name.to_string(), rank, alpha, a_weights, b_weights }
    }

    /// Scaling factor: alpha / rank
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// Compute the LoRA delta: (B @ A @ x) * scaling
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // h = A @ x  (rank-dimensional)
        let mut h = vec![0.0f32; self.rank];
        for (i, row) in self.a_weights.iter().enumerate() {
            h[i] = row.iter().zip(x).map(|(w, xi)| w * xi).sum();
        }

        // out = B @ h  (out_dim-dimensional)
        let scaling = self.scaling();
        self.b_weights.iter().map(|row| {
            let dot: f32 = row.iter().zip(&h).map(|(w, hi)| w * hi).sum();
            dot * scaling
        }).collect()
    }

    pub fn param_count(&self) -> usize {
        self.a_weights.len() * self.a_weights.first().map_or(0, |r| r.len()) +
        self.b_weights.len() * self.b_weights.first().map_or(0, |r| r.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_layer() {
        let layer = LoraLayer::new("q_proj", 256, 256, 16, 32.0);
        assert_eq!(layer.rank, 16);
        assert_eq!(layer.a_weights.len(), 16);
        assert_eq!(layer.b_weights.len(), 256);
    }

    #[test]
    fn test_forward_zeros() {
        let layer = LoraLayer::new("test", 4, 4, 2, 4.0);
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let out = layer.forward(&x);
        assert_eq!(out.len(), 4);
        // B is zeros, so output should be zeros
        assert!(out.iter().all(|v| v.abs() < 1e-6));
    }

    #[test]
    fn test_scaling() {
        let layer = LoraLayer::new("test", 4, 4, 16, 32.0);
        assert!((layer.scaling() - 2.0).abs() < f32::EPSILON);
    }
}
