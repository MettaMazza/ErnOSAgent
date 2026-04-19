// Ern-OS — Linear probes for concept verification

/// A trained linear probe for a specific concept.
pub struct LinearProbe {
    pub concept: String,
    pub weights: Vec<f32>,
    pub bias: f32,
    pub accuracy: f32,
}

impl LinearProbe {
    /// Predict whether the concept is present in the given activations.
    pub fn predict(&self, activations: &[f32]) -> f32 {
        if self.weights.len() != activations.len() { return 0.0; }
        let dot: f32 = self.weights.iter().zip(activations).map(|(w, a)| w * a).sum();
        sigmoid(dot + self.bias)
    }
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < f32::EPSILON);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_probe_predict() {
        let probe = LinearProbe {
            concept: "test".into(), weights: vec![1.0, 0.0],
            bias: 0.0, accuracy: 0.95,
        };
        let result = probe.predict(&[2.0, 0.0]);
        assert!(result > 0.5);
    }
}
