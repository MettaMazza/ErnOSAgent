// Ern-OS — EWC (Elastic Weight Consolidation)
//! Prevents catastrophic forgetting by penalizing changes to important weights.

/// EWC regularization term.
pub struct EwcRegularizer {
    /// Fisher information diagonal for each parameter.
    pub fisher: Vec<f32>,
    /// Snapshot of parameters at consolidation time.
    pub theta_star: Vec<f32>,
    /// Regularization strength.
    pub lambda: f32,
}

impl EwcRegularizer {
    pub fn new(fisher: Vec<f32>, theta_star: Vec<f32>, lambda: f32) -> Self {
        Self { fisher, theta_star, lambda }
    }

    /// Compute EWC penalty: (λ/2) * Σ F_i * (θ_i - θ*_i)²
    pub fn penalty(&self, current_params: &[f32]) -> f32 {
        if current_params.len() != self.theta_star.len() { return 0.0; }
        let sum: f32 = self.fisher.iter()
            .zip(current_params)
            .zip(&self.theta_star)
            .map(|((f, theta), theta_star)| {
                f * (theta - theta_star).powi(2)
            })
            .sum();
        (self.lambda / 2.0) * sum
    }

    /// Compute EWC gradient: λ * F_i * (θ_i - θ*_i)
    pub fn gradient(&self, current_params: &[f32]) -> Vec<f32> {
        if current_params.len() != self.theta_star.len() {
            return vec![0.0; current_params.len()];
        }
        self.fisher.iter()
            .zip(current_params)
            .zip(&self.theta_star)
            .map(|((f, theta), theta_star)| {
                self.lambda * f * (theta - theta_star)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_penalty_at_optimum() {
        let ewc = EwcRegularizer::new(vec![1.0, 1.0], vec![0.5, 0.5], 1.0);
        let penalty = ewc.penalty(&[0.5, 0.5]);
        assert!(penalty.abs() < 1e-6);
    }

    #[test]
    fn test_nonzero_penalty() {
        let ewc = EwcRegularizer::new(vec![1.0, 1.0], vec![0.5, 0.5], 1.0);
        let penalty = ewc.penalty(&[1.0, 1.0]);
        assert!(penalty > 0.0);
    }

    #[test]
    fn test_gradient() {
        let ewc = EwcRegularizer::new(vec![1.0], vec![0.5], 1.0);
        let grad = ewc.gradient(&[1.0]);
        assert!((grad[0] - 0.5).abs() < 1e-6);
    }
}
