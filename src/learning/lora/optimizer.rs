// Ern-OS — AdamW optimizer for LoRA training

/// AdamW optimizer state for a single parameter.
pub struct AdamWState {
    pub m: Vec<f32>,  // First moment
    pub v: Vec<f32>,  // Second moment
    pub t: usize,     // Timestep
}

impl AdamWState {
    pub fn new(size: usize) -> Self {
        Self { m: vec![0.0; size], v: vec![0.0; size], t: 0 }
    }

    /// Compute AdamW update.
    pub fn step(
        &mut self,
        params: &mut [f32],
        grads: &[f32],
        lr: f32,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
    ) {
        self.t += 1;
        let bc1 = 1.0 - beta1.powi(self.t as i32);
        let bc2 = 1.0 - beta2.powi(self.t as i32);

        for i in 0..params.len().min(grads.len()) {
            self.m[i] = beta1 * self.m[i] + (1.0 - beta1) * grads[i];
            self.v[i] = beta2 * self.v[i] + (1.0 - beta2) * grads[i] * grads[i];

            let m_hat = self.m[i] / bc1;
            let v_hat = self.v[i] / bc2;

            // AdamW: weight decay applied to params, not grads
            params[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * params[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step() {
        let mut state = AdamWState::new(3);
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];
        state.step(&mut params, &grads, 0.001, 0.9, 0.999, 1e-8, 0.01);
        // Params should have changed
        assert!((params[0] - 1.0).abs() > 1e-6);
    }
}
