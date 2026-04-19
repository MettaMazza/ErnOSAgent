// Ern-OS — LoRA forward pass through target modules

use super::weights::LoraLayer;

/// Apply LoRA adapter to a base model output.
pub fn apply_lora(base_output: &[f32], lora: &LoraLayer, input: &[f32]) -> Vec<f32> {
    let delta = lora.forward(input);
    base_output.iter().zip(delta.iter())
        .map(|(b, d)| b + d)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_identity() {
        let lora = LoraLayer::new("test", 4, 4, 2, 4.0);
        let base = vec![1.0, 2.0, 3.0, 4.0];
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let result = apply_lora(&base, &lora, &input);
        // B is zeros, so result should equal base
        assert_eq!(result.len(), 4);
        assert!((result[0] - 1.0).abs() < 1e-5);
    }
}
