// Ern-OS — SFT cross-entropy loss

/// Cross-entropy loss for supervised fine-tuning.
pub fn cross_entropy_loss(logits: &[f32], target_idx: usize) -> f32 {
    if logits.is_empty() || target_idx >= logits.len() { return 0.0; }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let log_softmax = (logits[target_idx] - max) - sum.ln();
    -log_softmax
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correct_prediction() {
        let logits = vec![0.0, 0.0, 10.0]; // Strong prediction on idx=2
        let loss = cross_entropy_loss(&logits, 2);
        assert!(loss < 0.1);
    }

    #[test]
    fn test_wrong_prediction() {
        let logits = vec![10.0, 0.0, 0.0]; // Strong prediction on idx=0
        let loss = cross_entropy_loss(&logits, 2); // But target is idx=2
        assert!(loss > 5.0);
    }
}
