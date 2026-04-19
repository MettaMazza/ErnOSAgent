// Ern-OS — DPO loss (Direct Preference Optimization)

/// DPO loss: -log(σ(β * (log(π/π_ref)(chosen) - log(π/π_ref)(rejected))))
pub fn dpo_loss(
    chosen_log_ratio: f32,
    rejected_log_ratio: f32,
    beta: f32,
) -> f32 {
    let diff = beta * (chosen_log_ratio - rejected_log_ratio);
    -log_sigmoid(diff)
}

fn log_sigmoid(x: f32) -> f32 {
    if x >= 0.0 { -(1.0 + (-x).exp()).ln() }
    else { x - (1.0 + x.exp()).ln() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_correct_preference() {
        let loss = dpo_loss(0.5, -0.5, 0.1);
        assert!(loss >= 0.0);
        assert!(loss < 1.0);
    }

    #[test]
    fn test_wrong_preference_higher_loss() {
        let correct = dpo_loss(0.5, -0.5, 0.1);
        let wrong = dpo_loss(-0.5, 0.5, 0.1);
        assert!(wrong > correct);
    }
}
