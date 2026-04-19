// Ern-OS — SimPO loss (Simple Preference Optimization)

/// SimPO loss — reference-free preference optimization.
/// L = -log(σ(β * (log_π(chosen) - log_π(rejected) - γ)))
pub fn simpo_loss(
    chosen_log_prob: f32,
    rejected_log_prob: f32,
    beta: f32,
    gamma: f32,
) -> f32 {
    let diff = beta * (chosen_log_prob - rejected_log_prob - gamma);
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
    fn test_preferred_lower_loss() {
        let loss1 = simpo_loss(-1.0, -3.0, 0.1, 0.0); // chosen much better
        let loss2 = simpo_loss(-2.0, -2.5, 0.1, 0.0); // chosen slightly better
        assert!(loss1 < loss2);
    }
}
