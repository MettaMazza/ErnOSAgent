// Ern-OS — KTO loss (Kahneman-Tversky Optimization)

/// KTO loss — unpaired preference optimization.
/// Uses prospect theory: losses are weighted more heavily than gains.
pub fn kto_loss(
    log_prob: f32,
    ref_log_prob: f32,
    is_desirable: bool,
    beta: f32,
    lambda_d: f32,
    lambda_u: f32,
) -> f32 {
    let kl = ref_log_prob - log_prob;
    if is_desirable {
        lambda_d * (1.0 - sigmoid(beta * (log_prob - ref_log_prob - kl)))
    } else {
        lambda_u * (1.0 - sigmoid(beta * (ref_log_prob - log_prob - kl)))
    }
}

fn sigmoid(x: f32) -> f32 { 1.0 / (1.0 + (-x).exp()) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_desirable_lower_loss() {
        let loss = kto_loss(-1.0, -1.5, true, 0.1, 1.0, 1.0);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_undesirable() {
        let loss = kto_loss(-3.0, -1.0, false, 0.1, 1.0, 1.0);
        assert!(loss >= 0.0);
    }
}
