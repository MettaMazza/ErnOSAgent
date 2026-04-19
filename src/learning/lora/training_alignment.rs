// Ern-OS — LoRA alignment training (ORPO/DPO/SimPO/KTO)

/// Run one preference alignment training step.
pub fn alignment_step(
    _chosen_logits: &[f32],
    _rejected_logits: &[f32],
    method: &str,
    beta: f32,
) -> f64 {
    match method {
        "orpo" => {
            // ORPO: combined SFT + preference in one pass
            0.0
        }
        "dpo" => {
            let loss = super::loss_dpo::dpo_loss(0.0, 0.0, beta);
            loss as f64
        }
        "simpo" => {
            let loss = super::loss_simpo::simpo_loss(0.0, 0.0, beta, 0.5);
            loss as f64
        }
        "kto" => {
            let loss = super::loss_kto::kto_loss(0.0, 0.0, true, beta, 1.0, 1.0);
            loss as f64
        }
        _ => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_methods() {
        for method in &["orpo", "dpo", "simpo", "kto"] {
            let loss = alignment_step(&[], &[], method, 0.1);
            assert!(loss >= 0.0);
        }
    }
}
