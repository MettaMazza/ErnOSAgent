// Ern-OS — SAE → Steering bridge
//! Maps SAE feature activations to steering vector adjustments.

use super::FeatureActivation;
use crate::steering::SteeringVector;

/// A steering rule — maps a feature to a vector adjustment.
pub struct SteeringRule {
    pub feature_index: usize,
    pub vector_name: String,
    pub threshold: f32,
    pub strength_scale: f32,
}

/// Evaluate steering rules against current activations.
pub fn evaluate_rules(
    activations: &[FeatureActivation],
    rules: &[SteeringRule],
    vectors: &mut [SteeringVector],
) {
    for rule in rules {
        if let Some(fa) = activations.iter().find(|a| a.feature_index == rule.feature_index) {
            if fa.activation > rule.threshold {
                if let Some(v) = vectors.iter_mut().find(|v| v.name == rule.vector_name) {
                    v.strength = (fa.activation * rule.strength_scale).min(2.0);
                    v.active = true;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_rules() {
        let activations = vec![FeatureActivation {
            feature_index: 41, label: "Curiosity Drive".into(),
            activation: 3.0, baseline: 1.0, delta: 2.0,
        }];
        let rules = vec![SteeringRule {
            feature_index: 41, vector_name: "curiosity".into(),
            threshold: 2.0, strength_scale: 0.5,
        }];
        let mut vectors = vec![SteeringVector {
            name: "curiosity".into(), path: "v.gguf".into(),
            strength: 0.0, active: false, description: String::new(),
        }];

        evaluate_rules(&activations, &rules, &mut vectors);
        assert!(vectors[0].active);
        assert!((vectors[0].strength - 1.5).abs() < f32::EPSILON);
    }
}
