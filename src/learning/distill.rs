// Ern-OS — Knowledge distillation

use serde::{Deserialize, Serialize};

/// Distillation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillConfig {
    pub teacher_model: String,
    pub student_model: String,
    pub temperature: f32,
    pub alpha: f32,
}

impl Default for DistillConfig {
    fn default() -> Self {
        Self {
            teacher_model: "gemma-4-31B".to_string(),
            student_model: "gemma-4-12B".to_string(),
            temperature: 2.0,
            alpha: 0.5,
        }
    }
}

/// Compute soft-target cross-entropy loss for distillation.
pub fn distillation_loss(
    teacher_logits: &[f32],
    student_logits: &[f32],
    temperature: f32,
) -> f32 {
        tracing::info!(module = "distill", fn_name = "distillation_loss", "distill::distillation_loss called");
    if teacher_logits.len() != student_logits.len() || teacher_logits.is_empty() {
        return 0.0;
    }

    let t_softmax = softmax_temperature(teacher_logits, temperature);
    let s_log_softmax = log_softmax_temperature(student_logits, temperature);

    let loss: f32 = t_softmax.iter().zip(s_log_softmax.iter())
        .map(|(p, log_q)| -p * log_q)
        .sum();

    loss * temperature * temperature
}

fn softmax_temperature(logits: &[f32], temp: f32) -> Vec<f32> {
    let scaled: Vec<f32> = logits.iter().map(|x| x / temp).collect();
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn log_softmax_temperature(logits: &[f32], temp: f32) -> Vec<f32> {
    let scaled: Vec<f32> = logits.iter().map(|x| x / temp).collect();
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|x| (x - max).exp()).collect();
    let log_sum: f32 = exps.iter().sum::<f32>().ln();
    scaled.iter().map(|x| x - max - log_sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax_temperature(&logits, 1.0);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_distillation_loss() {
        let teacher = vec![1.0, 2.0, 3.0];
        let student = vec![1.0, 2.0, 3.0];
        let loss = distillation_loss(&teacher, &student, 1.0);
        // Same logits should give minimal loss
        assert!(loss > 0.0); // Cross-entropy is always positive
    }
}
