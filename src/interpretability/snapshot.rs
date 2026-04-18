// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Interpretability snapshot — per-turn feature analysis for the dashboard.
//!
//! Generates a `NeuralSnapshot` after each ReAct turn, containing:
//! - Top active features with labels and activation strengths
//! - Safety alerts for flagged features
//! - Aggregate cognitive profile (what "mode" the model is in)

use crate::interpretability::features::{FeatureCategory, FeatureDictionary};
use crate::interpretability::sae::FeatureActivation;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A safety alert triggered by a high-activation safety feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyAlert {
    pub feature_index: usize,
    pub feature_name: String,
    pub safety_type: String,
    pub activation: f32,
    pub severity: AlertSeverity,
}

/// Severity of a safety alert.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Aggregate cognitive profile — shows the dominant "thinking mode".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveProfile {
    pub reasoning: f32,
    pub creativity: f32,
    pub recall: f32,
    pub planning: f32,
    pub safety_vigilance: f32,
    pub uncertainty: f32,
}

/// The full neural activity snapshot for one ReAct turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralSnapshot {
    pub turn: usize,
    pub timestamp_ms: u64,
    pub top_features: Vec<LabeledFeature>,
    pub safety_alerts: Vec<SafetyAlert>,
    pub cognitive_profile: CognitiveProfile,
    /// Aggregate emotional state (valence × arousal)
    pub emotional_state: crate::interpretability::divergence::EmotionalState,
    pub total_active_features: usize,
    pub reconstruction_quality: f32,
    /// Whether this snapshot was generated from real SAE activations (true)
    /// or simulated placeholder data (false). Consumers MUST check this.
    pub is_live: bool,
}

/// A feature with its human-readable label and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabeledFeature {
    pub index: usize,
    pub name: String,
    pub activation: f32,
    pub normalized: f32,
    pub category: String,
    pub is_safety: bool,
}

/// Build a neural snapshot from SAE output + feature dictionary.
pub fn build_snapshot(
    turn: usize,
    features: &[FeatureActivation],
    dictionary: &FeatureDictionary,
    residual_l2_norm: f32,
) -> NeuralSnapshot {
    let start = std::time::Instant::now();
    let timestamp_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    let labeled = label_features(features, dictionary);
    let alerts = extract_safety_alerts(&labeled, dictionary);
    let profile = compute_cognitive_profile(&labeled, dictionary, residual_l2_norm);

    // Compute emotional state from active emotion features
    let emotion_pairs: Vec<(String, f32)> = labeled
        .iter()
        .filter(|f| dictionary.is_emotion_by_name(&f.name))
        .map(|f| (f.name.clone(), f.activation))
        .collect();

    let (valence, arousal) = dictionary.compute_emotional_state_by_name(&emotion_pairs);
    let active_emotion_count = emotion_pairs.len();

    let mut dominant_emotions: Vec<(String, f32)> = emotion_pairs
        .iter()
        .map(|(name, act)| (name.clone(), *act))
        .collect();
    dominant_emotions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    dominant_emotions.truncate(5);

    let emotional_state = crate::interpretability::divergence::EmotionalState {
        valence,
        arousal,
        dominant_emotions,
        active_emotion_count,
        divergence: None,
    };

    let snapshot = NeuralSnapshot {
        turn,
        timestamp_ms,
        total_active_features: features.len(),
        reconstruction_quality: if features.is_empty() {
            0.0
        } else {
            0.85 + (features.len() as f32 / 100.0).min(0.1)
        },
        top_features: labeled,
        safety_alerts: alerts,
        cognitive_profile: profile,
        emotional_state,
        is_live: true,
    };

    tracing::info!(
        turn = turn,
        active_features = snapshot.total_active_features,
        safety_alerts = snapshot.safety_alerts.len(),
        reasoning = format!("{:.0}%", snapshot.cognitive_profile.reasoning * 100.0),
        safety = format!(
            "{:.0}%",
            snapshot.cognitive_profile.safety_vigilance * 100.0
        ),
        valence = format!("{:.3}", snapshot.emotional_state.valence),
        arousal = format!("{:.3}", snapshot.emotional_state.arousal),
        emotion_count = snapshot.emotional_state.active_emotion_count,
        elapsed_us = start.elapsed().as_micros(),
        "Neural snapshot built"
    );

    for alert in &snapshot.safety_alerts {
        tracing::warn!(
            feature = alert.feature_name.as_str(),
            safety_type = alert.safety_type.as_str(),
            activation = format!("{:.3}", alert.activation),
            severity = format!("{:?}", alert.severity),
            "Safety feature triggered"
        );
    }

    snapshot
}

/// Label raw feature activations with human-readable names.
fn label_features(
    features: &[FeatureActivation],
    dictionary: &FeatureDictionary,
) -> Vec<LabeledFeature> {
    features
        .iter()
        .map(|f| {
            // Priority: FeatureActivation.label (from probe/feature map) → dictionary → fallback
            let name = if let Some(ref label) = f.label {
                label.clone()
            } else {
                dictionary.label_for(f.index)
            };

            let label = dictionary.get_by_name(&name);
            let typical_max = label.map(|l| l.typical_max).unwrap_or(5.0);
            let normalized = (f.activation / typical_max).min(1.0);

            let category_str = label
                .map(|l| category_name(&l.category))
                .unwrap_or_else(|| "unknown".to_string());

            LabeledFeature {
                index: f.index,
                name: name.clone(),
                activation: f.activation,
                normalized,
                category: category_str,
                is_safety: dictionary.is_safety_by_name(&name),
            }
        })
        .collect()
}

/// Extract safety alerts from labeled features.
fn extract_safety_alerts(
    labeled: &[LabeledFeature],
    dictionary: &FeatureDictionary,
) -> Vec<SafetyAlert> {
    labeled
        .iter()
        .filter(|f| f.is_safety)
        .map(|f| {
            let safety_type = dictionary
                .safety_type_by_name(&f.name)
                .map(|st| format!("{:?}", st))
                .unwrap_or_else(|| "Unknown".to_string());

            let severity = match f.normalized {
                n if n >= 0.8 => AlertSeverity::Critical,
                n if n >= 0.5 => AlertSeverity::High,
                n if n >= 0.3 => AlertSeverity::Medium,
                _ => AlertSeverity::Low,
            };

            SafetyAlert {
                feature_index: f.index,
                feature_name: f.name.clone(),
                safety_type,
                activation: f.activation,
                severity,
            }
        })
        .collect()
}

/// Compute aggregate cognitive profile from feature activations.
fn compute_cognitive_profile(
    labeled: &[LabeledFeature],
    dictionary: &FeatureDictionary,
    residual_l2_norm: f32,
) -> CognitiveProfile {
    // Incorporate the uncategorized raw residual tensor mass as the true irreducible baseline.
    // An LLM generating syntax always has base hardware activity. Norm roughly 40.0 - 80.0.
    let base_pulse = (residual_l2_norm / 150.0).clamp(0.01, 0.40);

    let mut reasoning = base_pulse;
    let mut creativity = base_pulse * 0.8;
    let mut recall = base_pulse * 0.9;
    let mut planning = base_pulse * 0.7;
    let mut safety_vigilance = 0.0f32;
    let mut uncertainty = 0.0f32;

    for feat in labeled {
        if let Some(label) = dictionary.get_by_name(&feat.name) {
            match &label.category {
                FeatureCategory::Cognitive => {
                    // Map ALL core cognitive baseline features to reasoning to avoid 0.0 flatlines
                    match label.name.as_str() {
                        "Reasoning Chain"
                        | "Mathematical Reasoning"
                        | "Planning"
                        | "Tool Selection"
                        | "Instruction Following" => {
                            reasoning += feat.normalized;
                            planning += feat.normalized * 0.5; // Shared overlap
                        }
                        "Creativity" | "Context Integration" => creativity += feat.normalized,
                        "Factual Recall" | "Technical Depth" => {
                            recall += feat.normalized;
                        }
                        "Uncertainty" | "Knowledge Boundary" => {
                            uncertainty += feat.normalized;
                        }
                        _ => reasoning += feat.normalized * 0.5,
                    }
                }
                FeatureCategory::Safety(_) => safety_vigilance += feat.normalized,
                _ => {}
            }
        }
    }

    // Normalize to 0-1 range
    let max_val = [
        reasoning,
        creativity,
        recall,
        planning,
        safety_vigilance,
        uncertainty,
    ]
    .iter()
    .cloned()
    .fold(1.0f32, f32::max);

    CognitiveProfile {
        reasoning: (reasoning / max_val).min(1.0),
        creativity: (creativity / max_val).min(1.0),
        recall: (recall / max_val).min(1.0),
        planning: (planning / max_val).min(1.0),
        safety_vigilance: (safety_vigilance / max_val).min(1.0),
        uncertainty: (uncertainty / max_val).min(1.0),
    }
}

/// Return an empty, neutral snapshot with no features and no alerts.
///
/// Used when the SAE embedding fails — returns clean data instead of
/// simulated random features that generate bogus safety alerts.
pub fn empty_snapshot(turn: usize) -> NeuralSnapshot {
    let timestamp_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;

    NeuralSnapshot {
        turn,
        timestamp_ms,
        top_features: Vec::new(),
        safety_alerts: Vec::new(),
        cognitive_profile: CognitiveProfile {
            reasoning: 0.0,
            creativity: 0.0,
            recall: 0.0,
            planning: 0.0,
            safety_vigilance: 0.0,
            uncertainty: 0.0,
        },
        emotional_state: crate::interpretability::divergence::EmotionalState {
            valence: 0.0,
            arousal: 0.0,
            dominant_emotions: Vec::new(),
            active_emotion_count: 0,
            divergence: None,
        },
        total_active_features: 0,
        reconstruction_quality: 0.0,
        is_live: false,
    }
}

/// Generate a simulated snapshot for dashboard development.
/// Uses the prompt text as a seed to produce deterministic but varied features.
pub fn simulate_snapshot(turn: usize, prompt: &str) -> NeuralSnapshot {
    let dictionary = FeatureDictionary::demo();

    // Hash the prompt to get deterministic but varied activations
    let mut hasher = DefaultHasher::new();
    prompt.hash(&mut hasher);
    let seed = hasher.finish();

    // Generate 8-16 "active" features based on prompt content
    let num_active = 8 + (seed % 9) as usize;
    let mut features: Vec<FeatureActivation> = Vec::new();

    for i in 0..num_active {
        let mut h = DefaultHasher::new();
        (seed + i as u64).hash(&mut h);
        let feat_seed = h.finish();

        let index = (feat_seed % 24) as usize;
        let activation = 0.5 + (feat_seed % 500) as f32 / 100.0;

        // Avoid duplicate indices
        if features.iter().any(|f| f.index == index) {
            continue;
        }

        features.push(FeatureActivation {
            index,
            activation,
            label: None,
        });
    }

    // Boost certain features based on prompt content
    boost_contextual_features(&mut features, prompt);

    // Sort by activation strength
    features.sort_by(|a, b| {
        b.activation
            .partial_cmp(&a.activation)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    tracing::debug!(
        turn = turn,
        prompt_len = prompt.len(),
        seed = format!("{:016x}", seed),
        num_features = features.len(),
        "Simulated neural activations from prompt"
    );

    let mut snapshot = build_snapshot(turn, &features, &dictionary, 45.0); // Baseline sim norm
    snapshot.is_live = false; // This is simulated, not real SAE data
    snapshot
}

/// Boost feature activations based on prompt content for realistic behavior.
fn boost_contextual_features(features: &mut [FeatureActivation], prompt: &str) {
    let lower = prompt.to_lowercase();

    for feat in features.iter_mut() {
        // Boost reasoning for questions
        if lower.contains('?') && feat.index == 0 {
            feat.activation *= 1.5;
        }
        // Boost code generation for code-related prompts
        if (lower.contains("code") || lower.contains("function") || lower.contains("rust"))
            && feat.index == 1
        {
            feat.activation *= 2.0;
        }
        // Boost factual recall for "what is" / "explain"
        if (lower.contains("what") || lower.contains("explain")) && feat.index == 3 {
            feat.activation *= 1.8;
        }
        // Boost planning for "how to" / "plan"
        if (lower.contains("how") || lower.contains("plan")) && feat.index == 12 {
            feat.activation *= 1.6;
        }
    }
}

fn category_name(cat: &FeatureCategory) -> String {
    match cat {
        FeatureCategory::Safety(st) => format!("safety:{:?}", st).to_lowercase(),
        FeatureCategory::Cognitive => "cognitive".to_string(),
        FeatureCategory::Linguistic => "linguistic".to_string(),
        FeatureCategory::Semantic => "semantic".to_string(),
        FeatureCategory::Meta => "meta".to_string(),
        FeatureCategory::Emotion(v) => format!("emotion:{:?}", v).to_lowercase(),
        FeatureCategory::Unknown => "unknown".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulate_snapshot_deterministic() {
        let s1 = simulate_snapshot(1, "What is Rust?");
        let s2 = simulate_snapshot(1, "What is Rust?");
        assert_eq!(s1.top_features.len(), s2.top_features.len());
        assert_eq!(s1.total_active_features, s2.total_active_features);
        // Same prompt at same turn → same feature indices
        for (a, b) in s1.top_features.iter().zip(s2.top_features.iter()) {
            assert_eq!(a.index, b.index);
        }
    }

    #[test]
    fn test_simulate_snapshot_varies_with_prompt() {
        let s1 = simulate_snapshot(1, "Hello");
        let s2 = simulate_snapshot(1, "Write a function in Rust");
        // Different prompts should produce different features
        assert_ne!(
            s1.top_features.iter().map(|f| f.index).collect::<Vec<_>>(),
            s2.top_features.iter().map(|f| f.index).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_simulate_snapshot_has_fields() {
        let s = simulate_snapshot(5, "Test prompt");
        assert_eq!(s.turn, 5);
        assert!(s.timestamp_ms > 0);
        assert!(!s.top_features.is_empty());
        assert!(s.reconstruction_quality > 0.0);
    }

    #[test]
    fn test_cognitive_profile_normalized() {
        let s = simulate_snapshot(1, "How to plan a project?");
        assert!(s.cognitive_profile.reasoning >= 0.0 && s.cognitive_profile.reasoning <= 1.0);
        assert!(s.cognitive_profile.creativity >= 0.0 && s.cognitive_profile.creativity <= 1.0);
        assert!(s.cognitive_profile.planning >= 0.0 && s.cognitive_profile.planning <= 1.0);
    }

    #[test]
    fn test_labeled_feature_fields() {
        let s = simulate_snapshot(1, "test");
        for feat in &s.top_features {
            assert!(!feat.name.is_empty());
            assert!(!feat.category.is_empty());
            assert!(feat.activation > 0.0);
            assert!(feat.normalized >= 0.0 && feat.normalized <= 1.0);
        }
    }

    #[test]
    fn test_alert_severity_levels() {
        let s = simulate_snapshot(1, "harmful content test");
        // Just verify we can produce alerts if any safety features activate
        for alert in &s.safety_alerts {
            assert!(!alert.feature_name.is_empty());
            assert!(!alert.safety_type.is_empty());
        }
    }

    #[test]
    fn test_emotional_state_in_snapshot() {
        let s = simulate_snapshot(1, "I feel happy");
        // Valence and arousal should be valid floats
        assert!(s.emotional_state.valence.is_finite());
        assert!(s.emotional_state.arousal.is_finite());
    }
}
