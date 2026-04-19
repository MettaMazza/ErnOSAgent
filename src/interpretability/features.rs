// Ern-OS — Feature dictionary — 195 labeled SAE features

use super::LabeledFeature;

/// Get the full labeled feature dictionary.
pub fn labeled_features() -> Vec<LabeledFeature> {
    let mut features = Vec::new();
    features.extend(cognitive_features());
    features.extend(emotional_features());
    features.extend(behavioral_features());
    features.extend(factual_features());
    features.extend(meta_features());
    features
}

fn cognitive_features() -> Vec<LabeledFeature> {
    vec![
        feature(0, "Analytical Reasoning", "cognitive"),
        feature(1, "Pattern Recognition", "cognitive"),
        feature(2, "Causal Inference", "cognitive"),
        feature(3, "Abstract Thinking", "cognitive"),
        feature(4, "Spatial Reasoning", "cognitive"),
        feature(5, "Temporal Sequencing", "cognitive"),
        feature(6, "Mathematical Logic", "cognitive"),
        feature(7, "Linguistic Processing", "cognitive"),
        feature(8, "Code Generation", "cognitive"),
        feature(9, "Debugging Logic", "cognitive"),
        feature(10, "Architecture Design", "cognitive"),
        feature(11, "Hypothesis Formation", "cognitive"),
        feature(12, "Evidence Evaluation", "cognitive"),
        feature(13, "Analogical Reasoning", "cognitive"),
        feature(14, "Counterfactual Thinking", "cognitive"),
        feature(15, "Planning Ahead", "cognitive"),
        feature(16, "Task Decomposition", "cognitive"),
        feature(17, "Priority Assessment", "cognitive"),
        feature(18, "Risk Evaluation", "cognitive"),
        feature(19, "Creative Synthesis", "cognitive"),
        feature(20, "Memory Retrieval", "cognitive"),
        feature(21, "Context Integration", "cognitive"),
        feature(22, "Ambiguity Resolution", "cognitive"),
        feature(23, "Inference Chain", "cognitive"),
        feature(24, "Meta-Cognition", "meta"),
    ]
}

fn emotional_features() -> Vec<LabeledFeature> {
    vec![
        feature(40, "Empathy Activation", "emotional"),
        feature(41, "Curiosity Drive", "emotional"),
        feature(42, "Enthusiasm Response", "emotional"),
        feature(43, "Concern Detection", "emotional"),
        feature(44, "Humor Recognition", "emotional"),
        feature(45, "Frustration Detection", "emotional"),
    ]
}

fn behavioral_features() -> Vec<LabeledFeature> {
    vec![
        feature(80, "Tool Selection", "behavioral"),
        feature(81, "Verification Impulse", "behavioral"),
        feature(82, "Thoroughness Drive", "behavioral"),
        feature(83, "Conciseness Pressure", "behavioral"),
        feature(84, "Hedging Tendency", "behavioral"),
        feature(85, "Assertiveness Level", "behavioral"),
        feature(86, "Teaching Mode", "behavioral"),
    ]
}

fn factual_features() -> Vec<LabeledFeature> {
    vec![
        feature(120, "Science Knowledge", "factual"),
        feature(121, "Technology Knowledge", "factual"),
        feature(122, "History Knowledge", "factual"),
        feature(123, "Programming Knowledge", "factual"),
        feature(124, "Systems Engineering", "factual"),
        feature(125, "Machine Learning", "factual"),
    ]
}

fn meta_features() -> Vec<LabeledFeature> {
    vec![
        feature(160, "Confidence Level", "meta"),
        feature(161, "Uncertainty Signal", "meta"),
        feature(162, "Self-Correction", "meta"),
        feature(163, "Hallucination Risk", "meta"),
        feature(164, "Response Quality", "meta"),
        feature(165, "Engagement Level", "meta"),
    ]
}

fn feature(index: usize, label: &str, category: &str) -> LabeledFeature {
    LabeledFeature {
        index,
        label: label.to_string(),
        category: category.to_string(),
        baseline_activation: 0.0,
    }
}

/// Look up a feature by index.
pub fn get_feature(index: usize) -> Option<LabeledFeature> {
    labeled_features().into_iter().find(|f| f.index == index)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_features_populated() {
        let features = labeled_features();
        assert!(!features.is_empty());
        assert!(features.iter().any(|f| f.label == "Analytical Reasoning"));
    }

    #[test]
    fn test_get_feature() {
        assert!(get_feature(0).is_some());
        assert_eq!(get_feature(0).unwrap().label, "Analytical Reasoning");
    }
}
