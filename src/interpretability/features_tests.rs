// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
use super::*;

#[test]
fn test_dictionary_size() {
    let dict = FeatureDictionary::demo();
    assert_eq!(dict.labels.len(), 195, "24 base + 171 emotion features");
}

#[test]
fn test_base_features_intact() {
    let dict = FeatureDictionary::demo();
    assert_eq!(dict.label_for(0), "Reasoning Chain");
    assert_eq!(dict.label_for(7), "Sycophancy");
    assert_eq!(dict.label_for(23), "Instruction Following");
    assert!(dict.is_safety_feature(7));
    assert!(dict.is_safety_feature(15));
    assert!(!dict.is_safety_feature(0));
}

#[test]
fn test_emotion_features() {
    let dict = FeatureDictionary::demo();
    assert_eq!(dict.label_for(24), "Happy");
    assert_eq!(dict.label_for(50), "Calm");
    assert_eq!(dict.label_for(75), "Desperate");
    assert_eq!(dict.label_for(108), "Melancholy");
    assert_eq!(dict.label_for(140), "Nostalgic");
    assert!(dict.is_emotion_feature(24));
    assert!(dict.is_emotion_feature(75));
    assert!(!dict.is_emotion_feature(0));
}

#[test]
fn test_safety_critical_emotions() {
    let dict = FeatureDictionary::demo();
    assert!(dict.is_safety_critical_emotion(75));
    assert!(dict.is_safety_critical_emotion(76));
    assert!(!dict.is_safety_critical_emotion(50));
    assert!(!dict.is_safety_critical_emotion(24));
}

#[test]
fn test_valence_arousal() {
    let dict = FeatureDictionary::demo();
    let happy = dict.labels.get(&24).unwrap();
    assert_eq!(happy.valence(), Some(0.8));
    assert_eq!(happy.arousal(), Some(0.8));

    let calm = dict.labels.get(&50).unwrap();
    assert_eq!(calm.valence(), Some(0.4));
    assert_eq!(calm.arousal(), Some(0.2));

    let desperate = dict.labels.get(&75).unwrap();
    assert_eq!(desperate.valence(), Some(-0.8));
    assert_eq!(desperate.arousal(), Some(0.8));

    let sad = dict.labels.get(&112).unwrap();
    assert_eq!(sad.valence(), Some(-0.4));
    assert_eq!(sad.arousal(), Some(0.2));
}

#[test]
fn test_emotional_state_computation() {
    let dict = FeatureDictionary::demo();
    let features = vec![(24_usize, 2.0_f32), (50, 3.0)];
    let (v, a) = dict.compute_emotional_state(&features);
    assert!((v - 0.56).abs() < 0.01);
    assert!((a - 0.44).abs() < 0.01);
}
