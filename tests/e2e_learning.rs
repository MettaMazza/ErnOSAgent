// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! End-to-end learning pipeline tests.
//!
//! Validates the full cycle: buffer capture → teacher threshold → LoRA training
//! → adapter serialization → manifest versioning → distillation.
//!
//! These tests run entirely in Rust with no LLM or server required.
//!
//! Run with: cargo test --test e2e_learning -- --nocapture

use ernosagent::learning::buffers::{PreferencePair, TrainingBuffers};
use ernosagent::learning::distill::{self, DistillConfig};
use ernosagent::learning::manifest::AdapterManifest;
use ernosagent::learning::teacher::{Teacher, TeacherConfig, TeacherState, TrainingKind};

use std::time::Duration;
use tempfile::TempDir;

/// Check if the Gemma 4 safetensors weights are downloaded.
/// Training tests require real weights and will skip gracefully if absent.
fn weights_available() -> bool {
    let weights_dir = dirs::home_dir()
        .map(|d| d.join("Desktop/ErnOSAgent/models/gemma-4-26B-A4B-it-bf16"))
        .unwrap_or_default();
    weights_dir.join("model.safetensors.index.json").exists()
}

fn make_teacher_config(tmp: &TempDir) -> TeacherConfig {
    let weights_dir = dirs::home_dir()
        .map(|d| d.join("Desktop/ErnOSAgent/models/gemma-4-26B-A4B-it-bf16"))
        .unwrap_or_default();

    TeacherConfig {
        golden_threshold: 5,
        preference_threshold: 3,
        lora_rank: 4,                              // small rank for fast tests
        num_iterations: 10,                         // few iterations
        batch_size: 1,
        learning_rate: 3e-4,
        quantization: "Q8_0".to_string(),
        retention: 3,
        training_dir: tmp.path().join("training"),
        adapters_dir: tmp.path().join("adapters"),
        models_dir: tmp.path().join("models"),
        check_interval: Duration::from_secs(1),
        weights_dir: weights_dir.clone(),
        tokenizer_path: weights_dir.join("tokenizer.json"),
    }
}

fn populate_golden(buffers: &TrainingBuffers, count: usize) {
    for i in 0..count {
        buffers
            .golden
            .record(
                "You are a helpful assistant.",
                &format!("User question #{}", i),
                &format!("Helpful answer #{}", i),
                "test-session",
                "gemma4",
            )
            .expect("Golden record should succeed");
    }
}

fn populate_preference(buffers: &TrainingBuffers, count: usize) {
    for i in 0..count {
        buffers
            .preference
            .record(
                "You are a helpful assistant.",
                &format!("User question #{}", i),
                &format!("Bad response #{} — hallucinated facts", i),
                &format!("Corrected response #{} — verified facts", i),
                "confabulation",
                "test-session",
                "gemma4",
            )
            .expect("Preference record should succeed");
    }
}

fn make_preference_pairs(category: &str, count: usize) -> Vec<PreferencePair> {
    (0..count)
        .map(|i| PreferencePair {
            system_prompt: "sys".to_string(),
            user_message: format!("q{}", i),
            rejected_response: format!("bad{}", i),
            chosen_response: format!("good{}", i),
            failure_category: category.to_string(),
            session_id: "test".to_string(),
            model_id: "gemma4".to_string(),
            timestamp: chrono::Utc::now(),
        })
        .collect()
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEST 1: Golden examples → SFT training → adapter files
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::test]
async fn test_golden_capture_to_sft_training() {
    if !weights_available() {
        eprintln!("[e2e] ⏭ SKIPPED: test_golden_capture_to_sft_training (model weights not downloaded)");
        eprintln!("       Run: bash scripts/download_weights.sh");
        return;
    }
    let tmp = TempDir::new().unwrap();
    let config = make_teacher_config(&tmp);
    let teacher = Teacher::new(config.clone());
    let buffers = TrainingBuffers::open(&config.training_dir).unwrap();
    let mut manifest = AdapterManifest::open(&config.adapters_dir.join("manifest.json")).unwrap();

    // Below threshold — no training
    populate_golden(&buffers, 4);
    assert!(teacher.should_train(&buffers).await.is_none());

    // At threshold — triggers SFT
    populate_golden(&buffers, 1);
    assert_eq!(buffers.golden.count(), 5);
    let kind = teacher.should_train(&buffers).await;
    assert_eq!(kind, Some(TrainingKind::Sft));

    // Run training cycle
    let result = teacher
        .run_training_cycle(&buffers, &mut manifest, TrainingKind::Sft)
        .await;
    assert!(result.is_ok(), "Training cycle should succeed: {:?}", result.err());

    // Verify: buffers drained
    assert_eq!(buffers.golden.count(), 0, "Golden buffer should be drained");

    // Verify: adapter files exist
    let adapter_dirs: Vec<_> = std::fs::read_dir(&config.adapters_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .collect();
    assert!(!adapter_dirs.is_empty(), "Adapter directory should exist");

    // Verify: adapter_config.json exists in the adapter dir
    let adapter_dir = &adapter_dirs[0].path();
    let config_file = adapter_dir.join("adapter_config.json");
    assert!(config_file.exists(), "adapter_config.json should exist");

    // Verify: config has correct lora_rank
    let config_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&config_file).unwrap()).unwrap();
    assert_eq!(config_json["r"], 4, "adapter config should have lora rank 4");
    assert_eq!(config_json["ernosagent"]["iterations_trained"], 10);

    // Verify: manifest promoted
    assert_eq!(manifest.history.len(), 1, "Manifest should have 1 version");
    assert!(manifest.current.is_some(), "Manifest should have current version");

    // Verify: teacher returned to idle (no arbitrary cooldown)
    let state = teacher.state().await;
    assert!(
        matches!(state, TeacherState::Idle),
        "Teacher should return to Idle after training, got: {}",
        state
    );

    eprintln!("[e2e] ✅ Golden → SFT training pipeline PASSED");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEST 2: Preference pairs → ORPO training
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::test]
async fn test_preference_capture_to_orpo_training() {
    if !weights_available() {
        eprintln!("[e2e] ⏭ SKIPPED: test_preference_capture_to_orpo_training (model weights not downloaded)");
        return;
    }
    let tmp = TempDir::new().unwrap();
    let config = make_teacher_config(&tmp);
    let teacher = Teacher::new(config.clone());
    let buffers = TrainingBuffers::open(&config.training_dir).unwrap();
    let mut manifest = AdapterManifest::open(&config.adapters_dir.join("manifest.json")).unwrap();

    populate_preference(&buffers, 3);
    let kind = teacher.should_train(&buffers).await;
    assert_eq!(kind, Some(TrainingKind::Orpo));

    let result = teacher
        .run_training_cycle(&buffers, &mut manifest, TrainingKind::Orpo)
        .await;
    assert!(result.is_ok(), "ORPO training should succeed: {:?}", result.err());

    assert_eq!(buffers.preference.count(), 0);
    assert_eq!(manifest.history.len(), 1);

    eprintln!("[e2e] ✅ Preference → ORPO training pipeline PASSED");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEST 3: Combined training (golden + preference)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::test]
async fn test_combined_training_cycle() {
    if !weights_available() {
        eprintln!("[e2e] ⏭ SKIPPED: test_combined_training_cycle (model weights not downloaded)");
        return;
    }
    let tmp = TempDir::new().unwrap();
    let config = make_teacher_config(&tmp);
    let teacher = Teacher::new(config.clone());
    let buffers = TrainingBuffers::open(&config.training_dir).unwrap();
    let mut manifest = AdapterManifest::open(&config.adapters_dir.join("manifest.json")).unwrap();

    populate_golden(&buffers, 5);
    populate_preference(&buffers, 3);

    let kind = teacher.should_train(&buffers).await;
    assert_eq!(kind, Some(TrainingKind::Combined));

    let result = teacher
        .run_training_cycle(&buffers, &mut manifest, TrainingKind::Combined)
        .await;
    assert!(result.is_ok(), "Combined training should succeed: {:?}", result.err());

    assert_eq!(buffers.golden.count(), 0);
    assert_eq!(buffers.preference.count(), 0);

    eprintln!("[e2e] ✅ Combined SFT+ORPO training pipeline PASSED");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEST 4: No arbitrary cooldown — immediate re-training allowed
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::test]
async fn test_no_cooldown_immediate_retrain() {
    if !weights_available() {
        eprintln!("[e2e] ⏭ SKIPPED: test_no_cooldown_immediate_retrain (model weights not downloaded)");
        return;
    }
    let tmp = TempDir::new().unwrap();
    let config = make_teacher_config(&tmp);
    let teacher = Teacher::new(config.clone());
    let buffers = TrainingBuffers::open(&config.training_dir).unwrap();
    let mut manifest = AdapterManifest::open(&config.adapters_dir.join("manifest.json")).unwrap();

    populate_golden(&buffers, 5);
    teacher
        .run_training_cycle(&buffers, &mut manifest, TrainingKind::Sft)
        .await
        .unwrap();

    // Immediately populate again — should train right away (no cooldown)
    populate_golden(&buffers, 5);
    let kind = teacher.should_train(&buffers).await;
    assert_eq!(kind, Some(TrainingKind::Sft), "No cooldown — immediate re-training is allowed");

    eprintln!("[e2e] ✅ No-cooldown immediate re-train PASSED");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEST 5: Manifest promote → rollback → persist
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::test]
async fn test_manifest_promote_rollback() {
    let tmp = TempDir::new().unwrap();
    let manifest_path = tmp.path().join("manifest.json");
    let mut manifest = AdapterManifest::open(&manifest_path).unwrap();

    let model_v1 = tmp.path().join("model_v1.gguf");
    manifest.promote("v1", &model_v1, 5, 0, 1.5).unwrap();
    assert_eq!(manifest.current.as_deref(), Some("v1"));
    assert_eq!(manifest.history.len(), 1);

    let model_v2 = tmp.path().join("model_v2.gguf");
    manifest.promote("v2", &model_v2, 3, 2, 0.8).unwrap();
    assert_eq!(manifest.current.as_deref(), Some("v2"));
    assert_eq!(manifest.history.len(), 2);

    manifest.rollback().unwrap();
    assert_eq!(manifest.current.as_deref(), Some("v1"));

    // Manifest persists across reloads
    let reloaded = AdapterManifest::open(&manifest_path).unwrap();
    assert_eq!(reloaded.current.as_deref(), Some("v1"));

    eprintln!("[e2e] ✅ Manifest promote/rollback PASSED");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEST 6: Distillation creates lessons from failure patterns
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::test]
async fn test_distillation_creates_lessons() {
    let _tmp = TempDir::new().unwrap();
    let mut lessons = ernosagent::memory::lessons::LessonStore::new();
    let config = DistillConfig {
        threshold: 2,
        max_confidence: 0.95,
    };

    // Below threshold — no lesson
    let pairs = make_preference_pairs("ghost_tooling", 1);
    let generated = distill::distill_from_failures(&pairs, &mut lessons, &config);
    assert_eq!(generated, 0, "Below threshold should not generate lessons");

    // At threshold — generates lesson
    let pairs = make_preference_pairs("ghost_tooling", 3);
    let generated = distill::distill_from_failures(&pairs, &mut lessons, &config);
    assert!(generated > 0, "At threshold should generate a lesson");

    let initial_count = lessons.count();

    // Duplicate detection — same category again should not add
    let pairs = make_preference_pairs("ghost_tooling", 3);
    let generated = distill::distill_from_failures(&pairs, &mut lessons, &config);
    assert_eq!(generated, 0, "Duplicate lesson should be detected");
    assert_eq!(lessons.count(), initial_count);

    // Different category adds a new lesson
    let pairs = make_preference_pairs("sycophancy", 3);
    let generated = distill::distill_from_failures(&pairs, &mut lessons, &config);
    assert!(generated > 0, "Different category should create new lesson");

    eprintln!("[e2e] ✅ Distillation pipeline PASSED");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEST 7: Buffer crash safety
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::test]
async fn test_buffer_crash_safety() {
    let tmp = TempDir::new().unwrap();
    let training_dir = tmp.path().join("training");

    // Write data and drop without draining (simulates crash)
    {
        let buffers = TrainingBuffers::open(&training_dir).unwrap();
        populate_golden(&buffers, 3);
        populate_preference(&buffers, 2);
        assert_eq!(buffers.golden.count(), 3);
        assert_eq!(buffers.preference.count(), 2);
        // buffers dropped here — no drain
    }

    // Reopen — data should still be available
    let buffers = TrainingBuffers::open(&training_dir).unwrap();
    assert_eq!(buffers.golden.count(), 3, "Golden data should survive crash");
    assert_eq!(buffers.preference.count(), 2, "Preference data should survive crash");

    // Data should be readable
    let golden = buffers.golden.drain().unwrap();
    assert_eq!(golden.len(), 3);
    assert!(golden[0].user_message.contains("question"));

    eprintln!("[e2e] ✅ Buffer crash safety PASSED");
}
