// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
use super::*;

#[test]
fn test_gpu_backend_detect() {
    let _backend = GpuBackend::detect();
    #[cfg(target_os = "macos")]
    assert_eq!(_backend, GpuBackend::Metal);
}

#[test]
fn test_model_params_default() {
    let params = ModelParams::default();
    assert_eq!(params.n_gpu_layers, -1);
    assert!(params.use_mmap);
    assert!(!params.use_mlock);
}

#[test]
fn test_context_params_default() {
    let params = ContextParams::default();
    assert_eq!(params.n_ctx, 4096);
    assert_eq!(params.n_batch, 512);
    assert!(params.n_threads > 0);
    assert!(params.n_threads <= 8);
}

#[test]
fn test_sampling_params_default() {
    let params = SamplingParams::default();
    assert!((params.temperature - 0.7).abs() < f32::EPSILON);
    assert_eq!(params.top_k, 40);
}

#[test]
fn test_model_load_file_not_found() {
    let params = ModelParams {
        model_path: "/nonexistent/model.gguf".to_string(),
        ..Default::default()
    };
    let result = LlamaModel::load(params);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("not found"));
}

#[test]
fn test_model_load_stub() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("gemma-4-E2B-it-Q4_K_M.gguf");
    std::fs::write(&model_path, b"fake gguf data").unwrap();

    let params = ModelParams {
        model_path: model_path.to_str().unwrap().to_string(),
        ..Default::default()
    };
    let model = LlamaModel::load(params).unwrap();
    assert_eq!(model.info().architecture, "gemma2");
    assert_eq!(model.info().n_params, 2_300_000_000);
    assert_eq!(model.info().n_ctx_train, 128_000);
    assert_eq!(model.info().quantization, "Q4_K_M");
    assert!(!model.has_mmproj());
}

#[test]
fn test_model_with_mmproj() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("gemma-4-E4B-it-Q4_K_M.gguf");
    let mmproj_path = tmp.path().join("mmproj.gguf");
    std::fs::write(&model_path, b"fake").unwrap();
    std::fs::write(&mmproj_path, b"fake").unwrap();

    let params = ModelParams {
        model_path: model_path.to_str().unwrap().to_string(),
        mmproj_path: Some(mmproj_path.to_str().unwrap().to_string()),
        ..Default::default()
    };
    let model = LlamaModel::load(params).unwrap();
    assert!(model.has_mmproj());
    assert_eq!(model.info().n_params, 4_500_000_000);
}

#[test]
fn test_context_creation() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("test.gguf");
    std::fs::write(&model_path, b"fake").unwrap();

    let model = LlamaModel::load(ModelParams {
        model_path: model_path.to_str().unwrap().to_string(),
        ..Default::default()
    }).unwrap();

    let ctx = LlamaContext::new(&model, ContextParams::default());
    assert!(ctx.is_ok());
}

#[test]
fn test_cancel_flag() {
    let flag = CancelFlag::new();
    assert!(!flag.is_cancelled());
    flag.cancel();
    assert!(flag.is_cancelled());
    flag.reset();
    assert!(!flag.is_cancelled());
}

#[test]
fn test_stub_generation() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("test.gguf");
    std::fs::write(&model_path, b"fake").unwrap();

    let model = LlamaModel::load(ModelParams {
        model_path: model_path.to_str().unwrap().to_string(),
        ..Default::default()
    }).unwrap();

    let mut ctx = LlamaContext::new(&model, ContextParams::default()).unwrap();
    let cancel = CancelFlag::new();
    let mut tokens = Vec::new();

    let count = ctx.generate(
        &[], &SamplingParams::default(), 100, &cancel,
        |tok| tokens.push(tok),
    ).unwrap();

    assert!(count > 0);
    assert!(!tokens.is_empty());
    assert!(tokens.last().unwrap().is_eos);
}

#[test]
fn test_stub_generation_cancellation() {
    let tmp = tempfile::TempDir::new().unwrap();
    let model_path = tmp.path().join("test.gguf");
    std::fs::write(&model_path, b"fake").unwrap();

    let model = LlamaModel::load(ModelParams {
        model_path: model_path.to_str().unwrap().to_string(),
        ..Default::default()
    }).unwrap();

    let mut ctx = LlamaContext::new(&model, ContextParams::default()).unwrap();
    let cancel = CancelFlag::new();
    cancel.cancel();

    let mut tokens = Vec::new();
    let count = ctx.generate(
        &[], &SamplingParams::default(), 100, &cancel,
        |tok| tokens.push(tok),
    ).unwrap();

    assert_eq!(count, 0);
    assert!(tokens.is_empty());
}

#[test]
fn test_native_model_info() {
    let info = NativeModelInfo {
        name: "gemma-4-E2B".to_string(),
        architecture: "gemma2".to_string(),
        n_params: 2_300_000_000,
        n_ctx_train: 128_000,
        n_vocab: 262_144,
        n_embd: 2304,
        n_layer: 28,
        n_head: 8,
        file_size: 3_106_731_392,
        quantization: "Q4_K_M".to_string(),
    };
    assert_eq!(info.name, "gemma-4-E2B");
    assert_eq!(info.file_size, 3_106_731_392);
}
