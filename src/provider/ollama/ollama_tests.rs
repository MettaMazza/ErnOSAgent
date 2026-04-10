// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Tests for the Ollama provider.

use super::*;
use crate::config::OllamaConfig;
use crate::provider::Provider;


fn test_config() -> OllamaConfig {
    OllamaConfig {
        host: "http://localhost:11434".to_string(),
        port: 11434,
        keep_alive: -1,
    }
}

#[test]
fn test_provider_id() {
    let provider = OllamaProvider::new(&test_config());
    assert_eq!(provider.id(), "ollama");
}

#[test]
fn test_parse_show_response_gemma4() {
    let provider = OllamaProvider::new(&test_config());
    let body = serde_json::json!({
        "details": {
            "family": "gemma4",
            "families": ["gemma4"],
            "parameter_size": "26B",
            "quantization_level": "Q4_K_M",
            "format": "gguf"
        },
        "model_info": {
            "general.context_length": 262144
        },
        "parameters": "temperature 0.7\ntop_k 40\ntop_p 0.9\nnum_ctx 131072",
        "template": "{{ .System }}\n{{ .Prompt }}"
    });

    let spec = provider.parse_show_response("gemma4:26b", &body);
    assert_eq!(spec.name, "gemma4:26b");
    assert_eq!(spec.provider, "ollama");
    assert_eq!(spec.family, "gemma4");
    assert_eq!(spec.parameter_size, "26B");
    assert_eq!(spec.context_length, 262144);
    assert_eq!(spec.default_temperature, 0.7);
    assert_eq!(spec.default_top_k, 40);
    assert!(spec.capabilities.text);
    assert!(!spec.capabilities.audio);
}

#[test]
fn test_parse_show_response_vision_model() {
    let provider = OllamaProvider::new(&test_config());
    let body = serde_json::json!({
        "details": {
            "family": "llama",
            "families": ["llama", "clip"],
            "parameter_size": "11B",
            "quantization_level": "Q4_0",
            "format": "gguf"
        },
        "model_info": {},
        "parameters": "",
    });

    let spec = provider.parse_show_response("llava:11b", &body);
    assert!(spec.capabilities.vision);
    assert!(spec.capabilities.video);
}

#[test]
fn test_parse_show_response_minimal() {
    let provider = OllamaProvider::new(&test_config());
    let body = serde_json::json!({});

    let spec = provider.parse_show_response("unknown", &body);
    assert_eq!(spec.name, "unknown");
    assert_eq!(spec.context_length, 0);
    assert!(spec.capabilities.text);
}
