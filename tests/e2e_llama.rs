// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! End-to-end integration test: llama-server → Provider → ReAct pipeline.
//!
//! Tests are split into layers:
//! 1. Model spec auto-derivation (no inference)
//! 2. Raw inference (one-shot chat completion)
//! 3. Full ReAct pipeline (tool calling)  
//! 4. Neo4j knowledge graph
//!
//! Run with: cargo test --test e2e_llama -- --nocapture

use ernosagent::inference::context;
use ernosagent::memory::MemoryManager;
use ernosagent::prompt;
use ernosagent::provider::llamacpp::LlamaCppProvider;
use ernosagent::provider::{Message, Provider, StreamEvent};
use ernosagent::react::r#loop::{execute_react_loop, ReactConfig, ReactEvent};
use ernosagent::react::reply;
use ernosagent::tools::executor::ToolExecutor;
use std::sync::Arc;
use tokio::sync::mpsc;

/// The llama-server binary (built from latest llama.cpp source).
const LLAMA_SERVER_BIN: &str = "/tmp/llama.cpp.build/build/bin/llama-server";

/// Gemma 4 26B-A4B-it Q4_K_M — the production model.
const MODEL_GGUF: &str = "/Users/mettamazza/Desktop/ErnOSAgent/models/gemma-4-26b-it-Q4_K_M.gguf";

/// Port for the test server.
const TEST_PORT: u16 = 8199;

fn skip_if_missing() -> bool {
    if !std::path::Path::new(LLAMA_SERVER_BIN).exists() {
        eprintln!("[e2e] SKIP: llama-server not found at {}", LLAMA_SERVER_BIN);
        return true;
    }
    if !std::path::Path::new(MODEL_GGUF).exists() {
        eprintln!("[e2e] SKIP: model GGUF not found at {}", MODEL_GGUF);
        return true;
    }
    false
}

struct TestServer {
    process: std::process::Child,
    port: u16,
}

impl TestServer {
    async fn start(port: u16) -> Self {
        let process = std::process::Command::new(LLAMA_SERVER_BIN)
            .args([
                "--model",
                MODEL_GGUF,
                "--port",
                &port.to_string(),
                "--ctx-size",
                "8192",
                "--n-gpu-layers",
                "-1",
                "--no-warmup",
            ])
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("Failed to start llama-server");

        let mut server = Self { process, port };

        let client = reqwest::Client::new();
        let url = format!("http://localhost:{}/health", port);
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(180);

        while tokio::time::Instant::now() < deadline {
            match client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    eprintln!("[e2e] llama-server healthy on port {}", port);
                    return server;
                }
                _ => tokio::time::sleep(std::time::Duration::from_millis(500)).await,
            }
        }

        server.kill();
        panic!(
            "llama-server failed to become healthy on port {} within 180s",
            port
        );
    }

    fn kill(&mut self) {
        self.process.kill().ok();
        self.process.wait().ok();
    }

    fn make_config(&self) -> ernosagent::config::LlamaCppConfig {
        ernosagent::config::LlamaCppConfig {
            server_binary: LLAMA_SERVER_BIN.to_string(),
            port: self.port,
            model_path: MODEL_GGUF.to_string(),
            mmproj_path: String::new(),
            n_gpu_layers: -1,
            extra_args: Vec::new(),
            embedding_model_path: String::new(),
            embedding_port: 8081,
        }
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.kill();
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEST 1: Model spec auto-derivation
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::test]
async fn test_e2e_model_spec_derivation() {
    if skip_if_missing() {
        return;
    }

    let server = TestServer::start(TEST_PORT).await;
    let config = server.make_config();
    let provider = LlamaCppProvider::new(&config);

    // Verify model spec auto-derivation
    let spec = provider
        .get_model_spec("gemma4")
        .await
        .expect("Must derive spec");

    eprintln!("[e2e] Auto-derived spec:");
    eprintln!("  Name: {}", spec.name);
    eprintln!("  Context: {}", spec.context_length);
    eprintln!("  Params: {}", spec.parameter_size);
    eprintln!("  Quant: {}", spec.quantization_level);
    eprintln!("  Format: {}", spec.format);
    eprintln!("  Status: {}", spec.status_line());

    assert!(spec.is_derived(), "Spec must be auto-derived");
    assert!(spec.context_length > 0, "Context length must be > 0");
    assert!(spec.capabilities.text, "Must support text");

    // Verify health endpoint
    let status = provider.health().await.expect("Health check must work");
    assert!(status.available, "Server must be available");
    eprintln!(
        "[e2e] Health: available={}, latency={:?}ms",
        status.available, status.latency_ms
    );

    eprintln!("[e2e] ✅ Model spec derivation test PASSED");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEST 2: Raw inference (streaming SSE)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::test]
async fn test_e2e_raw_inference() {
    if skip_if_missing() {
        return;
    }

    let server = TestServer::start(TEST_PORT + 1).await;
    let config = server.make_config();
    let provider: Arc<dyn Provider> = Arc::new(LlamaCppProvider::new(&config));

    let messages = vec![
        Message {
            role: "system".to_string(),
            content: "You are a helpful assistant.".to_string(),
            images: Vec::new(),
        },
        Message {
            role: "user".to_string(),
            content: "Say hello in exactly one word.".to_string(),
            images: Vec::new(),
        },
    ];

    eprintln!("[e2e] Running raw inference...");
    let (tx, mut rx) = mpsc::channel::<StreamEvent>(256);

    let provider_clone = Arc::clone(&provider);
    let msgs = messages.clone();
    let handle = tokio::spawn(async move { provider_clone.chat("gemma4", &msgs, None, tx).await });

    let mut response = String::new();
    let mut got_done = false;

    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::Token(t) => {
                response.push_str(&t);
                eprint!("{}", t);
            }
            StreamEvent::Done { .. } => {
                got_done = true;
                break;
            }
            StreamEvent::Error(e) => {
                eprintln!("\n[e2e] ❌ Inference error: {}", e);
            }
            _ => {}
        }
    }

    handle
        .await
        .expect("Inference task panicked")
        .expect("Inference should succeed");

    eprintln!(
        "\n[e2e] Response ({} chars): {}",
        response.len(),
        response.trim()
    );
    assert!(got_done, "Must receive Done event");
    assert!(!response.is_empty(), "Response must not be empty");

    eprintln!("[e2e] ✅ Raw inference test PASSED");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEST 3: ReAct loop pipeline (validates wiring)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::test]
async fn test_e2e_react_pipeline() {
    if skip_if_missing() {
        return;
    }

    let server = TestServer::start(TEST_PORT + 2).await;
    let config = server.make_config();
    let provider: Arc<dyn Provider> = Arc::new(LlamaCppProvider::new(&config));

    let model_spec = provider
        .get_model_spec("gemma4")
        .await
        .expect("Must derive spec");

    let core_prompt = prompt::core::build_core_prompt();
    let context_prompt = prompt::context::build_context_prompt(
        &model_spec,
        "E2E Test",
        0,
        0.0,
        &["reply_request".to_string()],
        &ernosagent::steering::vectors::SteeringConfig::default(),
        "Memory: test mode",
        "",
    );
    let system_prompt = prompt::assemble_system_prompt(&core_prompt, &context_prompt, "");

    let messages = context::build_context(
        &system_prompt,
        &[],
        &[Message {
            role: "user".to_string(),
            content: "Hello! Say hi back.".to_string(),
            images: Vec::new(),
        }],
        model_spec.context_length,
    );

    let tools = vec![reply::reply_request_tool()];
    let executor = ToolExecutor::new();
    let react_config = ReactConfig {
        observer_enabled: false,
        observer_model: None,
        context_length: 131072,
    };

    eprintln!("[e2e] Running ReAct loop (timeout 30s)...");
    let (tx, mut rx) = mpsc::channel::<ReactEvent>(256);

    let provider_clone = Arc::clone(&provider);
    let sys = system_prompt.clone();
    let result_handle = tokio::spawn(async move {
        execute_react_loop(
            &provider_clone,
            "gemma4",
            messages,
            &tools,
            &executor,
            &react_config,
            &sys,
            "",
            tx,
            None,
            "e2e-test",
            #[cfg(feature = "discord")]
            None,
            None,
        )
        .await
    });

    // Collect events with a 120s timeout.
    // Gemma 4 26B should follow tool-calling instructions and call reply_request.
    // We time-bound this to prevent infinite loops in CI.
    let mut events_received = 0_usize;
    let mut turns_seen = 0_usize;

    let _timeout_result = tokio::time::timeout(std::time::Duration::from_secs(120), async {
        while let Some(event) = rx.recv().await {
            events_received += 1;
            match &event {
                ReactEvent::TurnStarted { turn } => {
                    turns_seen = *turn;
                    eprintln!("[e2e] ReAct turn {}", turn);
                    // After 3 turns, abort — pipeline is validated
                    if *turn >= 3 {
                        eprintln!("[e2e] Pipeline validated after {} turns, aborting", turn);
                        return;
                    }
                }
                ReactEvent::Token(t) => eprint!("{}", t),
                ReactEvent::Error(e) => eprintln!("\n[e2e] Error: {}", e),
                ReactEvent::ResponseReady { text } => {
                    eprintln!("\n[e2e] ✅ Got response: {} chars", text.len());
                    return;
                }
                _ => {}
            }
        }
    })
    .await;

    // Abort the react loop
    result_handle.abort();

    eprintln!("[e2e] Events: {}, Turns: {}", events_received, turns_seen);
    assert!(events_received > 0, "Must receive at least one event");
    assert!(turns_seen >= 1, "Must complete at least one turn");

    eprintln!("[e2e] ✅ ReAct pipeline test PASSED (wiring validated)");
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// TEST 4: Neo4j Knowledge Graph
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

#[tokio::test]
async fn test_e2e_neo4j_knowledge_graph() {
    let tmp = tempfile::TempDir::new().unwrap();

    let mut mgr = MemoryManager::new(
        tmp.path(),
        "bolt://localhost:7687",
        "neo4j",
        "ernosagent",
        "neo4j",
    )
    .await
    .unwrap();

    if !mgr.kg_available() {
        eprintln!("[e2e] SKIP: Neo4j not available (auth or connection failed)");
        return;
    }

    let kg = mgr.knowledge_graph.as_ref().unwrap();

    // Create entities
    let rust_id = kg
        .upsert_entity(
            "Rust",
            "language",
            &serde_json::json!({"paradigm": "systems"}),
        )
        .await
        .expect("Failed to create Rust entity");

    let alice_id = kg
        .upsert_entity("Alice", "person", &serde_json::json!({"role": "engineer"}))
        .await
        .expect("Failed to create Alice entity");

    eprintln!("[e2e] Created entities: {} and {}", rust_id, alice_id);

    // Create relation
    kg.upsert_relation(&alice_id, &rust_id, "knows", 0.9)
        .await
        .expect("Failed to create relation");

    // Verify counts
    let entity_count = kg.entity_count().await.unwrap();
    let relation_count = kg.relation_count().await.unwrap();
    eprintln!(
        "[e2e] KG state: {} entities, {} relations",
        entity_count, relation_count
    );

    assert!(entity_count >= 2);
    assert!(relation_count >= 1);

    // Search
    let results = kg.search_entities("Rust", 10).await.unwrap();
    assert!(!results.is_empty(), "Should find Rust entity");

    // Memory manager status
    let mem_status = mgr.status_summary().await;
    assert!(!mem_status.contains("KG: offline"));

    // Ingest turn
    mgr.ingest_turn(
        "What is Rust?",
        "Rust is a systems programming language.",
        "e2e-test",
    )
    .await
    .unwrap();
    assert_eq!(mgr.timeline.entry_count(), 2);

    eprintln!("[e2e] ✅ Neo4j knowledge graph test PASSED");
}
