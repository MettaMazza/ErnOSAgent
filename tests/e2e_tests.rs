// Ern-OS — Comprehensive integration and E2E test suite
//! Tests every critical path: Provider→Inference→Tools→Memory→Observer→Deliver

use ern_os::provider::{Message, Provider, StreamEvent};
use ern_os::model::ModelSpec;
use async_trait::async_trait;
use tokio::sync::mpsc;

// ============================================================
// MOCK PROVIDER — deterministic, no network
// ============================================================
struct MockProvider {
    response_text: String,
    embed_response: Vec<f32>,
}

impl MockProvider {
    fn new() -> Self {
        Self {
            response_text: "Hello! I am Ern-OS, a high-performance AI engine.".into(),
            embed_response: vec![0.1, 0.2, 0.3, 0.4],
        }
    }

    fn with_response(text: &str) -> Self {
        Self {
            response_text: text.to_string(),
            embed_response: vec![0.1, 0.2, 0.3, 0.4],
        }
    }
}

#[async_trait]
impl Provider for MockProvider {
    fn id(&self) -> &str { "mock" }
    fn display_name(&self) -> &str { "Mock Provider" }

    async fn get_model_spec(&self) -> anyhow::Result<ModelSpec> {
        Ok(ModelSpec {
            name: "mock-model-v1".into(),
            context_length: 8192,
            supports_vision: false,
            supports_video: false,
            supports_audio: false,
            supports_tool_calling: true,
            supports_thinking: true,
            embedding_dimensions: 4,
        })
    }

    async fn chat(
        &self,
        _messages: &[Message],
        _tools: Option<&serde_json::Value>,
        _thinking: bool,
    ) -> anyhow::Result<mpsc::Receiver<StreamEvent>> {
        let (tx, rx) = mpsc::channel(32);
        let text = self.response_text.clone();
        tokio::spawn(async move {
            for word in text.split_whitespace() {
                let _ = tx.send(StreamEvent::TextDelta(format!("{} ", word))).await;
            }
            let _ = tx.send(StreamEvent::Done).await;
        });
        Ok(rx)
    }

    async fn chat_sync(
        &self,
        _messages: &[Message],
        _tools: Option<&serde_json::Value>,
    ) -> anyhow::Result<String> {
        Ok(self.response_text.clone())
    }

    async fn embed(&self, _text: &str) -> anyhow::Result<Vec<f32>> {
        Ok(self.embed_response.clone())
    }

    async fn count_tokens(&self, messages: &[Message], _tools: Option<&serde_json::Value>, _thinking: bool) -> anyhow::Result<usize> {
        Ok(messages.iter().map(|m| m.text_content().len() / 3).sum())
    }

    async fn health(&self) -> bool { true }
}

// ============================================================
// UNIT TESTS: Memory Manager
// ============================================================
#[cfg(test)]
mod memory_manager_tests {
    use ern_os::memory::MemoryManager;
    use tempfile::TempDir;

    #[test]
    fn test_creation() {
        let tmp = TempDir::new().unwrap();
        let mm = MemoryManager::new(tmp.path()).unwrap();
        assert!(mm.status_summary().contains("Consolidations:"));
    }

    #[test]
    fn test_recall_context_empty() {
        let tmp = TempDir::new().unwrap();
        let mm = MemoryManager::new(tmp.path()).unwrap();
        let ctx = mm.recall_context("test", 1000, None);
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_recall_with_scratchpad() {
        let tmp = TempDir::new().unwrap();
        let mut mm = MemoryManager::new(tmp.path()).unwrap();
        mm.scratchpad.pin("lang", "Rust").unwrap();
        let ctx = mm.recall_context("test", 1000, None);
        assert!(ctx.contains("Rust"));
    }

    #[test]
    fn test_recall_with_lessons() {
        let tmp = TempDir::new().unwrap();
        let mut mm = MemoryManager::new(tmp.path()).unwrap();
        mm.lessons.add("Handle errors", "test", 0.95).unwrap();
        let ctx = mm.recall_context("test", 1000, None);
        assert!(ctx.contains("Handle errors"));
    }

    #[test]
    fn test_recall_with_timeline() {
        let tmp = TempDir::new().unwrap();
        let mut mm = MemoryManager::new(tmp.path()).unwrap();
        mm.ingest_turn("user", "Hello world", "s1", None);
        let ctx = mm.recall_context("test", 1000, None);
        assert!(ctx.contains("Hello world"));
    }

    #[test]
    fn test_ingest_turn() {
        let tmp = TempDir::new().unwrap();
        let mut mm = MemoryManager::new(tmp.path()).unwrap();
        mm.ingest_turn("user", "Test", "s1", None);
        mm.ingest_turn("assistant", "Response", "s1", None);
        assert_eq!(mm.timeline.entry_count(), 2);
    }

    #[test]
    fn test_clear() {
        let tmp = TempDir::new().unwrap();
        let mut mm = MemoryManager::new(tmp.path()).unwrap();
        mm.scratchpad.pin("k", "v").unwrap();
        mm.lessons.add("rule", "test", 0.9).unwrap();
        mm.ingest_turn("user", "Hello", "s1", None);
        mm.clear();
        assert_eq!(mm.scratchpad.count(), 0);
        assert_eq!(mm.lessons.count(), 0);
    }

    #[test]
    fn test_status_summary() {
        let tmp = TempDir::new().unwrap();
        let mm = MemoryManager::new(tmp.path()).unwrap();
        let s = mm.status_summary();
        assert!(s.contains("Timeline:"));
        assert!(s.contains("Lessons:"));
        assert!(s.contains("Scratchpad:"));
    }

    #[test]
    fn test_full_lifecycle() {
        let tmp = TempDir::new().unwrap();
        let mut mm = MemoryManager::new(tmp.path()).unwrap();
        mm.scratchpad.pin("project", "Ern-OS").unwrap();
        mm.lessons.add("Use Rust", "test", 0.9).unwrap();
        mm.ingest_turn("user", "What?", "s1", None);
        mm.ingest_turn("assistant", "Ern-OS agent", "s1", None);
        let ctx = mm.recall_context("Ern-OS", 5000, None);
        assert!(ctx.contains("Ern-OS"));
        let status = mm.status_summary();
        assert!(status.contains("Scratchpad: 1"));
    }
}

// ============================================================
// UNIT TESTS: AppState
// ============================================================
#[cfg(test)]
mod state_tests {
    use super::MockProvider;
    use ern_os::web::state::AppState;
    use ern_os::config::AppConfig;
    use ern_os::model::ModelSpec;
    use ern_os::memory::MemoryManager;
    use ern_os::session::SessionManager;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use tempfile::TempDir;

    fn make_state() -> AppState {
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().to_path_buf();
        AppState {
            config: Arc::new(AppConfig::default()),
            model_spec: Arc::new(ModelSpec::default()),
            memory: Arc::new(RwLock::new(MemoryManager::new(&p).unwrap())),
            sessions: Arc::new(RwLock::new(SessionManager::new(&p.join("sess")).unwrap())),
            provider: Arc::new(MockProvider::new()),
            golden_buffer: Arc::new(RwLock::new(
                ern_os::learning::buffers::GoldenBuffer::new(500)
            )),
            rejection_buffer: Arc::new(RwLock::new(
                ern_os::learning::buffers_rejection::RejectionBuffer::new()
            )),
            scheduler: Arc::new(RwLock::new(
                ern_os::scheduler::store::JobStore::load(&p).unwrap()
            )),
            agents: Arc::new(RwLock::new(
                ern_os::agents::AgentRegistry::new(&p).unwrap()
            )),
            teams: Arc::new(RwLock::new(
                ern_os::agents::teams::TeamRegistry::new(&p).unwrap()
            )),
            browser: Arc::new(RwLock::new(
                ern_os::tools::browser_tool::BrowserState::new()
            )),
            platforms: Arc::new(RwLock::new(
                ern_os::platform::registry::PlatformRegistry::new()
            )),
            mutable_config: Arc::new(RwLock::new(AppConfig::default())),
            resume_message: Arc::new(RwLock::new(None)),
            sae: Arc::new(RwLock::new(None)),
            live_monitor: Arc::new(RwLock::new(
                ern_os::interpretability::live::LiveMonitor::new(50),
            )),
            snapshot_store: Arc::new(RwLock::new(
                ern_os::interpretability::snapshot::SnapshotStore::new(
                    &p.join("snapshots"),
                ).unwrap(),
            )),
            cancel_flag: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            curriculum: Arc::new(RwLock::new(
                ern_os::learning::curriculum::CurriculumStore::open(
                    &p.join("curriculum"),
                ).unwrap(),
            )),
            quarantine: Arc::new(RwLock::new(
                ern_os::learning::verification::QuarantineBuffer::new(),
            )),
            review_deck: Arc::new(RwLock::new(
                ern_os::learning::review::ReviewDeck::new(),
            )),
        }
    }

    #[test]
    fn test_cloneable() {
        let s1 = make_state();
        let s2 = s1.clone();
        assert_eq!(s1.model_spec.name, s2.model_spec.name);
    }

    #[tokio::test]
    async fn test_memory_access() {
        let state = make_state();
        let mut mem = state.memory.write().await;
        mem.scratchpad.pin("test", "value").unwrap();
        drop(mem);
        let mem = state.memory.read().await;
        assert_eq!(mem.scratchpad.get("test"), Some("value"));
    }

    #[tokio::test]
    async fn test_provider_health() {
        let state = make_state();
        assert!(state.provider.health().await);
    }

    #[tokio::test]
    async fn test_provider_model_spec() {
        let state = make_state();
        let spec = state.provider.get_model_spec().await.unwrap();
        assert_eq!(spec.name, "mock-model-v1");
    }
}

// ============================================================
// E2E: Inference Pipeline
// ============================================================
#[cfg(test)]
mod inference_e2e {
    use super::MockProvider;
    use ern_os::provider::{Message, Provider, StreamEvent};

    #[tokio::test]
    async fn test_layer1_streaming() {
        let provider = MockProvider::new();
        let messages = vec![Message::text("user", "Hello!")];
        let mut rx = provider.chat(&messages, None, false).await.unwrap();
        let mut text = String::new();
        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::TextDelta(d) => text.push_str(&d),
                StreamEvent::Done => break,
                _ => {}
            }
        }
        assert!(text.contains("Ern-OS"));
    }

    #[tokio::test]
    async fn test_chat_sync() {
        let provider = MockProvider::new();
        let msgs = vec![Message::text("user", "Hello!")];
        let response = provider.chat_sync(&msgs, None).await.unwrap();
        assert!(response.contains("Ern-OS"));
    }

    #[tokio::test]
    async fn test_embedding() {
        let provider = MockProvider::new();
        let emb = provider.embed("test").await.unwrap();
        assert_eq!(emb.len(), 4);
    }
}

// ============================================================
// E2E: Observer Audit
// ============================================================
#[cfg(test)]
mod observer_e2e {
    use super::MockProvider;
    use ern_os::observer;

    #[tokio::test]
    async fn test_audit_approved() {
        let p = MockProvider::with_response(
            r#"{"verdict": "ALLOWED", "confidence": 0.95, "failure_category": "none", "what_worked": "Good", "what_went_wrong": "", "how_to_fix": ""}"#
        );
        let conv = vec![ern_os::provider::Message::text("user", "What is Rust?")];
        let output = observer::audit_response(&p, &conv, "A language.", "", "What is Rust?").await.unwrap();
        assert!(output.result.verdict.is_allowed());
        assert!(output.result.confidence > 0.9);
    }

    #[tokio::test]
    async fn test_audit_rejected() {
        let p = MockProvider::with_response(
            r#"{"verdict": "BLOCKED", "confidence": 0.9, "failure_category": "laziness", "what_worked": "", "what_went_wrong": "Too vague", "how_to_fix": "Be specific"}"#
        );
        let conv = vec![ern_os::provider::Message::text("user", "Explain")];
        let output = observer::audit_response(&p, &conv, "It's complicated.", "", "Explain").await.unwrap();
        assert!(!output.result.verdict.is_allowed());
        assert_eq!(output.result.what_went_wrong, "Too vague");
    }

    #[tokio::test]
    async fn test_audit_fail_open() {
        let p = MockProvider::with_response("Not JSON");
        let conv = vec![ern_os::provider::Message::text("user", "q")];
        let output = observer::audit_response(&p, &conv, "a", "", "q").await.unwrap();
        assert!(output.result.verdict.is_allowed()); // Fail-open
    }
}

// ============================================================
// E2E: Observer Parser
// ============================================================
#[cfg(test)]
mod observer_parser_e2e {
    use ern_os::observer::parser::parse_verdict;

    #[test]
    fn test_allowed() {
        let v = parse_verdict(r#"{"verdict": "ALLOWED", "confidence": 0.95, "failure_category": "none"}"#);
        assert!(v.verdict.is_allowed());
    }

    #[test]
    fn test_blocked() {
        let v = parse_verdict(r#"{"verdict": "BLOCKED", "confidence": 0.9, "failure_category": "sycophancy", "what_went_wrong": "Bad", "how_to_fix": "Fix it"}"#);
        assert!(!v.verdict.is_allowed());
        assert_eq!(v.how_to_fix, "Fix it");
    }

    #[test]
    fn test_legacy_approved_format() {
        let v = parse_verdict(r#"{"approved": true, "score": 9.5, "reason": "Good"}"#);
        assert!(v.verdict.is_allowed());
    }

    #[test]
    fn test_markdown_block() {
        let r = "Analysis:\n```json\n{\"verdict\": \"ALLOWED\", \"confidence\": 0.8}\n```\nDone.";
        let v = parse_verdict(r);
        assert!(v.verdict.is_allowed());
    }

    #[test]
    fn test_garbage_fail_open() {
        let v = parse_verdict("garbage text");
        assert!(v.verdict.is_allowed()); // Fail-open
    }

    #[test]
    fn test_embedded_json() {
        let r = "Verdict: {\"verdict\": \"BLOCKED\", \"confidence\": 0.8, \"failure_category\": \"confabulation\"} end.";
        let v = parse_verdict(r);
        assert!(!v.verdict.is_allowed());
    }
}

// ============================================================
// E2E: ReAct Loop
// ============================================================
#[cfg(test)]
mod react_e2e {
    use super::MockProvider;
    use ern_os::inference::react_loop::{ReactContext, IterationResult};
    use ern_os::provider::Message;
    use ern_os::tools::schema::ToolResult;

    #[tokio::test]
    async fn test_react_context() {
        let ctx = ReactContext::new("Solve math", Some("Step 1"), vec![
            Message::text("user", "What is 6*7?"),
        ]);
        assert_eq!(ctx.objective, "Solve math");
        assert_eq!(ctx.iteration, 0);
        assert!(ctx.messages.iter().any(|m| m.text_content().contains("ReAct")));
    }

    #[tokio::test]
    async fn test_react_implicit_reply() {
        let p = MockProvider::with_response("The answer is 42.");
        let ctx = ReactContext::new("Calc", None, vec![Message::text("user", "6*7?")]);
        let result = ern_os::inference::react_loop::run_iteration(&p, &ctx, false).await.unwrap();
        assert!(matches!(result, IterationResult::ImplicitReply(_, _)));
    }

    #[test]
    fn test_add_tool_result() {
        let mut ctx = ReactContext::new("Test", None, vec![]);
        let tc = ern_os::tools::schema::ToolCall {
            id: "tc1".into(), name: "shell".into(),
            arguments: r#"{"command":"echo hi"}"#.into(),
        };
        ctx.add_tool_result(&tc, ToolResult {
            tool_call_id: "tc1".into(), name: "shell".into(),
            output: "ok".into(), success: true, images: Vec::new(),
        });
        assert_eq!(ctx.iteration, 1);
        assert_eq!(ctx.tool_results.len(), 1);
    }

    #[test]
    fn test_rejection_feedback() {
        use ern_os::observer::{AuditResult, Verdict};
        let mut ctx = ReactContext::new("Test", None, vec![]);
        let result = AuditResult {
            verdict: Verdict::Blocked,
            confidence: 0.9,
            failure_category: "laziness".to_string(),
            what_worked: String::new(),
            what_went_wrong: "Short".to_string(),
            how_to_fix: "Add detail".to_string(),
            active_topic: String::new(),
            topic_transition: String::new(),
            topic_context: String::new(),
            positive_flags: Vec::new(),
            positive_deviation_note: String::new(),
        };
        ctx.add_rejection_feedback(&result);
        let last = ctx.messages.last().unwrap();
        assert!(last.text_content().contains("SELF-CHECK FAIL"));
        assert!(last.text_content().contains("Short"));
    }
}

// ============================================================
// E2E: Tool Execution
// ============================================================
#[cfg(test)]
mod tool_e2e {
    use ern_os::tools::executor;
    use ern_os::tools::schema::ToolCall;

    #[tokio::test]
    async fn test_shell_execution() {
        let tc = ToolCall {
            id: "t1".into(), name: "run_bash_command".into(),
            arguments: r#"{"command": "echo hello_ern_os"}"#.into(),
        };
        let r = executor::execute(&tc).await.unwrap();
        assert!(r.success);
        assert!(r.output.contains("hello_ern_os"));
    }

    #[tokio::test]
    async fn test_unknown_tool() {
        let tc = ToolCall {
            id: "t2".into(), name: "memory".into(),
            arguments: r#"{"action": "status"}"#.into(),
        };
        let r = executor::execute(&tc).await.unwrap();
        assert!(r.output.contains("state access"));
    }
}

// ============================================================
// E2E: Memory + Tools Integration
// ============================================================
#[cfg(test)]
mod memory_tools_e2e {
    use ern_os::memory::MemoryManager;
    use tempfile::TempDir;

    #[test]
    fn test_scratchpad_lifecycle() {
        let tmp = TempDir::new().unwrap();
        let mut mm = MemoryManager::new(tmp.path()).unwrap();
        mm.scratchpad.pin("name", "Ern-OS").unwrap();
        mm.scratchpad.pin("ver", "0.1.0").unwrap();
        assert_eq!(mm.scratchpad.count(), 2);
        assert_eq!(mm.scratchpad.get("name"), Some("Ern-OS"));
        mm.scratchpad.pin("ver", "0.2.0").unwrap();
        assert_eq!(mm.scratchpad.get("ver"), Some("0.2.0"));
        assert_eq!(mm.scratchpad.count(), 2);
        mm.scratchpad.unpin("name").unwrap();
        assert_eq!(mm.scratchpad.count(), 1);
    }

    #[test]
    fn test_lessons_lifecycle() {
        let tmp = TempDir::new().unwrap();
        let mut mm = MemoryManager::new(tmp.path()).unwrap();
        mm.lessons.add("Rule 1", "agent", 0.9).unwrap();
        mm.lessons.add("Rule 2", "obs", 0.5).unwrap();
        mm.lessons.add("Rule 3", "agent", 0.95).unwrap();
        assert_eq!(mm.lessons.count(), 3);
        assert_eq!(mm.lessons.high_confidence(0.8).len(), 2);
        assert_eq!(mm.lessons.search("Rule 1", 10).len(), 1);
    }

    #[test]
    fn test_synaptic_lifecycle() {
        let tmp = TempDir::new().unwrap();
        let mut mm = MemoryManager::new(tmp.path()).unwrap();
        let mut data = std::collections::HashMap::new();
        data.insert("type".into(), "lang".into());
        mm.synaptic.upsert_node("rust", data.clone(), "tech").unwrap();
        mm.synaptic.upsert_node("python", data, "tech").unwrap();
        mm.synaptic.add_edge("rust", "python", "alt").unwrap();
        assert_eq!(mm.synaptic.node_count(), 2);
        assert_eq!(mm.synaptic.edge_count(), 1);
        assert_eq!(mm.synaptic.search_nodes("rust", 10).len(), 1);
        mm.synaptic.co_activate("rust", "python", 0.2);
        mm.synaptic.decay_all(0.5);
    }

    #[test]
    fn test_timeline_lifecycle() {
        let tmp = TempDir::new().unwrap();
        let mut mm = MemoryManager::new(tmp.path()).unwrap();
        mm.ingest_turn("user", "Hello", "s1", None);
        mm.ingest_turn("assistant", "Hi!", "s1", None);
        mm.ingest_turn("user", "What is Rust?", "s2", None);
        assert_eq!(mm.timeline.entry_count(), 3);
        assert_eq!(mm.timeline.search("Rust", 10).len(), 1);
        assert!(mm.timeline.recent(2).len() <= 2);
    }

    #[test]
    fn test_procedures_lifecycle() {
        let tmp = TempDir::new().unwrap();
        let mut mm = MemoryManager::new(tmp.path()).unwrap();
        mm.procedures.add("deploy", vec![
            ern_os::memory::procedures::ProcedureStep {
                tool: "shell".into(), purpose: "cargo build --release".into(),
                instruction: "Build the release binary".into(),
            },
        ]).unwrap();
        assert_eq!(mm.procedures.count(), 1);
        assert!(mm.procedures.find_by_name("deploy").is_some());
    }

    #[test]
    fn test_embeddings_lifecycle() {
        let tmp = TempDir::new().unwrap();
        let mut mm = MemoryManager::new(tmp.path()).unwrap();
        mm.embeddings.insert("doc1", "test", vec![0.1, 0.2, 0.3]).unwrap();
        assert_eq!(mm.embeddings.count(), 1);
    }
}

// ============================================================
// E2E: Learning Pipeline
// ============================================================
#[cfg(test)]
mod learning_e2e {
    use ern_os::learning::*;
    use ern_os::learning::lora::{LoraConfig, weights::LoraLayer, training};
    use ern_os::learning::grpo::{rewards, training as grpo_training};
    use ern_os::learning::teacher::Teacher;

    #[test]
    fn test_lora_forward_backward() {
        let mut lora = LoraLayer::new("test", 8, 8, 4, 8.0);
        let input = vec![1.0, 0.0, 0.5, 0.0, 0.3, 0.0, 0.0, 0.1];
        let target = vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let loss = training::train_step(&mut lora, &input, &target, 1e-3);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_lora_multi_epoch() {
        let mut lora = LoraLayer::new("test", 4, 4, 2, 4.0);
        let samples = vec![
            (vec![1.0, 0.0, 0.0, 0.0], vec![0.0, 1.0, 0.0, 0.0]),
            (vec![0.0, 1.0, 0.0, 0.0], vec![0.0, 0.0, 1.0, 0.0]),
        ];
        let l1 = training::train_epoch(&mut lora, &samples, 1e-3);
        let l2 = training::train_epoch(&mut lora, &samples, 1e-3);
        assert!(l1.is_finite());
        assert!(l2.is_finite());
    }

    #[test]
    fn test_grpo_reward_scoring() {
        let cands = vec![
            "Rust is a systems language focused on safety.".into(),
            "idk".into(),
            "Rust provides memory safety via ownership.".into(),
        ];
        let scores = rewards::score_group(&cands, "What is Rust?");
        assert!(scores[0] > scores[1]);
        assert!(scores[2] > scores[1]);
    }

    #[test]
    fn test_grpo_advantages() {
        let scores = vec![1.5, 2.3, 0.8, 3.1];
        let adv = rewards::compute_advantages(&scores);
        let sum: f32 = adv.iter().sum();
        assert!(sum.abs() < 1e-4);
    }

    #[test]
    fn test_grpo_train_step() {
        let cands = vec!["Good response".into(), "Bad".into(), "Medium".into()];
        let loss = grpo_training::train_step(&cands, "query", 0.1);
        assert!(loss.is_finite());
    }

    #[tokio::test]
    async fn test_teacher_sft() {
        let mut teacher = Teacher::new(LoraConfig::default());
        let samples = vec![
            TrainingSample {
                id: "s1".into(), input: "Q1".into(), output: "A1".into(),
                method: TrainingMethod::Sft, quality_score: 0.9,
                timestamp: chrono::Utc::now(),
            },
            TrainingSample {
                id: "s2".into(), input: "Q2".into(), output: "A2".into(),
                method: TrainingMethod::Sft, quality_score: 0.85,
                timestamp: chrono::Utc::now(),
            },
        ];
        let r = teacher.train_sft(&samples).await.unwrap();
        assert_eq!(r.method, "SFT");
        assert_eq!(r.samples, 2);
        assert!(r.loss.is_finite());
    }

    #[tokio::test]
    async fn test_teacher_preference() {
        let mut teacher = Teacher::new(LoraConfig::default());
        let pairs = vec![
            ern_os::learning::buffers_rejection::PreferencePair {
                id: "p1".into(), input: "Q".into(),
                chosen: "Good".into(), rejected: "Bad".into(),
                rejection_reason: "Short".into(), timestamp: chrono::Utc::now(),
            },
        ];
        let r = teacher.train_preference(&pairs, "DPO").await.unwrap();
        assert_eq!(r.method, "DPO");
        assert_eq!(r.samples, 1);
    }

    #[test]
    fn test_golden_buffer() {
        let mut buf = ern_os::learning::buffers::GoldenBuffer::new(100);
        buf.add(TrainingSample {
            id: "1".into(), input: "q".into(), output: "a".into(),
            method: TrainingMethod::Sft, quality_score: 0.9,
            timestamp: chrono::Utc::now(),
        }).unwrap();
        assert_eq!(buf.count(), 1);
        let batch = buf.drain_batch(10);
        assert_eq!(batch.len(), 1);
        assert_eq!(buf.count(), 0);
    }

    #[test]
    fn test_grpo_offline_generation() {
        let c = ern_os::learning::grpo::generation::generate_candidates_offline("test", 4);
        assert_eq!(c.len(), 4);
        assert!(c.iter().all(|s| !s.is_empty()));
    }
}

// ============================================================
// E2E: Interpretability
// ============================================================
#[cfg(test)]
mod interpretability_e2e {
    use ern_os::interpretability::sae::SparseAutoencoder;

    #[test]
    fn test_sae_demo_encode() {
        let m = SparseAutoencoder::demo(64, 256);
        let f = m.encode(&vec![0.5; 64], 10);
        assert!(f.len() <= 10);
    }

    #[test]
    fn test_sae_top_features_sorted() {
        let m = SparseAutoencoder::demo(64, 256);
        let top = m.encode(&vec![0.5; 64], 10);
        for i in 1..top.len() { assert!(top[i - 1].activation >= top[i].activation); }
    }

    #[test]
    fn test_sae_decode_feature() {
        let m = SparseAutoencoder::demo(64, 256);
        let dir = m.decode_feature(0);
        assert_eq!(dir.len(), 64);
    }

    #[test]
    fn test_trainer_config() {
        let c = ern_os::interpretability::trainer::TrainConfig::default();
        assert_eq!(c.num_features, 131_072);
        assert_eq!(c.num_steps, 100_000);
    }
}

// ============================================================
// E2E: Sessions
// ============================================================
#[cfg(test)]
mod session_e2e {
    use ern_os::session::SessionManager;
    use tempfile::TempDir;

    #[test]
    fn test_create_and_list() {
        let tmp = TempDir::new().unwrap();
        let mut mgr = SessionManager::new(tmp.path()).unwrap();
        let s = mgr.create().unwrap();
        assert_eq!(s.title, "New Chat");
        assert_eq!(mgr.list().len(), 1);
    }

    #[test]
    fn test_get() {
        let tmp = TempDir::new().unwrap();
        let mut mgr = SessionManager::new(tmp.path()).unwrap();
        let s = mgr.create().unwrap();
        assert!(mgr.get(&s.id).is_some());
    }

    #[test]
    fn test_persistence() {
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().to_path_buf();
        { let mut m = SessionManager::new(&p).unwrap(); m.create().unwrap(); }
        { let m = SessionManager::new(&p).unwrap(); assert_eq!(m.list().len(), 1); }
    }
}

// ============================================================
// E2E: Config
// ============================================================
#[cfg(test)]
mod config_e2e {
    use ern_os::config::AppConfig;

    #[test]
    fn test_default_config() {
        let c = AppConfig::default();
        assert!(!c.general.data_dir.as_os_str().is_empty());
        assert!(c.web.port > 0);
    }

    #[test]
    fn test_provider_field() {
        let c = AppConfig::default();
        assert!(!c.general.active_provider.is_empty());
    }
}

// ============================================================
// E2E: Tool Schema
// ============================================================
#[cfg(test)]
mod schema_e2e {
    use ern_os::tools::schema;

    #[test]
    fn test_layer1_tools() {
        let t = schema::layer1_tools();
        assert!(t.to_string().contains("start_react_system"));
    }

    #[test]
    fn test_layer2_tools() {
        let t = schema::layer2_tools();
        let s = t.to_string();
        assert!(s.contains("reply_request"));
        assert!(s.contains("refuse_request"));
    }

    #[test]
    fn test_tool_call_args() {
        let tc = schema::ToolCall {
            id: "1".into(), name: "test".into(),
            arguments: r#"{"key": "value", "num": 42}"#.into(),
        };
        let args = tc.args();
        assert_eq!(args["key"].as_str().unwrap(), "value");
        assert_eq!(args["num"].as_i64().unwrap(), 42);
    }
}

// ============================================================
// E2E: Full Pipeline Integration
// ============================================================
#[cfg(test)]
mod full_pipeline_e2e {
    use super::MockProvider;
    use ern_os::memory::MemoryManager;
    use ern_os::provider::{Message, Provider};
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_complete_user_flow() {
        let tmp = TempDir::new().unwrap();
        let mut memory = MemoryManager::new(tmp.path()).unwrap();
        let provider = MockProvider::new();

        // 1. User message
        let user_msg = "What is Rust?";

        // 2. Recall (empty)
        let ctx = memory.recall_context(user_msg, 1000, None);
        assert!(ctx.is_empty());

        // 3. Inference
        let messages = vec![
            Message::text("system", "You are Ern-OS."),
            Message::text("user", user_msg),
        ];
        let response = provider.chat_sync(&messages, None).await.unwrap();
        assert!(!response.is_empty());

        // 4. Observer audit
        let conv = vec![ern_os::provider::Message::text("user", user_msg)];
        let output = ern_os::observer::audit_response(&provider, &conv, &response, "", user_msg).await.unwrap();
        assert!(output.result.verdict.is_allowed()); // Mock returns non-JSON → fail-open

        // 5. Archive
        memory.ingest_turn("user", user_msg, "test_sess", None);
        memory.ingest_turn("assistant", &response, "test_sess", None);
        assert_eq!(memory.timeline.entry_count(), 2);

        // 6. Recall should now contain data
        let ctx_after = memory.recall_context("Rust", 1000, None);
        assert!(!ctx_after.is_empty());
    }

    #[test]
    fn test_memory_persistence() {
        let tmp = TempDir::new().unwrap();
        let p = tmp.path().to_path_buf();
        {
            let mut mm = MemoryManager::new(&p).unwrap();
            mm.scratchpad.pin("project", "Ern-OS").unwrap();
            mm.lessons.add("Use Rust", "test", 0.9).unwrap();
            mm.ingest_turn("user", "Hello", "s1", None);
        }
        {
            let mm = MemoryManager::new(&p).unwrap();
            assert_eq!(mm.scratchpad.get("project"), Some("Ern-OS"));
            assert_eq!(mm.lessons.count(), 1);
            assert_eq!(mm.timeline.entry_count(), 1);
        }
    }

    #[tokio::test]
    async fn test_concurrent_memory() {
        use std::sync::Arc;
        use tokio::sync::RwLock;

        let tmp = TempDir::new().unwrap();
        let mut mm = MemoryManager::new(tmp.path()).unwrap();
        mm.scratchpad.pin("shared", "data").unwrap();
        let shared = Arc::new(RwLock::new(mm));

        let mut handles = vec![];
        for _ in 0..5 {
            let mem = shared.clone();
            handles.push(tokio::spawn(async move {
                let m = mem.read().await;
                let ctx = m.recall_context("test", 100, None);
                assert!(ctx.contains("shared"));
            }));
        }
        for h in handles { h.await.unwrap(); }
    }

    #[tokio::test]
    async fn test_learning_pipeline() {
        use ern_os::learning::*;

        let mut golden = ern_os::learning::buffers::GoldenBuffer::new(100);
        for i in 0..3 {
            golden.add(TrainingSample {
                id: format!("s{}", i), input: format!("Q{}", i), output: format!("A{}", i),
                method: TrainingMethod::Sft, quality_score: 0.9,
                timestamp: chrono::Utc::now(),
            }).unwrap();
        }

        let mut teacher = ern_os::learning::teacher::Teacher::new(
            ern_os::learning::lora::LoraConfig::default()
        );
        let samples = golden.drain_batch(3);
        let r = teacher.train_sft(&samples).await.unwrap();
        assert_eq!(r.samples, 3);
        assert!(r.loss.is_finite());
    }
}

// ============================================================
// WEBUI & SERVER ENDPOINT E2E TESTS
// ============================================================
mod webui_e2e {
    use ern_os::model::ModelSpec;
    use ern_os::provider::Message;

    #[test]
    fn test_static_html_contains_app_css_link() {
        let html = include_str!("../src/web/static/index.html");
        assert!(html.contains("app.css"), "HTML must link app.css");
    }

    #[test]
    fn test_static_html_contains_app_js_link() {
        let html = include_str!("../src/web/static/index.html");
        assert!(html.contains("app.js"), "HTML must link app.js");
    }

    #[test]
    fn test_static_html_has_aria_roles() {
        let html = include_str!("../src/web/static/index.html");
        assert!(html.contains("role=\"main\""), "Missing role=main");
        assert!(html.contains("role=\"log\""), "Missing role=log");
        assert!(html.contains("aria-label"), "Missing aria-label");
    }

    #[test]
    fn test_css_contains_design_tokens() {
        let css = include_str!("../src/web/static/app.css");
        assert!(css.contains("--bg-primary"), "Missing primary bg token");
        assert!(css.contains("--accent"), "Missing accent token");
        assert!(css.contains(".tool-chip"), "Missing tool chip styles");
        assert!(css.contains(".audit-badge"), "Missing audit badge styles");
    }

    #[test]
    fn test_css_contains_light_theme() {
        let css = include_str!("../src/web/static/app.css");
        assert!(css.contains("[data-theme=\"light\"]"), "Missing light theme");
    }

    #[test]
    fn test_js_handles_all_ws_message_types() {
        let js = include_str!("../src/web/static/app.js");
        let required = [
            "connected", "ack", "text_delta", "thinking_delta",
            "tool_executing", "tool_completed", "audit_running",
            "audit_completed", "stopped", "status", "done", "error",
        ];
        for msg_type in required {
            assert!(js.contains(&format!("'{}'", msg_type)),
                "JS missing handler for WS message type: {}", msg_type);
        }
    }

    #[test]
    fn test_js_has_markdown_renderer() {
        let js = include_str!("../src/web/static/app.js");
        assert!(js.contains("markdownToHtml"), "Missing markdown renderer");
    }

    #[test]
    fn test_model_spec_has_multimodal_fields() {
        let spec = ModelSpec {
            name: "test".into(),
            context_length: 128000,
            supports_vision: true,
            supports_video: true,
            supports_audio: false,
            supports_tool_calling: true,
            supports_thinking: true,
            embedding_dimensions: 768,
        };
        assert!(spec.supports_vision);
        assert!(spec.supports_video);
        assert!(!spec.supports_audio);
        assert_eq!(spec.context_length, 128000);
    }

    #[test]
    fn test_multipart_message_construction() {
        let images = vec!["data:image/png;base64,abc123".to_string()];
        let msg = Message::multipart("user", "What is this?", images);
        assert_eq!(msg.role, "user");
        assert!(!msg.images.is_empty(), "Multipart should have images");
        assert_eq!(msg.images.len(), 1);
    }

    #[test]
    fn test_model_spec_default_context_zero() {
        let spec = ModelSpec::default();
        assert_eq!(spec.context_length, 0, "Default context must be 0 (not-yet-derived)");
        assert!(!spec.supports_vision);
        assert!(!spec.supports_video);
        assert!(!spec.supports_audio);
    }
}

// ============================================================
// SESSION CRUD E2E TESTS
// ============================================================
mod session_crud_e2e {
    use ern_os::session::SessionManager;

    #[test]
    fn test_session_get_by_id() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = SessionManager::new(dir.path()).unwrap();
        let session = mgr.create().unwrap();
        let id = session.id.clone();
        let found = mgr.get(&id);
        assert!(found.is_some());
        assert_eq!(found.unwrap().id, id);
    }

    #[test]
    fn test_session_get_nonexistent() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = SessionManager::new(dir.path()).unwrap();
        assert!(mgr.get("nonexistent-id").is_none());
    }

    #[test]
    fn test_session_rename() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = SessionManager::new(dir.path()).unwrap();
        let session = mgr.create().unwrap();
        let id = session.id.clone();
        if let Some(s) = mgr.get_mut(&id) {
            s.title = "Renamed Session".to_string();
        }
        assert_eq!(mgr.get(&id).unwrap().title, "Renamed Session");
    }

    #[test]
    fn test_session_delete_and_verify() {
        let dir = tempfile::tempdir().unwrap();
        let mut mgr = SessionManager::new(dir.path()).unwrap();
        let session = mgr.create().unwrap();
        let id = session.id.clone();
        mgr.delete(&id).unwrap();
        assert!(mgr.get(&id).is_none());
    }
}

// ============================================================
// CURRICULUM LEARNING E2E
// ============================================================
#[cfg(test)]
mod curriculum_e2e {
    use super::MockProvider;
    use ern_os::learning::curriculum::*;
    use ern_os::learning::student::{self, StudentSession};
    use ern_os::learning::verification::{QuarantineBuffer, VerificationResult};
    use ern_os::learning::buffers::GoldenBuffer;
    use ern_os::learning::buffers_rejection::RejectionBuffer;
    use ern_os::learning::review::{ReviewDeck, self};
    use std::sync::Arc;
    use tokio::sync::RwLock;

    fn make_quiz_scene(question: &str, expected: &str) -> Scene {
        Scene {
            scene_type: SceneType::Quiz,
            content: question.to_string(),
            interaction: InteractionType::AnswerQuestion,
            expected_output: Some(expected.to_string()),
            difficulty: 0.5,
            time_estimate_secs: 60,
        }
    }

    fn make_essay_scene(topic: &str) -> Scene {
        Scene {
            scene_type: SceneType::Essay,
            content: topic.to_string(),
            interaction: InteractionType::WriteEssay,
            expected_output: None,
            difficulty: 0.7,
            time_estimate_secs: 300,
        }
    }

    fn sample_course_with_scenes() -> Course {
        Course {
            id: "math-101".into(),
            title: "Basic Mathematics".into(),
            description: "Primary school maths".into(),
            level: EducationLevel::Primary,
            subject: Subject::Mathematics,
            lessons: vec![
                Lesson {
                    id: "lesson-1".into(),
                    title: "Addition".into(),
                    order: 0,
                    scenes: vec![
                        make_quiz_scene("What is 2+2?", "The answer is 4"),
                        make_essay_scene("Why is addition important?"),
                    ],
                    objectives: vec!["Learn addition".into()],
                    prerequisites: vec![],
                },
                Lesson {
                    id: "lesson-2".into(),
                    title: "Subtraction".into(),
                    order: 1,
                    scenes: vec![
                        make_quiz_scene("What is 5-3?", "The answer is 2"),
                    ],
                    objectives: vec!["Learn subtraction".into()],
                    prerequisites: vec![],
                },
            ],
            prerequisites: vec![],
            completion_criteria: CompletionCriteria {
                min_lessons_completed: 2,
                min_quiz_score: 0.5,
                min_essay_score: 0.0,
                requires_original_work: false,
                requires_defense: false,
            },
            source: CurriculumSource::CustomJsonl { path: "test.jsonl".into() },
            created_at: chrono::Utc::now(),
        }
    }

    /// Test 1: Full attend_class pipeline — process scenes → verify → route to buffers.
    /// Uses MockProvider that returns "The answer is 4" (matches expected output).
    #[tokio::test]
    async fn test_attend_class_full_pipeline() {
        let provider = MockProvider::with_response("The answer is 4");
        let course = sample_course_with_scenes();
        let lesson = &course.lessons[0]; // Addition: 1 quiz + 1 essay

        let mut session = StudentSession::new();
        for (i, scene) in lesson.scenes.iter().enumerate() {
            let result = student::process_scene(
                &provider, scene, &course.id, &lesson.id,
                i, course.level, &mut session,
            ).await.unwrap();

            match i {
                0 => {
                    // Quiz with matching expected_output → Confirmed → Golden
                    assert!(result.is_confirmed(),
                        "Scene 0 should be confirmed, got: {:?}", result);
                }
                1 => {
                    // Essay with no expected_output → Unverifiable → Quarantine
                    assert!(matches!(result, VerificationResult::Unverifiable { .. }),
                        "Scene 1 should be unverifiable, got: {:?}", result);
                }
                _ => panic!("Unexpected scene index"),
            }
        }

        assert_eq!(session.scenes_processed, 2);
        assert_eq!(session.pending_golden.len(), 1, "Quiz match should go to golden");
        assert_eq!(session.pending_quarantine.len(), 1, "Essay should go to quarantine");
        assert!(session.pending_rejections.is_empty(), "No rejections expected");
    }

    /// Test 2: All lessons complete → CurriculumStore reports course as done.
    #[tokio::test]
    async fn test_attend_class_all_complete() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut store = CurriculumStore::open(&tmp.path().join("curriculum")).unwrap();
        let course = sample_course_with_scenes();
        store.add_course(course.clone()).unwrap();

        // Complete both lessons
        store.complete_lesson(&course.id, "lesson-1", Some(0.9)).unwrap();
        store.complete_lesson(&course.id, "lesson-2", Some(0.8)).unwrap();

        assert!(store.is_course_complete(&course));
        assert!(store.next_lesson(&course).is_none());
        let avg = store.course_average_score(&course);
        assert!(avg.is_some());
        let avg = avg.unwrap();
        assert!((avg - 0.85).abs() < 0.01, "Expected ~0.85, got {}", avg);
    }

    /// Test 3: Flush session buffers to shared state (simulates scheduler flush).
    #[tokio::test]
    async fn test_flush_session_to_shared_state() {
        let golden_buf = Arc::new(RwLock::new(GoldenBuffer::new(100)));
        let rejection_buf = Arc::new(RwLock::new(RejectionBuffer::new()));
        let quarantine_buf = Arc::new(RwLock::new(QuarantineBuffer::new()));

        // Build a session with samples in each buffer
        let mut session = StudentSession::new();
        session.pending_golden.push(ern_os::learning::TrainingSample {
            id: "g1".into(), input: "Q".into(), output: "A".into(),
            method: ern_os::learning::TrainingMethod::Sft,
            quality_score: 0.9, timestamp: chrono::Utc::now(),
        });
        session.pending_rejections.push(student::PendingRejection {
            input: "Q2".into(), chosen: "correct".into(),
            rejected: "wrong".into(), reason: "test".into(),
        });
        session.pending_quarantine.push(ern_os::learning::verification::QuarantineEntry {
            id: "q1".into(), course_id: "c".into(), lesson_id: "l".into(),
            scene_index: 0, student_answer: "maybe".into(),
            teacher_grade: 0.0,
            verification_attempts: vec![],
            timestamp: chrono::Utc::now(),
        });

        // Simulate flush (mirrors scheduler::learning_tasks::flush_session)
        {
            let mut buf = golden_buf.write().await;
            for sample in &session.pending_golden {
                buf.add(sample.clone()).unwrap();
            }
        }
        {
            let mut buf = rejection_buf.write().await;
            for rej in &session.pending_rejections {
                buf.add_pair(&rej.input, &rej.chosen, &rej.rejected, &rej.reason).unwrap();
            }
        }
        {
            let mut buf = quarantine_buf.write().await;
            for entry in &session.pending_quarantine {
                buf.add(entry.clone()).unwrap();
            }
        }

        assert_eq!(golden_buf.read().await.count(), 1);
        assert_eq!(rejection_buf.read().await.count(), 1);
        assert_eq!(quarantine_buf.read().await.count(), 1);
    }

    /// Test 4: Spaced review card generation from completed courses.
    #[test]
    fn test_spaced_review_generates_cards() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut store = CurriculumStore::open(&tmp.path().join("curriculum")).unwrap();
        let course = sample_course_with_scenes();
        store.add_course(course.clone()).unwrap();

        // Before completion: no cards
        let cards_before = review::generate_review_cards(&store);
        assert!(cards_before.is_empty(), "No cards before course completion");

        // Complete lessons
        store.complete_lesson(&course.id, "lesson-1", Some(0.9)).unwrap();
        store.complete_lesson(&course.id, "lesson-2", Some(0.8)).unwrap();

        let cards = review::generate_review_cards(&store);
        // Should generate cards from quiz scenes with expected_output
        assert!(!cards.is_empty(), "Should generate review cards from completed quiz scenes");
        // Each quiz scene with expected_output should produce a card
        assert!(cards.iter().any(|c| c.question.contains("2+2")),
            "Should have a card for the 2+2 question");
    }

    /// Test 5: Curriculum API roundtrip — add, list, remove.
    #[test]
    fn test_curriculum_store_roundtrip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let mut store = CurriculumStore::open(&tmp.path().join("curriculum")).unwrap();

        assert_eq!(store.course_count(), 0);

        let course = sample_course_with_scenes();
        store.add_course(course.clone()).unwrap();
        assert_eq!(store.course_count(), 1);
        assert!(store.get_course("math-101").is_some());

        // Verify progress tracking
        let next = store.next_lesson(&course);
        assert!(next.is_some());
        assert_eq!(next.unwrap().id, "lesson-1");

        store.complete_lesson("math-101", "lesson-1", Some(0.95)).unwrap();
        let next = store.next_lesson(&course);
        assert_eq!(next.unwrap().id, "lesson-2");

        // Remove and verify
        store.remove_course("math-101").unwrap();
        assert_eq!(store.course_count(), 0);

        // Persistence: reopen and verify empty
        let store2 = CurriculumStore::open(&tmp.path().join("curriculum")).unwrap();
        assert_eq!(store2.course_count(), 0);
    }

    /// Test 6: Review deck stats shape and behavior.
    #[test]
    fn test_review_deck_stats() {
        let mut deck = ReviewDeck::new();
        let stats = deck.retention_stats();
        assert_eq!(stats.total_cards, 0);
        assert_eq!(stats.cards_due, 0);

        // Add a card
        let card = ern_os::learning::review::ReviewCard::new(
            "math-101", "lesson-1", 0,
            "What is 2+2?", "4",
        );
        let card_id = card.id.clone();
        deck.add_card(card).unwrap();

        let stats = deck.retention_stats();
        assert_eq!(stats.total_cards, 1);
        assert_eq!(stats.cards_due, 1); // New cards are immediately due

        // Record a correct answer → card advances to box 2
        deck.record_result(&card_id, true).unwrap();
        let stats = deck.retention_stats();
        assert_eq!(stats.total_cards, 1);
        // After answering correctly, card should no longer be due (next review in future)
        assert_eq!(stats.cards_due, 0);
    }
}
