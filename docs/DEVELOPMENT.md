# Development Guide

How to build, test, extend, and contribute to ErnOSAgent.

---

## Build

### Standard Build

```bash
cargo build --release
```

### Feature-Gated Builds

```bash
# With Discord bot support
cargo build --release --features discord

# With audio input (requires local Whisper model)
cargo build --release --features audio

# Everything
cargo build --release --features all-platforms,audio
```

### From-Source llama-server

The Homebrew `llama.cpp` package may lag behind for new model architectures (Gemma 4, Qwen 3.5). Build from source:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp
cd /tmp/llama.cpp
cmake -B build -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j $(sysctl -n hw.ncpu) -t llama-server
```

Binary: `/tmp/llama.cpp/build/bin/llama-server`

---

## Testing

### Unit Tests

```bash
# All 645 unit tests (~1.3s)
cargo test --lib

# Specific module
cargo test --lib memory::
cargo test --lib provider::llamacpp::
cargo test --lib observer::
```

### Integration Tests

```bash
# All e2e tests (requires llama-server + GGUF model)
cargo test --test e2e_llama -- --nocapture --test-threads=1

# Skip Neo4j-dependent tests
cargo test --test e2e_llama -- --nocapture --test-threads=1 --skip neo4j

# Single e2e test
cargo test --test e2e_llama test_e2e_raw_inference -- --nocapture
```

**Prerequisites for e2e tests:**

| Requirement | Default Path | Override |
|------------|-------------|---------|
| llama-server binary | `/tmp/llama.cpp.build/build/bin/llama-server` | Edit `LLAMA_SERVER_BIN` in `tests/e2e_llama.rs` |
| GGUF model | `./models/gemma-4-26b-it-Q4_K_M.gguf` | Edit `MODEL_GGUF` in `tests/e2e_llama.rs` |
| Neo4j (optional) | `bolt://localhost:7687` | Start Neo4j Desktop or Docker |

### Test Architecture

```
tests/e2e_llama.rs
├── test_e2e_model_spec_derivation  — Server → health → /v1/models → /props
├── test_e2e_raw_inference          — Full SSE streaming pipeline
├── test_e2e_react_pipeline         — ReAct loop with tool detection
└── test_e2e_neo4j_knowledge_graph  — Neo4j CRUD + memory manager
```

Each test spawns its own llama-server on a unique port (8199, 8200, 8201) and cleans up via RAII (`TestServer` implements `Drop`).

---

## Adding a New Tool

### 1. Define the Tool

Create the tool definition in `src/tools/` or inline:

```rust
use crate::provider::ToolDefinition;

pub fn my_tool() -> ToolDefinition {
    ToolDefinition {
        r#type: "function".to_string(),
        function: serde_json::json!({
            "name": "my_tool",
            "description": "What this tool does",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query"
                    }
                },
                "required": ["query"]
            }
        }),
    }
}
```

### 2. Implement Execution

In `src/tools/executor.rs`, add a handler:

```rust
impl ToolExecutor {
    pub async fn execute(&self, call: &ToolCall) -> ToolResult {
        match call.name.as_str() {
            "reply_request" => { /* handled by ReAct loop */ }
            "my_tool" => self.execute_my_tool(call).await,
            _ => ToolResult {
                tool_call_id: call.id.clone(),
                name: call.name.clone(),
                output: String::new(),
                success: false,
                error: Some(format!("Unknown tool: {}", call.name)),
            },
        }
    }

    async fn execute_my_tool(&self, call: &ToolCall) -> ToolResult {
        let query = call.arguments["query"].as_str().unwrap_or("");
        // ... implementation ...
        ToolResult {
            tool_call_id: call.id.clone(),
            name: "my_tool".to_string(),
            output: "result here".to_string(),
            success: true,
            error: None,
        }
    }
}
```

### 3. Register the Tool

In `src/app.rs`, add it to the tools list:

```rust
let tools = vec![
    reply::reply_request_tool(),
    my_module::my_tool(),
];
```

---

## Adding a New Provider

### 1. Implement the `Provider` Trait

Create `src/provider/my_provider.rs`:

```rust
use crate::provider::*;

pub struct MyProvider { /* ... */ }

#[async_trait]
impl Provider for MyProvider {
    fn id(&self) -> &str { "myprovider" }
    fn display_name(&self) -> &str { "My Provider" }

    async fn chat(&self, model: &str, messages: &[Message],
                  tools: Option<&[ToolDefinition]>,
                  tx: mpsc::Sender<StreamEvent>) -> Result<()> {
        // Stream tokens via tx.send(StreamEvent::Token(...))
        // Parse tool calls via tx.send(StreamEvent::ToolCall {...})
        // Signal completion via tx.send(StreamEvent::Done {...})
    }

    // ... implement remaining trait methods
}
```

### 2. Register in Config

Add the provider's config struct to `src/config.rs` and add it to `AppConfig`.

### 3. Wire into App

In `src/app.rs`, match on the provider ID to create the correct instance.

---

## Adding a New Memory Tier

### 1. Create the Module

Create `src/memory/my_tier.rs` with a struct that handles persistence.

### 2. Wire into MemoryManager

In `src/memory/mod.rs`:

```rust
pub struct MemoryManager {
    // ... existing tiers ...
    pub my_tier: MyTier,
}

impl MemoryManager {
    pub async fn recall_context(&self, query: &str, budget: usize) -> String {
        // ... add your tier's contribution to the context ...
    }
}
```

### 3. Add Budget Allocation

Adjust the budget percentages in `recall_context()` to include your tier.

---

## Adding a New Observer Rule

### 1. Add to the Checklist

In `src/observer/rules.rs`, add rule #17 to `AUDIT_RULES`:

```
17. **MY_NEW_RULE** — Description of what this catches.
```

### 2. Add to the Name List

```rust
pub const RULE_NAMES: &[&str] = &[
    // ... existing 16 ...
    "my_new_rule",
];
```

### 3. Update Tests

Update `test_audit_rules_contains_all_16` → `test_audit_rules_contains_all_17` and `test_rule_names_count` → `assert_eq!(RULE_NAMES.len(), 17)`.

---

## Code Conventions

### Error Handling

- Use `anyhow::Result` for fallible functions
- Use `thiserror` for domain-specific error types
- Never `unwrap()` in library code (use `expect()` with a clear message only in binary entry points)
- Prefer `.context("what was happening")` over bare `?`

### Logging

```rust
use tracing::{info, warn, error, debug, instrument};

#[instrument(skip(self))]
async fn my_method(&self, query: &str) -> Result<()> {
    info!(query = %query, "Starting operation");
    // ...
    warn!("Something unexpected but recoverable");
    error!("Something broken");
}
```

### Async

- All I/O-bound operations are `async`
- CPU-bound work uses `tokio::task::spawn_blocking`
- Channels (`mpsc`) for streaming events
- No `futures::executor::block_on` inside async code (deadlock risk)

### Testing

- Every public method has at least one unit test
- Tests use `tempfile::TempDir` for filesystem isolation
- Async tests use `#[tokio::test]`
- E2E tests implement RAII cleanup (`Drop` trait)

---

## Debugging

### Trace Logging

```bash
RUST_LOG=debug cargo run 2>&1 | less
RUST_LOG=ernosagent::react=trace cargo run  # Just ReAct loop
RUST_LOG=ernosagent::provider=debug cargo run  # Provider only
```

### Manual Inference Test

```bash
# Start server manually
llama-server --model models/gemma-4-26b-it-Q4_K_M.gguf --port 8199 --ctx-size 8192 --n-gpu-layers -1

# Test health
curl localhost:8199/health

# Test chat
curl localhost:8199/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"max_tokens":32}'

# Test streaming
curl -N localhost:8199/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

### Neo4j

```bash
# Start Neo4j (Docker)
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/ernosagent neo4j:latest

# Browser UI
open http://localhost:7474

# Cypher queries
MATCH (n:Entity) RETURN n LIMIT 10;
MATCH (a)-[r]->(b) RETURN a.label, type(r), b.label, r.weight;
```

---

## Known Gotchas

### Ollama GGUFs Are Incompatible

Ollama stores models as split GGUF blobs under `~/.ollama/models/blobs/`. These are NOT loadable by standalone `llama-server`. Always download complete GGUF files from HuggingFace.

### `block_on` Inside Async = Deadlock

Never use `futures::executor::block_on()` inside a tokio async context. This was a real bug (fixed in `LlamaCppProvider::get_model_spec`). Use `.await` instead.

### Metal Library Load Time

On macOS, the first `llama-server` start loads the embedded Metal shader library, which takes ~8 seconds. Subsequent starts are faster. Account for this in health check timeouts.

### Gemma 4 Reasoning Content

Gemma 4 models emit `reasoning_content` (thinking tokens) before their response. These appear as `StreamEvent::Thinking` events and are NOT part of the final response. The TUI displays them with a 💭 prefix.
