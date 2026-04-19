# Testing

## Running Tests

```bash
# Run all tests (unit + E2E + doc)
cargo test

# Run with output
cargo test -- --nocapture

# Run a specific test
cargo test test_name

# Run only E2E tests
cargo test --test e2e_tests

# Run only unit tests
cargo test --lib
```

## Test Coverage Summary

| Category | Count | Location |
|----------|-------|----------|
| Unit (inline) | 304 | `src/**/*.rs` — `#[cfg(test)] mod tests` blocks |
| E2E (integration) | 76 | `tests/e2e_tests.rs` |
| **Total** | **380** | |
| **Failures** | **0** | |

## Unit Test Distribution

Tests are embedded in each module via `#[cfg(test)]` blocks:

| Module | Tests |
|--------|-------|
| `tools` (schema, shell, browser, dispatch, search_providers) | 47 |
| `observer` (parser, rules, insights, skills) | 38 |
| `interpretability` (sae, features, trainer, trainer_tests, collector, etc.) | 24 |
| `platform` (adapter, registry, router, discord, discord_handler) | 22 |
| `memory` (mod, scratchpad, lessons, timeline, procedures, etc.) | 21 |
| `learning::lora` (training, weights, loss, loss_dpo, loss_kto, loss_simpo, optimizer, ewc, adapters) | 20 |
| `provider` (stream_parser, llamacpp, ollama, openai_compat) | 16 |
| `scheduler` (job, store) | 13 |
| `learning` (mod, buffers, sleep, teacher, observer_buffer) | 13 |
| `agents` | 12 |
| `prompt` (hud, tools) | 11 |
| `web::handlers` | 9 |
| `memory::synaptic` (mod, plasticity, query, relationships) | 9 |
| `learning::grpo` (generation, rewards, training) | 8 |
| `steering` (vectors, server) | 7 |
| `inference` (fast_reply, react_loop, sub_agent) | 7 |
| `web` (state, ws, tool_dispatch) | 4 |
| `session` | 4 |
| `model` | 2 |
| `config` | 2 |
| `logging` | 1 |

## E2E Test Suites (`tests/e2e_tests.rs`)

| Suite | Tests | What It Tests |
|-------|-------|---------------|
| `memory_manager_tests` | 8 | MemoryManager init, all 7 tiers, recall, status |
| `learning_e2e` | 9 | TrainingSample, LoRA config, training step, GRPO rewards, golden buffer, preference pairs |
| `memory_tools_e2e` | 6 | Memory, scratchpad, synaptic, timeline, lessons tool execution |
| `observer_parser_e2e` | 5 | Verdict parsing edge cases: clean JSON, markdown, embedded, garbage |
| `interpretability_e2e` | 5 | SAE encode, top features, divergence, snapshot capture |
| `state_tests` | 4 | AppState construction, shared state access patterns |
| `react_e2e` | 4 | ReactContext construction, tool result injection, rejection feedback |
| `full_pipeline_e2e` | 4 | End-to-end message → memory → recall → response |
| `inference_e2e` | 3 | Stream event enum construction, tool call detection |
| `observer_e2e` | 3 | Audit rules, verdict defaults, fail-open |
| `session_e2e` | 3 | Session CRUD, persistence, auto-title |
| `schema_e2e` | 3 | Tool schema generation, layer1/layer2 completeness |
| `tool_e2e` | 2 | Shell command execution, tool result format |
| `config_e2e` | 2 | Config defaults, TOML roundtrip |

## Testing Patterns

### MockProvider

E2E tests use a `MockProvider` (defined in `tests/e2e_tests.rs`) that implements `Provider` trait with:
- `health()` → always `true`
- `get_model_spec()` → returns a test ModelSpec
- `chat()` → returns a channel with a `Done` event
- `chat_sync()` → returns an approved verdict JSON
- `embed()` → returns a zero vector

### Temp Directories

All tests that touch disk create a `tempfile::TempDir` and pass it as the data directory. This ensures:
- No test pollutes the real data directory
- Tests are fully isolated from each other
- Cleanup is automatic when `TempDir` is dropped

### No Network

No tests require a running `llama-server`, Ollama, or any network service. All provider calls go through the `MockProvider`.

## Build Verification

```bash
# Release build (must compile with zero errors)
cargo build --release

# Release build (must compile with zero errors, zero warnings)
cargo build --release
```
