# ErnOS Agent Governance & Rust Code Structure Workflow

> **Scope**: This file governs how any coding agent (AI or human) works on this project.
> These are NOT runtime code. They are operational mandates enforced with utmost scientific rigour.
> No regard for hack testing patterns. Every rule is load-bearing.

---

## 1. Rust Code Size & Structure Guidelines

These are the **mandatory** structural limits for all Rust source files in this project. Apply these rules when creating new files, modifying existing files, or conducting code reviews.

### 1.1 File Length Limits

| Range | Verdict | Action |
|-------|---------|--------|
| **~100–300 lines** | ✅ Ideal | Easy to reason about, easy to test |
| **~300–500 lines** | ⚠️ Acceptable | Only if the module has a clear single purpose |
| **~500+ lines** | 🔴 Split required | Must be refactored into smaller modules |

#### Exception: Operational Kernel (`src/prompt/core.rs`)

The operational kernel is a single `const` string literal containing the full-depth system prompt. It is exempt from the 300-line ideal because it is **prompt text, not code logic** — splitting it across files would fragment the kernel's coherence with no structural benefit. The file remains under 500 lines.

#### How to Split a File Over 500 Lines

1. Identify distinct responsibilities within the file
2. Extract each responsibility into its own submodule file
3. Keep the original file as a thin orchestrator that re-exports and delegates
4. Move tests into a dedicated `tests.rs` sibling file if they exceed ~100 lines

**Example split for `app.rs` (673 lines):**

```
src/app.rs          (200 lines — orchestrator only)
src/ui/tui.rs       (new — TUI rendering)
src/ui/keybindings.rs (new — key event handling)
```

---

### 1.2 Function Length Limits

| Range | Verdict | Action |
|-------|---------|--------|
| **~10–30 lines** | ✅ Sweet spot | Clear, testable, single-purpose |
| **~30–50 lines** | ⚠️ Review | Consider extracting helper functions |
| **~50+ lines** | 🔴 Refactor | Must be broken into smaller functions |

#### The "And" Test

If you can't describe what a function does **without using the word "and"**, it is doing too much and must be split.

- **Bad:** "This function parses the config **and** validates it **and** creates the directories."
- **Good:** "This function parses the config file into an `AppConfig` struct."

#### Common Extraction Patterns

- Extract validation logic into a `validate_*()` function
- Extract IO operations into a `read_*()` / `write_*()` function
- Extract transformation logic into a `transform_*()` / `build_*()` function
- Extract error handling into a `try_*()` wrapper

---

### 1.3 Struct/Trait Complexity Limits

| Range | Verdict | Action |
|-------|---------|--------|
| **~5–10 methods** per `impl` block | ✅ Comfortable | Well-focused responsibility |
| **~10–15 methods** | ⚠️ Review | Consider splitting into traits or helper modules |
| **~15+ methods** | 🔴 Refactor | The struct is wearing too many hats |

#### How to Split a Large `impl` Block

1. **Group methods by concern** — identify clusters of methods that serve the same sub-purpose
2. **Extract traits** — if a subset of methods represents a distinct capability, define a trait
3. **Extract helper structs** — if methods share state that the parent struct doesn't need, create a child struct
4. **Use extension traits** — for optional/secondary methods, define them in a separate `impl` block in another file

**Example: `MemoryManager` (14+ methods):**

```rust
// memory/manager.rs — core lifecycle
impl MemoryManager {
    pub fn new() -> Self { ... }
    pub fn status_summary() -> String { ... }
    pub fn kg_available() -> bool { ... }
}

// memory/ingest.rs — data ingestion concern
impl MemoryManager {
    pub fn ingest_turn() { ... }
    pub fn ingest_lesson() { ... }
}

// memory/recall.rs — retrieval concern
impl MemoryManager {
    pub fn recall_context() { ... }
    pub fn search_timeline() { ... }
}
```

---

### 1.4 Audit Checklist

When modifying or reviewing **any** Rust file, verify:

- [ ] File is under 500 lines (excluding tests)
- [ ] No function exceeds 50 lines
- [ ] No `impl` block has more than 15 methods
- [ ] Each function passes the "and" test
- [ ] Tests are proportional (1 test per public function **minimum**)
- [ ] Module has a clear single purpose described in the `//!` doc comment

---

## 2. Governance Mandates

These rules govern how the coding agent operates on this project. They are **non-negotiable**.

### 2.1 No Hardcoded Limits

- **Context window**: Auto-derived from the model via the provider API. NEVER define a context window value.
- **Temperature, top_k, top_p, num_predict**: Auto-derived from the model's reported defaults. NEVER invent these values.
- **Rolling window size**: Computed as a function of the model's `context_length`. NEVER set an arbitrary number.
- **Any model parameter**: If the model or provider reports it, that's what gets used. No overrides, no caps, no "recommendations".

### 2.2 No Safety Theatre

- Do not inject content filters the user did not request.
- Do not add output caps or sanitisation layers.
- Do not add arbitrary character limits on tool outputs.
- Do not truncate responses to "fit" some imagined constraint.
- The model has its own parameters — those govern it.

### 2.3 No Stubs, Placeholders, or TODOs

- Every function does something real or it does not exist.
- No `unimplemented!()`, no `todo!()`, no empty function bodies.
- No `// TODO: implement later` comments.
- No "simplified for now" implementations.
- If you cannot implement something fully, **say so and stop**. Do not leave a placeholder.

### 2.4 No Fallbacks

- If something fails, it fails cleanly and gracefully with a clear error message.
- No silent fallbacks that mask failures.
- No default values that silently replace failed operations.
- The ReAct loop has NO fallback — it exits ONLY via `reply_request` tool call.
- If a provider call fails, the error is displayed to the user. Not hidden behind a fallback.
- If something is optional, failure falls back to **off** — the feature is disabled, not silently degraded.

### 2.5 No Arbitrary Truncation

- Message history is managed against the model's actual `context_length`.
- No artificial character caps on any output.
- No arbitrary rolling windows unless the model's context physically requires it.
- When context must be managed, the consolidation engine handles it.

### 2.6 Clean Error Handling

- Every error path produces a human-readable error message.
- Errors are logged with full context (module, function, relevant state).
- Errors are displayed in the TUI cleanly — no panics, no raw stack traces.
- `anyhow::Result` with context everywhere. No bare `.unwrap()`.

### 2.7 Everything On by Default

- No optional feature flags for core capabilities.
- Everything ships **on by default**.
- If anything fails, it fails **loud** — never silently degrades.
- If something is genuinely optional and fails, it falls back to **off** (disabled entirely), not to a degraded alternative.

---

## 3. Testing Mandates

### 3.1 100% Test Coverage

- Every module gets tests **as it's built**. Not after.
- Unit tests for every public function.
- Integration tests for cross-module interactions.
- End-to-end tests for every single feature and module.
- Tests verify both **success paths AND error paths**.
- No test is a stub — every test asserts something meaningful.
- No hardcoded heuristics in tests — tests validate real behaviour.

### 3.2 Test Structure

```
tests/
├── unit/           # Per-module unit tests
├── integration/    # Cross-module interaction tests
└── e2e/            # Full-system end-to-end tests
```

- Tests exceeding ~100 lines in a module file must be extracted to a sibling `tests.rs`.
- Test names describe the scenario: `test_ingest_turn_rejects_empty_input`, not `test_1`.

---

## 4. Production Logging

- Every system has granular logging via `tracing`.
- Per-session rotating log files.
- Entry/exit logging for critical functions.
- Structured log fields (not string interpolation).
- Log levels used correctly:
  - `error` — system cannot continue this operation
  - `warn` — something unexpected, but recoverable
  - `info` — significant lifecycle events
  - `debug` — detailed operational state
  - `trace` — fine-grained diagnostics

---

## 5. Auto-Derive Everything

- Model specs come from the provider, always.
- If a provider doesn't report a value, the system **asks the user** or **reports the gap**. It does NOT invent a default.
- The only exception is the embedding model name (configurable, defaults to `nomic-embed-text`).

---

## 6. WebUI-Centric Architecture

The WebUI is the **single central hub** of the entire system. Every integration — current and future — connects to and operates through the WebUI. This is a non-negotiable architectural mandate.

### 6.1 The WebUI Is the Engine's Front Door

- The WebUI owns the WebSocket and REST API surface.
- All external consumers (Discord bots, Telegram adapters, mobile apps, CLI tools, third-party integrations) connect to the WebUI as **clients**, not directly to the inference engine.
- The inference engine, observer, tool executor, and session manager are internal services that the WebUI orchestrates.

```
┌─────────────────────────────────────────────────┐
│                   WebUI (Hub)                   │
│  ┌──────────┐  ┌───────┐  ┌──────────────────┐  │
│  │ REST API │  │  WS   │  │ Static Frontend  │  │
│  └────┬─────┘  └───┬───┘  └──────────────────┘  │
│       │            │                             │
│  ┌────┴────────────┴─────────────────────────┐   │
│  │         Internal Engine Services          │   │
│  │  Inference · Observer · Tools · Sessions  │   │
│  └───────────────────────────────────────────┘   │
└────────────────┬────────────────────────────────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
┌───┴───┐  ┌────┴────┐  ┌────┴───┐
│Discord│  │Telegram │  │Mobile  │
│ Bot   │  │ Adapter │  │  App   │
└───────┘  └─────────┘  └────────┘
    (all connect as WebSocket/REST clients)
```

### 6.2 Why This Matters

- **Single point of update**: Changes to the API, authentication, rate limiting, session management, or streaming protocol happen once in the WebUI and automatically apply to every connected platform.
- **No spaghetti integrations**: Platform adapters never import engine internals. They speak the WebUI's public API. Period.
- **Scalability**: New platforms are added by writing a thin adapter that connects to the existing WebUI API. Zero engine changes required.
- **Testability**: The entire system can be tested through the WebUI's API surface without any platform adapter installed.

### 6.3 Rules for Platform Adapters

1. A platform adapter is a **standalone client** that connects to the WebUI via WebSocket or REST.
2. It translates platform-specific messages (Discord events, Telegram updates, etc.) into the WebUI's message protocol.
3. It translates WebUI responses back into platform-specific formats.
4. It does **NOT** import, depend on, or call any internal engine module directly.
5. It can live in the same binary or as a separate process — the architecture supports both.

### 6.4 Rules for the WebUI API

1. The WebUI API is the **contract** between the engine and the outside world.
2. Any capability exposed to the frontend must be exposed via the same API endpoints that adapters use.
3. No backdoor functions, no special internal-only routes that bypass the API.
4. The WebSocket protocol is documented and versioned.

---

## 7. Platform, Model & Hardware Neutrality

The system must make **zero assumptions** about the platform, model, or hardware it runs on. This is enforced at every layer.

### 7.1 Model Neutrality

- The engine works with **any** model served via an OpenAI-compatible API.
- No model-family-specific code paths (no `if model.contains("gemma")`, no `if model.contains("llama")`).
- Model capabilities (vision, audio, tool calling) are **discovered** via the provider API, never assumed.
- Prompt formatting is handled by the provider/server (llama-server applies chat templates natively). The engine sends raw messages.
- If a model doesn't support a feature (e.g., vision), the system disables that input pathway cleanly — it does not crash.

### 7.2 Provider Neutrality

- The `Provider` trait is the universal interface. Every backend implements it identically.
- Provider selection is a config value, not a compile-time decision.
- The active provider can be changed at runtime without restarting the engine.
- No provider-specific logic leaks into the inference engine, observer, tools, or WebUI.
- Provider-specific code lives **only** inside `src/provider/<name>.rs`.

### 7.3 Hardware Neutrality

- The engine runs on **any** hardware: Apple Silicon, NVIDIA, AMD, CPU-only.
- GPU acceleration is the provider's responsibility (llama-server handles Metal/CUDA/ROCm/Vulkan). The engine makes no GPU calls.
- No conditional compilation based on hardware (`#[cfg(target_os)]` is allowed only for OS-specific filesystem paths or browser-open commands).
- Memory management decisions (batch size, context length) come from the model's reported specs, not from hardware detection.

### 7.4 Operating System Neutrality

- The engine compiles and runs on macOS, Linux, and Windows.
- OS-specific code is isolated behind helper functions (e.g., `open_browser()`).
- File paths use `std::path::PathBuf`, never hardcoded separators.
- Process management uses `tokio::process`, which is cross-platform.

---

## Summary

This workflow is enforced on **every** file touch, **every** code review, and **every** new module. There are no exceptions unless explicitly documented above (e.g., the operational kernel exemption). Scientific rigour, not shortcuts.
