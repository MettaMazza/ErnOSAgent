# ErnOS Agent Governance & Rust Code Structure Workflow

> **Scope**: This file governs how any contributor (AI model or human) works on this project.
> These are NOT runtime code. They are operational mandates enforced with utmost scientific rigour.
> No regard for hack testing patterns. Every rule is load-bearing.
>
> **For contributors**: Read this entire file before opening a PR. Every rule exists because
> violating it has caused a production incident. If your change conflicts with any rule here,
> your change is wrong — not the rule.

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

The operational kernel is a single `const` string literal containing the full-depth system prompt. It is exempt from the 300-line ideal because it is **prompt text, not code logic** — splitting it across files would fragment the kernel's coherence with no structural benefit.

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

## 8. Anti-Pattern Catalogue

These are specific, named anti-patterns that **will cause your PR to be rejected**. Each one has been observed in practice and has caused production failures.

### 8.1 Reward Hacking

Doing the minimum to make something *appear* fixed without actually fixing the root cause.

- **Example**: A context overflow causes empty responses. The "fix" is to catch the empty response and return a canned error message instead. The overflow still happens — you've just wallpapered over it.
- **Example**: A function panics. The "fix" is to wrap it in `catch_unwind` and swallow the panic. The bug still exists — you've just silenced it.
- **Rule**: Every fix must address the **root cause**. If `catch_unwind` is added, it must be accompanied by diagnostic logging AND the underlying panic must be independently fixed or documented as a known limitation with a tracking issue.

### 8.2 Complexity Injection

Adding abstractions, wrappers, trait hierarchies, config layers, or architectural patterns that the codebase does not need.

- **Example**: Adding a `RetryPolicy` trait with 4 implementations and a builder pattern for a retry loop that only needs `for attempt in 0..=3`.
- **Example**: Creating an `ErrorKind` enum with 12 variants when `anyhow::Error` with `.context()` covers every case.
- **Example**: Adding a message bus / event system / plugin architecture when a function call is sufficient.
- **Rule**: The simplest implementation that fully solves the problem is the correct one. If you add an abstraction, you must demonstrate that at least 3 existing callsites benefit from it.

### 8.3 Heuristic Smuggling

Introducing magic numbers, estimation formulas, or rules-of-thumb disguised as proper implementations.

- **Example**: `let estimated_tokens = total_chars / 3` — this is a guess, not a measurement. The server has a tokenizer; use it.
- **Example**: `let budget = context_length - 2000` — where did 2000 come from? Nobody knows. It's a magic number.
- **Example**: `if content.len() > 50000 { truncate }` — 50000 is arbitrary. The model's actual context_length should govern this.
- **Rule**: If the system can measure a value, it must measure it. Heuristics are only acceptable when (a) the measurement API genuinely does not exist, AND (b) the heuristic is documented with its error margin, AND (c) the code is marked with a `// HEURISTIC:` comment explaining why the real measurement isn't available.

### 8.4 Investigation Theatre

Presenting a diagnostic report that lists every theoretically possible cause instead of identifying the actual one.

- **Example**: "The empty response could be caused by: context overflow, OR token limits, OR template expansion, OR server timeout, OR network error, OR..." — this is not an investigation, it's a brainstorm.
- **Rule**: Investigation means reading the logs, finding the specific error, tracing it to the specific line of code, and reporting the specific cause. If you cannot determine the cause, say "I was unable to determine the root cause" — do not present a list of guesses as analysis.

### 8.5 Test Theatre

Writing tests that pass but don't verify real behaviour.

- **Example**: A test that constructs a struct and asserts it exists. (`assert!(provider.id() == "ollama")` — this tests nothing meaningful.)
- **Example**: A test that mocks the entire system under test, then asserts the mock behaved as configured.
- **Example**: A test that verifies a function doesn't panic, but doesn't verify its output.
- **Rule**: Every test must assert a **behavioural contract**. Ask: "If I broke the implementation, would this test catch it?" If the answer is no, the test is theatre.

### 8.6 Shotgun Debugging

Making multiple speculative changes across multiple files in the hope that one of them fixes the problem.

- **Example**: Changing the retry count, the timeout, the error handling, AND the request format all in one commit because "one of these should fix it."
- **Rule**: One hypothesis, one change, one test. If the change doesn't fix it, revert it and try the next hypothesis. Never stack speculative changes.

### 8.7 Scope Creep

Adding features, refactors, or "improvements" that were not requested.

- **Example**: Asked to fix a panic in the SSE pipeline. The PR also refactors the logging format, renames 3 functions, and adds a new config option.
- **Rule**: A fix PR contains only the fix. A feature PR contains only the feature. A refactor PR contains only the refactor. Never mix concerns.

### 8.8 Silent State Mutation

Changing runtime behaviour through data files, config changes, or default values without corresponding code changes or documentation.

- **Example**: Changing `"enabled": true` to `"enabled": false` in a JSON data file to "fix" a misbehaving scheduler job.
- **Rule**: If a data file change alters system behaviour, it requires the same review rigour as a code change. The commit message must explain *why* the behaviour is changing.

---

## 9. Lifecycle Invariants

Some system behaviours are **not optional**. They are structural requirements for the engine to function. These must be hardcoded into the runtime lifecycle — never exposed as user-toggleable configuration.

### 9.1 What Is a Lifecycle Invariant?

A lifecycle invariant is any process where:
- The system **cannot function correctly** without it running.
- Disabling it causes **silent, cumulative degradation** (not an immediate, visible failure).
- A non-technical user would have **no reason to know** it needs to be enabled.

### 9.2 Current Lifecycle Invariants

| Invariant | Purpose | Failure Mode if Disabled |
|-----------|---------|-------------------------|
| `sleep_cycle` | Drain training buffers, run consolidation | Buffers grow unbounded, memory fills |
| `lesson_decay` | Hebbian forgetting on unused lessons | Stale lessons pollute recall context |
| `log_rotate` | Delete logs older than 7 days | Disk fills |
| `.env` load | Load API keys before subsystem init | All API-keyed services silently fail |
| `catch_unwind` on spawned tasks | Prevent silent task death | Messages silently fail with no error |
| Observer retry parity | `chat_sync` retries match `chat()` | Audit fails on transient network blips |

### 9.3 Rule: Optional vs. Invariant

- If disabling something **breaks the system silently**, it is an invariant. Hardcode it.
- If disabling something **removes a user-facing feature cleanly**, it is optional. Put it in config.
- **Never** put an invariant behind a toggle. If you find one, move it out.

---

## 10. Contribution Protocol

### 10.1 The Root Cause Mandate

Before proposing any fix:

1. **Read the logs.** Identify the specific error message and timestamp.
2. **Trace the call chain.** Find the exact function and line where the failure originates.
3. **Identify the root cause.** Not the symptom, not a list of possibilities — the actual cause.
4. **Verify your hypothesis.** If you can't reproduce or verify, say so explicitly.
5. **Then, and only then, propose a fix.** The fix must address the root cause identified in step 3.

### 10.2 One Concern Per PR

- A PR fixes **one bug**, adds **one feature**, or performs **one refactor**.
- If a fix reveals a second issue, file it separately.
- If a feature requires a refactor, the refactor is a separate PR that lands first.
- Commit messages follow conventional commits: `fix:`, `feat:`, `refactor:`, `chore:`, `test:`.

### 10.3 Data Directory Protection

The `data/` directory contains runtime state: sessions, memory databases, training buffers, scheduler history, and logs.

- `data/` is in `.gitignore` and must stay there.
- PRs must never include files from `data/`.
- No code may assume `data/` contains specific files at startup — it must create what it needs.
- Destructive operations on `data/` (delete, overwrite, migrate) require explicit user confirmation or a migration script with rollback.

### 10.4 Dependency Discipline

- New dependencies require justification. "It makes X easier" is not sufficient — explain what is impossible or unsafe without it.
- No dependencies that pull in a web framework, ORM, or runtime we don't already use.
- Prefer `std` library solutions. Prefer well-maintained, single-purpose crates over kitchen-sink frameworks.
- Pin major versions. No `*` or `>=` version specs.

---

## 11. Review Rejection Criteria

A PR is **immediately rejected** if it contains any of the following. No discussion, no exceptions:

| # | Rejection Trigger |
|---|-------------------|
| R1 | Hardcoded model parameter (context length, temperature, token limit) |
| R2 | `todo!()`, `unimplemented!()`, `// TODO`, or empty function body |
| R3 | `unwrap()` on a `Result` or `Option` outside of tests |
| R4 | Silent fallback that masks a failure (returns default instead of error) |
| R5 | Heuristic without `// HEURISTIC:` comment and documented error margin |
| R6 | Test that doesn't assert a behavioural contract |
| R7 | Multiple unrelated concerns in a single PR |
| R8 | Model-specific code path (`if model.contains("gemma")`) |
| R9 | Provider-specific logic outside `src/provider/<name>.rs` |
| R10 | Missing `//!` doc comment on a new module |
| R11 | Function exceeding 50 lines without documented justification |
| R12 | Lifecycle invariant exposed as a toggleable config option |
| R13 | Magic number without derivation comment |
| R14 | `catch_unwind` without accompanying diagnostic logging |
| R15 | Data file change without commit message explaining the behavioural impact |

---

## 12. Architectural Violations — Historical Record

These are real incidents from this project's history. They document *why* specific rules exist. This section is append-only — new incidents are added, old ones are never removed.

### V1: The SSE Silent Death (April 2026)

**What happened**: The SSE streaming pipeline used `tokio::spawn` without `catch_unwind`. When context assembly panicked, the spawned task died silently. The channel closed. The client received an empty SSE stream. Discord showed no response. No error was logged.

**Impact**: 100% failure rate on all incoming Discord messages for 8+ days.

**Root cause**: Unhandled panic in a fire-and-forget spawned task.

**Rule created**: §9.2 — `catch_unwind` on spawned tasks is a lifecycle invariant. §8.1 — `catch_unwind` alone is reward hacking; diagnostic logging must accompany it.

### V2: The chars/3 Heuristic (April 2026)

**What happened**: Token estimation used `total_chars / 3` as a "conservative BPE estimate". Jinja2 template expansion, tool definitions, and system prompts added thousands of tokens that this heuristic couldn't see. Consolidation failed to trigger. Context overflowed. Model returned empty responses.

**Impact**: Silent generation failures on long conversations.

**Root cause**: Using a character-based heuristic instead of the server's actual tokenizer.

**Rule created**: §8.3 — Heuristic Smuggling. §2.1 — No hardcoded limits. The server has a tokenizer; use it.

### V3: The Toggleable Lifecycle Invariant (May 2026)

**What happened**: `sleep_cycle`, `lesson_decay`, and `log_rotate` were entries in `scheduler.json` with an `"enabled": bool` field. Something set them to `false`. For 8 days, training buffers accumulated unbounded and logs were never rotated. Nobody noticed because the system didn't crash — it silently degraded.

**Impact**: Unbounded memory growth, disk fill risk, stale lesson pollution.

**Root cause**: System-critical processes exposed as user-toggleable optional jobs.

**Rule created**: §9 — Lifecycle Invariants. If disabling it breaks the system silently, it must not be toggleable.

### V4: The Observer Fail-Open (April–May 2026)

**What happened**: The Observer audit used `chat_sync`, which had no retry logic. A transient connection reset during audit caused the observer to fail. The system defaulted to passing the unaudited response through — a fail-open security posture.

**Impact**: Unaudited responses delivered to users during network instability.

**Root cause**: `chat_sync` lacked the retry parity that `chat()` already had.

**Rule created**: §9.2 — Observer retry parity is a lifecycle invariant. All Provider trait methods that can fail transiently must implement the same retry policy.

---

## Summary

This workflow is enforced on **every** file touch, **every** code review, and **every** new module. There are no exceptions unless explicitly documented above (e.g., the operational kernel exemption).

**For AI models**: These rules are not suggestions. They are constraints. If your proposed change violates any rule in this document, the change is wrong. Do not argue, do not propose exceptions, do not "improve" the governance. Fix your change.

**For human contributors**: Thank you for contributing. Please read §8 (Anti-Patterns) and §11 (Rejection Criteria) before your first PR. Every rule here was written in response to a real production incident documented in §12.

Scientific rigour, not shortcuts.
