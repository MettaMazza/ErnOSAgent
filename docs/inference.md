# Inference Pipeline

The inference engine uses a **dual-layer architecture** defined across `src/inference/`.

## Module Map

| File | Purpose |
|------|---------|
| `mod.rs` | Module declarations |
| `router.rs` | Routes incoming messages to the correct layer |
| `fast_reply.rs` | Layer 1 fast reply logic |
| `react_loop.rs` | Layer 2 ReAct agentic loop |
| `react_observer.rs` | Observer integration within the ReAct loop |
| `sub_agent.rs` | Sub-agent isolation and execution |

## Layer 1: Fast Reply

The default path for simple interactions. The provider receives:
- System prompt with memory context (including learned skills)
- User message
- `layer1_tools` schema (18 tools: `start_react_system`, `propose_plan`, `run_bash_command`, `web_search`, `file_read`, `file_write`, `codebase_search`, `browser`, `memory`, `scratchpad`, `timeline`, `lessons`, `create_artifact`, `generate_image`, `steering`, `interpretability`, `learning`, `system_logs`)

**Outcomes** (determined by `consume_silently()` in `ws.rs`):

| Result | Action |
|--------|--------|
| `ConsumeResult::Reply` | Stream text to client, run observer audit, archive |
| `ConsumeResult::ToolCall("start_react_system")` | Escalate to Layer 2 |
| `ConsumeResult::ToolCall(other)` | Execute tool, inject result, re-infer |
| `ConsumeResult::Error` | Send error to client |

## Layer 2: ReAct Loop

Activated when the model calls `start_react_system` with an `objective` and optional `plan`. Defined in `src/inference/react_loop.rs`.

### ReactContext

```rust
pub struct ReactContext {
    pub objective: String,
    pub messages: Vec<Message>,
    pub tool_results: Vec<ToolResult>,
    pub iteration: usize,
}
```

- `new(objective, plan, base_messages)` — injects the ReAct system prompt
- `add_tool_result(result)` — appends tool output, increments iteration
- `add_rejection_feedback(reason, guidance)` — injects observer rejection as a system message

### IterationResult

Each call to `run_iteration()` returns one of:

| Variant | Description |
|---------|-------------|
| `Reply(text)` | Model called `reply_request` tool — final answer |
| `Refuse(reason)` | Model called `refuse_request` tool — explicit refusal |
| `ToolCall(tc)` | Model wants to execute a tool |
| `ImplicitReply(text)` | Model responded with plain text (no tool call) |

### Loop Flow

```
for iteration in 0..20 {
    match run_iteration(provider, &ctx, thinking) {
        Reply(text)     → observer audit → deliver → return
        Refuse(reason)  → deliver refusal → return
        ToolCall(tc)    → execute → ctx.add_tool_result() → continue
        ImplicitReply   → deliver → return
        Error           → report → return
    }
}
// Exceeded 20 iterations → error
```

The loop is capped at 20 iterations. The only exit via agentic path is `reply_request` or `refuse_request` tool calls.

## Observer Audit

Defined in `src/observer/`. Runs after every response in both layers.

### Audit Flow

1. `audit_response(provider, conversation, response)` sends a non-streaming request to the same provider with 1-to-1 context parity
2. Provider returns a JSON verdict
3. `parser::parse_verdict(text)` extracts the verdict from raw text (handles embedded JSON, markdown blocks, garbage)

### Verdict Structure

```rust
pub struct Verdict {
    pub approved: bool,
    pub score: f64,
    pub reason: Option<String>,
    pub guidance: Option<String>,
}
```

### Fail-Open Policy

If the observer response is not valid JSON, the verdict defaults to `approved: true` with `score: 5.0`. This prevents the observer from blocking the pipeline on parse failures.

### Training Signal Capture

After each verdict, `src/web/training_capture.rs` captures training data:
- **Approved** → `capture_approved()` → GoldenBuffer (SFT)
- **Rejected + retried** → `capture_rejection()` → RejectionBuffer (DPO preference pairs)

### Automatic Insight Extraction

After each completed exchange, `src/observer/insights.rs` runs background insight extraction. High-confidence insights (≥0.7) are stored in the Lessons memory tier with deduplication.

### Delayed Reinforcement

After tool chains complete in Layer 1, the chain (tools used, query, reply) is stashed in a per-connection `PendingToolChain`. On the NEXT user message, `classify_user_feedback()` analyses the message for implicit signals:

- **Approved** ("great", "thanks", "now...", "continue") → auto-creates a procedure via `procedures.add_if_new()`, bumps `success_count`, adds to golden buffer at 0.85 quality
- **Rejected** ("wrong", "not what I asked", "try again") → adds to rejection buffer as a preference pair
- **Neutral** (ambiguous) → no action

This runs in a background `tokio::spawn` — zero latency impact on the user's response.

### Observer Rules

`src/observer/rules.rs` defines audit rules applied to the prompt sent to the observer model. These rules govern what the observer checks for (relevance, completeness, accuracy).

## Tool Schema

Layer 1 and Layer 2 use different tool sets, defined in `src/tools/schema.rs`:

| Function | Tools Included |
|----------|---------------|
| `layer1_tools()` | `start_react_system`, `propose_plan`, `run_bash_command`, `web_search`, `file_read`, `file_write`, `codebase_search`, `browser`, `memory`, `scratchpad`, `timeline`, `lessons`, `create_artifact`, `generate_image`, `steering`, `interpretability`, `learning`, `system_logs` (18 tools) |
| `layer2_tools()` | `reply_request`, `refuse_request`, `extend_turns`, `run_bash_command`, `web_search`, `memory`, `scratchpad`, `synaptic`, `timeline`, `lessons`, `self_skills`, `learning`, `steering`, `interpretability`, `codebase_search`, `file_read`, `file_write`, `browser`, `create_artifact`, `generate_image`, `spawn_sub_agent`, `codebase_edit`, `system_recompile`, `checkpoint`, `system_logs` (25 tools) |

Tool calls from the model are dispatched by `tool_dispatch.rs` (for state-dependent tools) or `executor.rs` (for stateless tools like shell commands).
