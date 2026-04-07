---
description: Governance rules for working on ErnOSAgent — no artificial limits, no stubs, no fallbacks
---

# ErnOSAgent Governance Workflows

These rules govern how YOU (the coding agent) work on this project. These are NOT runtime code. They are your operational mandates.

## 1. No Hardcoded Limits

- **Context window**: Auto-derived from the model via the provider API. You NEVER define a context window value.
- **Temperature, top_k, top_p, num_predict**: Auto-derived from the model's reported defaults. You NEVER invent these values.
- **Rolling window size**: Computed as a function of the model's context_length. You NEVER set an arbitrary number.
- **Any model parameter**: If the model or provider reports it, that's what gets used. You do not override, cap, or "recommend" values.

## 2. No Safety Theatre

- You do not inject content filters the user did not request.
- You do not add output caps or sanitization layers.
- You do not add arbitrary character limits on tool outputs.
- You do not truncate responses to "fit" some imagined constraint.
- The model has its own parameters — those govern it.

## 3. No Stubs, Placeholders, or TODOs

- Every function you write does something real or it does not exist.
- No `unimplemented!()`, no `todo!()`, no empty function bodies.
- No `// TODO: implement later` comments.
- No "simplified for now" implementations.
- If you cannot implement something fully, you say so and stop. You do not leave a placeholder.

## 4. No Fallbacks

- If something fails, it fails cleanly and gracefully with a clear error message.
- No silent fallbacks that mask failures.
- No default values that silently replace failed operations.
- The ReAct loop has NO fallback — it exits ONLY via `reply_request` tool call.
- If a provider call fails, the error is displayed to the user. Not hidden behind a fallback.

## 5. No Arbitrary Truncation

- Message history is managed against the model's actual context_length.
- No artificial character caps on any output.
- No arbitrary rolling windows unless the model's context physically requires it.
- When context must be managed, the consolidation engine handles it.

## 6. Clean Error Handling

- Every error path produces a human-readable error message.
- Errors are logged with full context (module, function, relevant state).
- Errors are displayed in the TUI cleanly — no panics, no raw stack traces.
- `anyhow::Result` with context everywhere. No bare `.unwrap()`.

## 7. 100% Test Coverage

- Every module gets tests as it's built. Not after.
- Unit tests for every public function.
- Integration tests for cross-module interactions.
- Tests verify both success paths AND error paths.
- No test is a stub — every test asserts something meaningful.

## 8. Production Logging

- Every system has granular logging via `tracing`.
- Per-session rotating log files.
- Entry/exit logging for critical functions.
- Structured log fields (not string interpolation).
- Log levels used correctly: error, warn, info, debug, trace.

## 9. Auto-Derive Everything

- Model specs come from the provider, always.
- If a provider doesn't report a value, the system asks the user or reports the gap. It does NOT invent a default.
- The only exception is the embedding model name (configurable, defaults to `nomic-embed-text`).
