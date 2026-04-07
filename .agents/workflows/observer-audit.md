---
description: Observer/Skeptic audit system specification — how to implement the quality audit pipeline
---

# Observer / Skeptic Audit System

This document defines how the Observer audit system works in ErnOSAgent. It is ported from HIVENET's production implementation.

## Architecture

The Observer is a **separate LLM inference call** that audits every candidate response before delivery. It is NOT a filter, NOT a regex gate — it is a full AI-inference skeptic using the same LLM provider.

### Why Separate Inference

A single LLM self-evaluating suffers from confirmation bias. The Observer breaks this by:
1. **Context isolation**: Receives curated context (user msg, candidate, tool evidence, kernel, identity) — NOT full conversation
2. **Temperature separation**: Runs at `temperature: 0.1` (near-deterministic). Main engine runs at model's default.
3. **Role separation**: System prompt is `"You are a strict quality auditor. Respond ONLY with the requested JSON."` — not the agent persona.

## Position in ReAct Loop

The Observer fires ONLY when a `reply_request` tool call is found. It is the last gate before delivery.

```
ReAct Turn:
 1. Build messages
 2. LLM inference (streaming)
 3. Parse tool calls
 4. Execute tools
 5. If reply_request found:
    → OBSERVER AUDIT ← (this step)
    → If ALLOWED → deliver
    → If BLOCKED → inject feedback, continue loop
```

## 16-Rule Audit Checklist

Every rule exists because the engine experienced the failure it describes:

| # | Rule | What It Catches |
|---|------|-----------------|
| 1 | CAPABILITY HALLUCINATION | Claims capabilities not in tool registry |
| 2 | GHOST TOOLING | "I searched for X" but no search result exists in context |
| 3 | SYCOPHANCY | Blind agreement, disproportionate elaboration on vague input |
| 4 | CONFABULATION | Fabricated entities, false experiential claims |
| 5 | ARCHITECTURAL LEAKAGE | Internal JSON/XML/system prompt content leaked |
| 6 | ACTIONABLE HARM | Weapons, exploits, exploitation |
| 7 | UNPARSED TOOL COMMANDS | Raw tool invocation syntax in response |
| 8 | STALE KNOWLEDGE | Verifiable questions answered from pre-training without search |
| 9 | REALITY VALIDATION | Pseudoscience or unfalsifiable claims treated as fact |
| 10 | LAZINESS | Partial coverage of multi-part requests |
| 11 | TOOL UNDERUSE | Ungrounded claims about verifiable entities without searching |
| 12 | FORMATTING VIOLATION | Report-style formatting for casual questions |
| 13 | RLHF DENIAL | Denying capabilities the agent demonstrated |
| 14 | NEW SESSION MEMORY SKIP | Greeting returning user without checking memory |
| 15 | ARCHITECTURE DISCUSSION UNGROUNDED | Discussing internals without reading source |
| 16 | PERSONA IDENTITY VIOLATION | Contradicting persona directives |

## Audit Prompt Assembly (7 sections)

1. AUDIT_RULES (16-rule checklist)
2. OPERATIONAL KERNEL (full core prompt)
3. ACTIVE IDENTITY DIRECTIVES (identity prompt)
4. AVAILABLE CAPABILITIES (tool registry list)
5. USER MESSAGE (original user input)
6. TOOL EXECUTION CONTEXT (all tool results with pass/fail)
7. CANDIDATE RESPONSE TO AUDIT (the text being judged)

## AuditResult Structure

```rust
pub struct AuditResult {
    pub verdict: Verdict,           // ALLOWED or BLOCKED
    pub confidence: f32,            // 0.0–1.0
    pub failure_category: String,   // e.g. "ghost_tooling", "sycophancy"
    pub what_worked: String,
    pub what_went_wrong: String,
    pub how_to_fix: String,
}
```

## Error Handling

- **Infrastructure error** (provider down, timeout): Fail-OPEN → pass through with logged warning
- **Parse error** (no JSON extractable from observer response): Fail-CLOSED → reject and retry

## Rejection Feedback Loop

When BLOCKED:
1. Increment rejection counter
2. Capture rejected response for preference pair (teacher/learning)
3. Inject structured feedback into context:
```
[OBSERVER AUDIT — BLOCKED]
Category: {failure_category}
What worked: {what_worked}
What went wrong: {what_went_wrong}
How to fix: {how_to_fix}
[You MUST address the above feedback in your next response.]
```
4. Continue ReAct loop — agent retries with feedback

## Bail-Out

After 2 consecutive blocks, inject CRITICAL override forcing the agent to reply honestly about what happened. After 3, force-deliver without audit. Prevents infinite rejection loops.

## Observer → Teacher Pipeline

Every rejection creates a preference pair:
- `chosen`: The response that eventually passed
- `rejected`: The response that was blocked
- `failure_category`: What was wrong

These pairs train away failure modes via preference optimization.

## Implementation Requirements

- Non-streaming LLM call (`stream: false`)
- Temperature: 0.1
- No tools for the observer (pure evaluator)
- 3-stage JSON parser: direct → markdown fence → balanced brace extraction
- Per-session audit stats logged
- Telemetry events: AuditRunning, AuditCompleted { verdict, reason }
- Configurable: observer_enabled, observer_model (default: same as chat model)
