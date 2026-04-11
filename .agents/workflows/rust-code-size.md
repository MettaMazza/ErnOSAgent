---
description: Rust code size and structure guidelines — file length, function length, and struct/trait complexity limits
---

# Rust Code Size & Structure Guidelines

These are the mandatory structural limits for all Rust source files in this project. Apply these rules when creating new files, modifying existing files, or conducting code reviews.

// turbo-all

## File Length Limits

| Range | Verdict | Action |
|-------|---------|--------|
| **~100–300 lines** | ✅ Ideal | Easy to reason about, easy to test |
| **~300–500 lines** | ⚠️ Acceptable | Only if the module has a clear single purpose |
| **~500+ lines** | 🔴 Split required | Must be refactored into smaller modules |

### Exception: Operational Kernel (`src/prompt/core.rs`)
The operational kernel is a single `const` string literal containing the full-depth system prompt. It is exempt from the 300-line ideal because it is **prompt text, not code logic** — splitting it across files would fragment the kernel's coherence with no structural benefit. The file remains under 500 lines.

### How to split a file over 500 lines

1. Identify distinct responsibilities within the file
2. Extract each responsibility into its own submodule file
3. Keep the original file as a thin orchestrator that re-exports and delegates
4. Move tests into a dedicated `tests.rs` sibling file if they exceed ~100 lines

**Example split for `app.rs` (673 lines):**
```
src/app.rs (200 lines — orchestrator only)
src/ui/tui.rs (new — TUI rendering)
src/ui/keybindings.rs (new — key event handling)
```

## Function Length Limits

| Range | Verdict | Action |
|-------|---------|--------|
| **~10–30 lines** | ✅ Sweet spot | Clear, testable, single-purpose |
| **~30–50 lines** | ⚠️ Review | Consider extracting helper functions |
| **~50+ lines** | 🔴 Refactor | Must be broken into smaller functions |

### The "and" test
If you can't describe what a function does **without using the word "and"**, it is doing too much and must be split.

**Bad:** "This function parses the config **and** validates it **and** creates the directories."
**Good:** "This function parses the config file into an `AppConfig` struct."

### Common extraction patterns
- Extract validation logic into a `validate_*()` function
- Extract IO operations into a `read_*()` / `write_*()` function
- Extract transformation logic into a `transform_*()` / `build_*()` function
- Extract error handling into a `try_*()` wrapper

## Struct/Trait Complexity Limits

| Range | Verdict | Action |
|-------|---------|--------|
| **~5–10 methods** per `impl` block | ✅ Comfortable | Well-focused responsibility |
| **~10–15 methods** | ⚠️ Review | Consider splitting into traits or helper modules |
| **~15+ methods** | 🔴 Refactor | The struct is wearing too many hats |

### How to split a large impl block

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

## Audit Checklist

When modifying or reviewing any Rust file, check:

1. [ ] File is under 500 lines (excluding tests)
2. [ ] No function exceeds 50 lines
3. [ ] No `impl` block has more than 15 methods
4. [ ] Each function passes the "and" test
5. [ ] Tests are proportional (aim for 1 test per public function minimum)
6. [ ] Module has a clear single purpose described in the `//!` doc comment

## Current Status (as of 2026-04-09)

All previously-violating files have been refactored. Tests extracted to sibling `*_tests.rs` files using `#[path]`.

| File | Lines | Status |
|------|-------|--------|
| `src/app/mod.rs` | 477 | ✅ Split from 673 → mod.rs + keybindings.rs |
| `src/app/keybindings.rs` | 206 | ✅ Extracted |
| `src/provider/llamacpp.rs` | 457 | ✅ Split from 624 → + stream_parser.rs + tests |
| `src/provider/stream_parser.rs` | 116 | ✅ Shared SSE parser |
| `src/provider/ollama.rs` | 510 | ⚠️ Single-purpose, tests extracted |
| `src/config.rs` | 466 | ✅ Split from 553 → tests extracted |
| `src/memory/knowledge_graph.rs` | 460 | ✅ Split from 543 → tests extracted |
| `src/memory/mod.rs` | 316 | ✅ Split from 519 → tests extracted |

