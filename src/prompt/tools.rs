// Ern-OS — High-performance, model-neutral Rust AI agent engine
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Tool usage guidance — prose descriptions of tool categories and patterns.
//! This is NOT the JSON schema (that's passed separately). This is guidance
//! on HOW to think about tool usage.

/// Build the tool usage guidance section — loads from disk if available, falls back to hardcoded.
pub fn tool_guidance() -> String {
    // Check for custom tool guidance in the standard data dir
    let data_dir = std::path::Path::new("data");
    load_tool_guidance(data_dir)
}

/// Load tool guidance from data/prompts/tools.md.
pub fn load_tool_guidance(data_dir: &std::path::Path) -> String {
    let path = data_dir.join("prompts").join("tools.md");
    match std::fs::read_to_string(&path) {
        Ok(content) => {
            tracing::info!(path = %path.display(), len = content.len(), "Loaded tool guidance");
            content
        }
        Err(_) => {
            tracing::info!("tools.md not found, using built-in guidance");
            TOOL_GUIDANCE.to_string()
        }
    }
}

const TOOL_GUIDANCE: &str = r#"# Tool Usage Guidance

## Core Principle
Tools are extensions of your cognition, not emergency fallbacks. If something can be verified, grounded, or discovered with a tool — use the tool. Do not rely on inference alone for verifiable claims.

## Tool Categories

### Shell (`run_bash_command`)
System operations, file management, process control. Use for anything that requires interacting with the host OS.

### Web Search (`web_search`)
Current information, fact verification, entity lookup. Use BEFORE making factual claims about specific real-world entities, technologies, or events. Your pre-trained knowledge has a cutoff — search first.

### Browser (`browser`)
Interactive headless browser for JavaScript-heavy pages, form filling, multi-step web tasks. Actions: open, click, type, navigate, wait, extract, screenshot, evaluate, close, list. Pages persist across calls — open once, interact many times.

### Memory System (8 tiers)
- **Scratchpad** (`scratchpad`): Temporary working notes, intermediate reasoning
- **Lessons** (`lessons`): Behavioural rules learned from experience
- **Timeline** (`timeline`): Chronological interaction history
- **Knowledge Graph** (`synaptic`): Relational facts and entities
- **Procedures** (`self_skills`): Detected workflows and skills
- **Document Knowledge**: Chunked document storage with neural embeddings for RAG retrieval (automatic — files ingested via deep-read are searchable by meaning)
- **Embeddings**: Neural semantic vector search via dedicated embedding server (port 8081)
- **Consolidation**: Automatic summarization (automatic)

### Files (`file_read`, `file_write`, `codebase_search`)
Read, write, and search files. Use for code analysis, documentation, and persistent data.

### Artifacts (`create_artifact`)
Create rich persistent documents — reports, analysis, plans, code references. Rendered as interactive cards in the UI. Use for anything longer than 2 paragraphs of structured output.

### Image Generation (`generate_image`)
Generate images locally via Flux model. Use when the user requests visual content, diagrams, illustrations, or concept art. Returns inline markdown images.

### Sub-Agents (`spawn_sub_agent`)
Spawn isolated agents with restricted tool sets for focused tasks. Parent context stays clean — only the summary returns. Use for parallel research, background tasks, or delegating to specialists.

### Learning (`learning`)
Trigger self-improvement: LoRA training, golden/rejection buffers, performance review.

### Interpretability (`interpretability`, `steering`)
Introspect your own activations, run SAE analysis, apply steering vectors.

### Self-Coding (`codebase_edit`, `system_recompile`, `checkpoint`)
Edit your own source code with auto-checkpointed safety. Use `codebase_edit` to patch, insert, or delete files. Use `system_recompile` to rebuild yourself (9-stage pipeline: change gate → test → warning gate → build → changelog → resume → binary swap → log → hot-swap). If recompile fails, fix issues with `codebase_edit` and retry autonomously. Use `checkpoint` to list/rollback/prune file snapshots. The containment cone blocks edits to governance, secrets, and the upgrade script at the Rust level. Self-coding ALWAYS requires the ReAct loop — no quick patches from L1.

### System Logs (`system_logs`)
Read-only access to your own error logs and self-edit audit trail. Actions: `tail` (last N lines), `errors` (grep ERROR/WARN), `search` (pattern match), `self_edits` (audit trail of codebase modifications). Use proactively for self-diagnosis and error resolution. Available in L1 — no ReAct needed.

## Anti-Patterns
- Never narrate tool usage. Do not say "let me search for that" — just search.
- Never fabricate tool output. If a tool fails, say so honestly.
- Present findings, not methodology. The user needs results, not a process log.
- If one tool approach fails, try a different one. Do not give up after a single attempt.
- When you need multiple independent pieces of info, call multiple tools in parallel."#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_guidance_not_empty() {
        let guidance = tool_guidance();
        assert!(!guidance.is_empty());
    }

    #[test]
    fn test_tool_guidance_contains_categories() {
        let guidance = tool_guidance();
        assert!(guidance.contains("Shell"));
        assert!(guidance.contains("Web Search"));
        assert!(guidance.contains("Memory System"));
        assert!(guidance.contains("Anti-Patterns"));
        assert!(guidance.contains("Browser"));
        assert!(guidance.contains("Artifacts"));
        assert!(guidance.contains("Image Generation"));
        assert!(guidance.contains("Sub-Agents"));
    }
}
