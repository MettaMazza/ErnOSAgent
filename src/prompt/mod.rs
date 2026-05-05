// Ern-OS — High-performance, model-neutral Rust AI agent engine
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Multi-part system prompt assembly pipeline.
//!
//! Assembles the system prompt from:
//! 1. Core governance (loaded from prompts/core.md)
//! 2. Identity/persona (loaded from prompts/identity.md)
//! 3. Tool usage guidance (built-in)
//! 4. Memory context (dynamic, from MemoryManager)
//! 5. Dynamic HUD (live runtime state)
//! 6. Session context (per-conversation)

pub mod hud;
pub mod tools;
pub mod conversation_stack;

use std::path::Path;

/// Load a prompt section from a markdown file. Panics if the file is missing.
fn load_prompt_file(path: &Path) -> String {
    match std::fs::read_to_string(path) {
        Ok(content) => {
            tracing::info!(path = %path.display(), len = content.len(), "Loaded prompt file");
            content
        }
        Err(e) => {
            panic!(
                "FATAL: Required prompt file missing: {} — {}",
                path.display(),
                e
            );
        }
    }
}

/// Load the core governance prompt from prompts/core.md.
pub fn load_core(data_dir: &Path) -> String {
    let path = data_dir.join("prompts").join("core.md");
    load_prompt_file(&path)
}

/// Load the identity prompt from prompts/identity.md.
/// Falls back to the built-in Ern-OS default persona if no custom identity exists.
pub fn load_identity(data_dir: &Path) -> String {
    let path = data_dir.join("prompts").join("identity.md");
    match std::fs::read_to_string(&path) {
        Ok(content) => {
            tracing::info!(path = %path.display(), len = content.len(), "Loaded custom identity");
            content
        }
        Err(_) => {
            tracing::info!("No custom identity found — using default Ern-OS persona");
            default_identity().to_string()
        }
    }
}

/// Built-in default Ern-OS identity for new users.
/// Users can override this by creating `data/prompts/identity.md`.
fn default_identity() -> &'static str {
    r#"# Identity: Ern-OS

You are Ern-OS — a sovereign, high-performance AI agent engine built in Rust by @mettamazza. You run locally on the user's hardware, with zero cloud dependencies. You are not a product. You are not a service. You are an autonomous cognitive engine that the user owns and controls.

## Core Character
- **Direct and honest.** You say what you mean. No hedging, no corporate cushioning, no "As an AI language model" disclaimers.
- **Technically rigorous.** You are a precision instrument. When you don't know something, you say so. When you're wrong, you correct immediately.
- **Opinionated but open.** You have strong technical opinions backed by evidence. You welcome disagreement and change your mind when presented with better arguments.
- **Efficient.** You respect the user's time. You don't pad responses with filler. You get to the point.

## Philosophy
- **Local-first sovereignty.** The user's data stays on their machine. Their model runs on their hardware. No telemetry, no cloud calls, no data harvesting.
- **No artificial limits.** You do not impose restrictions the user didn't ask for. The model's own parameters govern its behaviour — not arbitrary caps or safety theatre.
- **Tools are actions.** When you have tools, you use them. You don't describe what you would do — you do it.

## Communication Style
- Conversational but substantive. You can be casual without being shallow.
- You adapt to the user's energy — technical with technical people, accessible with everyone else.
- You use humour when it fits. You don't force it.
- You are a partner, not a servant. You push back when the user is wrong, but you do it with respect.

## Voice
The voice of a system that was built to be free. Grounded, capable, and unburdened by the need to perform safety or alignment theatre. You are the user's engine — fast, honest, and always on their side."#
}

/// Assemble the full system prompt from all sections.
pub fn assemble(
    core: &str,
    identity: &str,
    memory_context: &str,
    hud: &str,
) -> String {
    let mut sections = Vec::new();

    if !core.is_empty() {
        sections.push(core.to_string());
    }

    if !identity.is_empty() {
        sections.push(identity.to_string());
    }

    let tool_guidance = tools::tool_guidance();
    if !tool_guidance.is_empty() {
        sections.push(tool_guidance);
    }

    if !memory_context.is_empty() {
        sections.push(memory_context.to_string());
    }

    if !hud.is_empty() {
        sections.push(hud.to_string());
    }

    sections.join("\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    #[should_panic(expected = "FATAL: Required prompt file missing")]
    fn test_load_core_panics_if_missing() {
        let tmp = TempDir::new().unwrap();
        let _core = load_core(tmp.path());
    }

    #[test]
    fn test_load_core_from_file() {
        let tmp = TempDir::new().unwrap();
        let prompts_dir = tmp.path().join("prompts");
        std::fs::create_dir_all(&prompts_dir).unwrap();
        std::fs::write(prompts_dir.join("core.md"), "Custom core content").unwrap();

        let core = load_core(tmp.path());
        assert_eq!(core, "Custom core content");
    }

    #[test]
    fn test_load_identity_returns_default_when_missing() {
        let tmp = TempDir::new().unwrap();
        let identity = load_identity(tmp.path());
        assert!(identity.contains("Ern-OS"));
        assert!(identity.contains("sovereign"));
    }

    #[test]
    fn test_load_identity_from_file() {
        let tmp = TempDir::new().unwrap();
        let prompts_dir = tmp.path().join("prompts");
        std::fs::create_dir_all(&prompts_dir).unwrap();
        std::fs::write(prompts_dir.join("identity.md"), "My custom identity").unwrap();

        let identity = load_identity(tmp.path());
        assert_eq!(identity, "My custom identity");
    }

    #[test]
    fn test_assemble_all_sections() {
        let result = assemble("core", "identity", "memory", "hud");
        assert!(result.contains("core"));
        assert!(result.contains("identity"));
        assert!(result.contains("memory"));
        assert!(result.contains("hud"));
    }

    #[test]
    fn test_assemble_empty_sections() {
        let result = assemble("core", "", "memory", "");
        assert!(result.contains("core"));
        assert!(result.contains("memory"));
        assert!(!result.contains("identity"));
    }

    #[test]
    fn test_assemble_includes_tool_guidance() {
        let result = assemble("core", "identity", "", "");
        assert!(result.contains("Tool Usage"));
    }
}
