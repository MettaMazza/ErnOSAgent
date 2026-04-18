// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Three-prompt assembly pipeline.

pub mod context;
pub mod core;
pub mod identity;

/// Assemble the full system prompt from all three components.
pub fn assemble_system_prompt(
    core_prompt: &str,
    context_prompt: &str,
    identity_prompt: &str,
) -> String {
    let mut prompt = String::new();

    if !core_prompt.is_empty() {
        prompt.push_str(core_prompt);
    }

    if !context_prompt.is_empty() {
        prompt.push_str("\n\n---\n\n");
        prompt.push_str(context_prompt);
    }

    if !identity_prompt.is_empty() {
        prompt.push_str("\n\n---\n\n");
        prompt.push_str(identity_prompt);
    }

    prompt
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assemble_all_three() {
        let result = assemble_system_prompt("core", "context", "identity");
        assert!(result.contains("core"));
        assert!(result.contains("context"));
        assert!(result.contains("identity"));
        assert!(result.contains("---"));
    }

    #[test]
    fn test_assemble_empty_sections() {
        let result = assemble_system_prompt("core", "", "identity");
        assert!(result.contains("core"));
        assert!(result.contains("identity"));
    }
}
