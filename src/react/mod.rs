// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! ReAct loop — Reason→Act→Observe cycle.

pub mod r#loop;
pub mod reply;

pub use r#loop::{ReactEvent, ReactResult};
