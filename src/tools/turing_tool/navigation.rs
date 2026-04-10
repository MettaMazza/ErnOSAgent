// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Navigation actions — label, goto, link, history, undo.

use super::{ActionResult, TuringState};
use crate::tools::schema::ToolCall;

pub(super) async fn action_label(
    call: &ToolCall,
    state: &TuringState,
) -> ActionResult {
    let name = match call.arguments.get("name").and_then(|v| v.as_str()) {
        Some(n) if !n.is_empty() => n,
        _ => {
            return (
                "Error: No label name provided. Pass 'name' with the label.".to_string(),
                false,
                Some("No label name provided.".to_string()),
            )
        }
    };

    let mut g = state.grid.lock().await;
    let (x, y, z) = g.get_cursor();
    match g.set_label(name).await {
        Ok(()) => (
            format!("Label '{}' set at coordinates ({}, {}, {}).", name, x, y, z),
            true,
            None,
        ),
        Err(e) => (
            format!("Error: Failed to set label: {}", e),
            false,
            Some(format!("Failed to set label: {}", e)),
        ),
    }
}

pub(super) async fn action_goto(
    call: &ToolCall,
    state: &TuringState,
) -> ActionResult {
    let name = match call.arguments.get("name").and_then(|v| v.as_str()) {
        Some(n) if !n.is_empty() => n,
        _ => {
            return (
                "Error: No label name provided for 'goto'. Pass 'name' with the label.".to_string(),
                false,
                Some("No label name provided.".to_string()),
            )
        }
    };

    let mut g = state.grid.lock().await;
    match g.goto_label(name).await {
        Some((x, y, z)) => (
            format!("Jumped to label '{}' at ({}, {}, {}).", name, x, y, z),
            true,
            None,
        ),
        None => {
            let available: Vec<String> = g.labels.keys().cloned().collect();
            let msg = if available.is_empty() {
                format!("Label '{}' not found. No labels have been set yet.", name)
            } else {
                format!(
                    "Label '{}' not found. Available labels: {}",
                    name,
                    available.join(", ")
                )
            };
            (msg, false, Some(format!("Label '{}' not found", name)))
        }
    }
}

pub(super) async fn action_link(
    call: &ToolCall,
    state: &TuringState,
) -> ActionResult {
    let tx = call.arguments.get("target_x").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
    let ty = call.arguments.get("target_y").and_then(|v| v.as_i64()).unwrap_or(0) as i32;
    let tz = call.arguments.get("target_z").and_then(|v| v.as_i64()).unwrap_or(0) as i32;

    let mut g = state.grid.lock().await;
    let (x, y, z) = g.get_cursor();

    match g.add_link((tx, ty, tz)).await {
        Ok(true) => (
            format!("Linked cell ({},{},{}) → ({},{},{}).", x, y, z, tx, ty, tz),
            true,
            None,
        ),
        Ok(false) => (
            format!(
                "Error: Cell ({},{},{}) is empty. Write data before linking.",
                x, y, z
            ),
            false,
            Some("Empty cell".to_string()),
        ),
        Err(e) => (
            format!("Error: Link failed: {}", e),
            false,
            Some(format!("Link failed: {}", e)),
        ),
    }
}

pub(super) async fn action_history(state: &TuringState) -> ActionResult {
    let g = state.grid.lock().await;
    let (x, y, z) = g.get_cursor();

    match g.get_history() {
        Some(hist) if !hist.is_empty() => {
            let mut out = format!(
                "--- Version History for Cell ({},{},{}) ({} entries) ---\n",
                x, y, z, hist.len()
            );
            for (i, snap) in hist.iter().enumerate() {
                let preview: String = snap.content.chars().take(100).collect();
                out.push_str(&format!(
                    "  v-{}: [{}] {} (at {})\n",
                    i + 1, snap.format, preview, snap.timestamp
                ));
            }
            (out, true, None)
        }
        Some(_) => (
            format!("Cell ({},{},{}) has no version history.", x, y, z),
            true,
            None,
        ),
        None => (
            format!("Cell ({},{},{}) is empty — no history.", x, y, z),
            true,
            None,
        ),
    }
}

pub(super) async fn action_undo(state: &TuringState) -> ActionResult {
    let mut g = state.grid.lock().await;
    let (x, y, z) = g.get_cursor();

    match g.undo().await {
        Ok(true) => {
            let content = g
                .read_current()
                .map(|c| c.content.clone())
                .unwrap_or_default();
            (
                format!(
                    "Undo successful. Cell ({},{},{}) restored to previous version.\nContent: {}",
                    x, y, z, content
                ),
                true,
                None,
            )
        }
        Ok(false) => (
            format!(
                "Cannot undo: Cell ({},{},{}) has no version history.",
                x, y, z
            ),
            false,
            Some("No version history".to_string()),
        ),
        Err(e) => (
            format!("Error: Undo failed: {}", e),
            false,
            Some(format!("Undo failed: {}", e)),
        ),
    }
}
