// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Turing Grid Tool — agent interface for the 3D Turing computation device.
//!
//! 14 actions matching HIVE's `operate_turing_grid` tool:
//! move, read, write, execute, scan, read_range, index,
//! label, goto, link, history, undo, pipeline, deploy_daemon
//!
//! Split into submodules by concern:
//! - `grid_actions`: move, read, write, scan, read_range, index
//! - `execution`: execute, pipeline, deploy_daemon
//! - `navigation`: label, goto, link, history, undo

mod execution;
mod grid_actions;
mod navigation;

use crate::computer::alu::ALU;
use crate::computer::turing_grid::TuringGrid;
use crate::tools::executor::ToolExecutor;
use crate::tools::schema::{ToolCall, ToolResult};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

/// Shared state for the Turing Grid and ALU.
/// Created once at startup and cloned into the tool handler closure.
#[derive(Clone)]
pub struct TuringState {
    pub grid: Arc<Mutex<TuringGrid>>,
    pub alu: Arc<ALU>,
}

impl TuringState {
    /// Create a new TuringState with the given data directory.
    pub async fn new(data_dir: &std::path::Path) -> Self {
        let grid_path = data_dir.join("computer_grid.json");
        let grid = if grid_path.exists() {
            TuringGrid::load(grid_path.clone())
                .await
                .unwrap_or_else(|_| TuringGrid::new(grid_path))
        } else {
            TuringGrid::new(grid_path)
        };

        let alu = ALU::new(Some(data_dir.join("computer_runtime")));
        if let Err(e) = alu.init().await {
            tracing::warn!(error = %e, "ALU runtime dir init failed");
        }

        Self {
            grid: Arc::new(Mutex::new(grid)),
            alu: Arc::new(alu),
        }
    }

    /// Create a TuringState with a dummy path (for testing).
    pub fn new_test() -> Self {
        Self {
            grid: Arc::new(Mutex::new(TuringGrid::new(PathBuf::from("test_grid.json")))),
            alu: Arc::new(ALU::new(Some(
                std::env::temp_dir().join("ernosagent_alu_test_tool"),
            ))),
        }
    }
}

/// Execute the operate_turing_grid tool.
pub(crate) fn execute_turing_tool(call: &ToolCall, state: &TuringState) -> ToolResult {
    let action = call
        .arguments
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("read")
        .to_lowercase();

    tracing::debug!(action = %action, "Turing Grid tool executing");

    let output = tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.block_on(dispatch_action(&action, call, state))
    });

    ToolResult {
        tool_call_id: call.id.clone(),
        name: call.name.clone(),
        output: output.0,
        success: output.1,
        error: output.2,
    }
}

/// Action result triple: (output_text, success, optional_error).
pub(crate) type ActionResult = (String, bool, Option<String>);

/// Dispatch to the correct action handler.
async fn dispatch_action(action: &str, call: &ToolCall, state: &TuringState) -> ActionResult {
    match action {
        "move" => grid_actions::action_move(call, state).await,
        "read" => grid_actions::action_read(state).await,
        "write" => grid_actions::action_write(call, state).await,
        "scan" => grid_actions::action_scan(call, state).await,
        "read_range" => grid_actions::action_read_range(call, state).await,
        "index" => grid_actions::action_index(state).await,
        "execute" => execution::action_execute(state).await,
        "pipeline" => execution::action_pipeline(call, state).await,
        "deploy_daemon" => execution::action_deploy_daemon(call, state).await,
        "label" => navigation::action_label(call, state).await,
        "goto" => navigation::action_goto(call, state).await,
        "link" => navigation::action_link(call, state).await,
        "history" => navigation::action_history(state).await,
        "undo" => navigation::action_undo(state).await,
        _ => (
            format!("Unknown action: '{}'. Available: move, read, write, execute, scan, read_range, index, label, goto, link, history, undo, pipeline, deploy_daemon", action),
            false,
            Some(format!("Unknown action: {}", action)),
        ),
    }
}

/// Register the turing grid tool with the executor.
pub fn register_tools(executor: &mut ToolExecutor, state: TuringState) {
    let s = state;
    executor.register(
        "operate_turing_grid",
        Box::new(move |call: &ToolCall| execute_turing_tool(call, &s)),
    );
}

#[cfg(test)]
#[path = "turing_tool_tests.rs"]
mod tests;
