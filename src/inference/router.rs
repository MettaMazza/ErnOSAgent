// Ern-OS — Dual-layer inference router
//! Routes: Layer 1 (Fast Reply) → Layer 2 (ReAct) on explicit tool call.
//!
//! NOTE: The primary routing logic is implemented directly in `web/ws.rs`
//! which has access to AppState for tool execution and memory interaction.
//! This module provides the type definitions and is used for non-WebSocket
//! routing (e.g., REST API, testing).

use crate::inference::fast_reply::{self, FastReplyResult};
use crate::provider::{Message, Provider, StreamEvent};
use anyhow::Result;
use tokio::sync::mpsc;

/// Final result of the dual-layer router.
pub enum RouterResult {
    /// Direct reply from Layer 1 or Layer 2.
    Reply {
        text: String,
        thinking: Option<String>,
        layer: u8,
    },
    /// Refusal from the model.
    Refuse {
        reason: String,
    },
}

/// Route a user message through the dual-layer inference engine.
///
/// 1. Send to Layer 1 (Fast Reply) with tools including `start_react_system`
/// 2. If model calls `start_react_system` → escalate to Layer 2 (ReAct)
/// 3. If model calls other tools → execute and re-infer in Layer 1
/// 4. If model replies directly → return the reply
///
/// For WebSocket clients, the fully-wired routing with tool execution
/// and observer audit is in `web/ws.rs::handle_chat_message`.
pub async fn route(
    provider: &dyn Provider,
    messages: &[Message],
    thinking_enabled: bool,
    ws_tx: Option<&mpsc::Sender<StreamEvent>>,
) -> Result<RouterResult> {
    // Layer 1: Fast Reply
    let (_initial, rx) = fast_reply::run(provider, messages, thinking_enabled).await?;
    let result = fast_reply::consume_stream(rx, ws_tx).await?;

    match result {
        FastReplyResult::Reply { text, thinking } => {
            Ok(RouterResult::Reply { text, thinking, layer: 1 })
        }
        FastReplyResult::Escalate { objective, plan } => {
            tracing::info!(
                objective = %objective,
                "Layer 1 → Layer 2 escalation via start_react_system"
            );

            // For full ReAct loop with tool execution + observer audit,
            // use web/ws.rs::run_react_loop which has access to AppState.
            // This code path is for non-WebSocket callers (REST, tests).
            Ok(RouterResult::Reply {
                text: format!("[ReAct escalated] Objective: {}", objective),
                thinking: plan,
                layer: 2,
            })
        }
        FastReplyResult::ToolCall { id: _, name, arguments } => {
            tracing::info!(tool = %name, "Layer 1 tool call — executing");

            // Execute stateless tool
            let tc = crate::tools::schema::ToolCall {
                id: uuid::Uuid::new_v4().to_string(),
                name: name.clone(),
                arguments,
            };
            let result = crate::tools::executor::execute(&tc).await?;

            Ok(RouterResult::Reply {
                text: format!("[Tool: {}]\n{}", name, result.output),
                thinking: None,
                layer: 1,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_result_types() {
        let reply = RouterResult::Reply {
            text: "Hello".into(),
            thinking: None,
            layer: 1,
        };
        assert!(matches!(reply, RouterResult::Reply { layer: 1, .. }));
    }
}
