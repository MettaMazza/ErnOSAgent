//! ReAct loop engine — the core Reason→Act→Observe cycle.
//!
//! The loop can ONLY be exited by the agent calling the `reply_request` tool.
//! There is NO fallback. If the model fails to call reply_request, the engine
//! injects an error and continues the loop.

use crate::observer::{audit, Verdict};
use crate::provider::{Message, Provider, StreamEvent, ToolDefinition};
use crate::tools::executor::ToolExecutor;
use crate::tools::schema::{self, ToolCall, ToolResult};
use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::mpsc;

/// Events emitted during the ReAct loop for TUI/telemetry display.
#[derive(Debug, Clone)]
pub enum ReactEvent {
    /// A new reasoning turn started.
    TurnStarted { turn: usize },
    /// A text token arrived.
    Token(String),
    /// A thinking token arrived.
    Thinking(String),
    /// A tool is being executed.
    ToolExecuting { name: String, id: String },
    /// A tool completed.
    ToolCompleted { name: String, result: ToolResult },
    /// Observer audit is running.
    AuditRunning,
    /// Observer audit completed.
    AuditCompleted { verdict: Verdict, reason: String },
    /// The loop is complete and a response is ready.
    ResponseReady { text: String },
    /// An error occurred.
    Error(String),
}

/// The final result of a ReAct loop execution.
#[derive(Debug, Clone)]
pub struct ReactResult {
    /// The delivered response text.
    pub response: String,
    /// Number of reasoning turns taken.
    pub turns: usize,
    /// All tool results from the loop.
    pub tool_results: Vec<ToolResult>,
    /// Observer audit passes.
    pub audit_passes: usize,
    /// Observer audit rejections.
    pub audit_rejections: usize,
}

/// Configuration for the ReAct loop.
pub struct ReactConfig {
    pub observer_enabled: bool,
    pub observer_model: Option<String>,
    pub max_audit_rejections: usize,
}

/// Execute the ReAct loop.
///
/// The loop continues until `reply_request` is called and the observer approves.
/// There are NO fallbacks — if the model doesn't call reply_request, the engine
/// injects an error reminding it that the tool is required.
pub async fn execute_react_loop(
    provider: &Arc<dyn Provider>,
    model: &str,
    initial_messages: Vec<Message>,
    tools: &[ToolDefinition],
    executor: &ToolExecutor,
    config: &ReactConfig,
    system_prompt: &str,
    identity_prompt: &str,
    event_tx: mpsc::Sender<ReactEvent>,
) -> Result<ReactResult> {
    let mut messages = initial_messages;
    let mut turn = 0_usize;
    let mut all_tool_results: Vec<ToolResult> = Vec::new();
    let mut audit_passes = 0_usize;
    let mut total_audit_rejections = 0_usize;
    let mut consecutive_audit_rejections = 0_usize;

    loop {
        turn += 1;
        let _ = event_tx.send(ReactEvent::TurnStarted { turn }).await;

        tracing::info!(turn = turn, messages = messages.len(), "ReAct turn starting");

        // 1. REASON: LLM inference
        let (tx, mut rx) = mpsc::channel::<StreamEvent>(256);

        let provider_clone = Arc::clone(provider);
        let model_owned = model.to_string();
        let msgs_clone = messages.clone();
        let tools_clone = tools.to_vec();

        let inference_handle = tokio::spawn(async move {
            provider_clone
                .chat(&model_owned, &msgs_clone, Some(&tools_clone), tx)
                .await
        });

        // Collect the response
        let mut response_text = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        while let Some(event) = rx.recv().await {
            match event {
                StreamEvent::Token(token) => {
                    response_text.push_str(&token);
                    let _ = event_tx.send(ReactEvent::Token(token)).await;
                }
                StreamEvent::Thinking(token) => {
                    let _ = event_tx.send(ReactEvent::Thinking(token)).await;
                }
                StreamEvent::ToolCall { id, name, arguments } => {
                    if let Ok(args) = serde_json::from_str::<serde_json::Value>(&arguments) {
                        tool_calls.push(ToolCall { id, name, arguments: args });
                    }
                }
                StreamEvent::Done { .. } => break,
                StreamEvent::Error(e) => {
                    let _ = event_tx.send(ReactEvent::Error(e.clone())).await;
                    tracing::error!(error = %e, turn = turn, "Inference error in ReAct loop");
                }
            }
        }

        // Wait for inference to complete
        inference_handle
            .await
            .context("Inference task panicked")?
            .context("Inference failed")?;

        // 2. CHECK: Process tool calls
        let has_reply = tool_calls.iter().any(schema::is_reply_request);
        let has_other_tools = tool_calls.iter().any(|tc| !schema::is_reply_request(tc));
        let has_no_calls = tool_calls.is_empty();

        // Execute non-reply tools first
        if has_other_tools {
            for call in &tool_calls {
                if schema::is_reply_request(call) {
                    continue;
                }

                let _ = event_tx
                    .send(ReactEvent::ToolExecuting {
                        name: call.name.clone(),
                        id: call.id.clone(),
                    })
                    .await;

                let result = executor.execute(call);

                let _ = event_tx
                    .send(ReactEvent::ToolCompleted {
                        name: call.name.clone(),
                        result: result.clone(),
                    })
                    .await;

                // Inject tool result into context
                messages.push(Message {
                    role: "tool".to_string(),
                    content: result.format_for_context(),
                    images: Vec::new(),
                });

                all_tool_results.push(result);
            }
        }

        // Handle reply_request
        if has_reply {
            let reply_call = tool_calls
                .iter()
                .find(|tc| schema::is_reply_request(tc))
                .expect("reply_request must exist if has_reply is true");

            let reply_text = schema::extract_reply_text(reply_call)
                .unwrap_or_else(|| response_text.clone());

            if reply_text.trim().is_empty() {
                // Empty reply — inject error and continue
                messages.push(Message {
                    role: "system".to_string(),
                    content: "[ERROR: reply_request was called with an empty message. \
                              You must provide meaningful content in the message field.]".to_string(),
                    images: Vec::new(),
                });
                continue;
            }

            // Observer audit (if enabled)
            if config.observer_enabled {
                let _ = event_tx.send(ReactEvent::AuditRunning).await;

                let audit_model = config
                    .observer_model
                    .as_deref()
                    .unwrap_or(model);

                let tool_context = ToolExecutor::format_tool_context(&all_tool_results);
                let capabilities = executor.available_tools().join(", ");

                let audit_result = audit::audit_response(
                    provider,
                    audit_model,
                    &messages
                        .iter()
                        .rev()
                        .find(|m| m.role == "user")
                        .map(|m| m.content.as_str())
                        .unwrap_or(""),
                    &reply_text,
                    &tool_context,
                    &capabilities,
                    system_prompt,
                    identity_prompt,
                )
                .await;

                let _ = event_tx
                    .send(ReactEvent::AuditCompleted {
                        verdict: audit_result.verdict.clone(),
                        reason: audit_result.failure_category.clone(),
                    })
                    .await;

                if audit_result.verdict.is_allowed() {
                    audit_passes += 1;

                    tracing::info!(
                        turns = turn,
                        tools = all_tool_results.len(),
                        audit_passes = audit_passes,
                        "ReAct loop complete (audit passed)"
                    );

                    let _ = event_tx
                        .send(ReactEvent::ResponseReady {
                            text: reply_text.clone(),
                        })
                        .await;

                    return Ok(ReactResult {
                        response: reply_text,
                        turns: turn,
                        tool_results: all_tool_results,
                        audit_passes,
                        audit_rejections: total_audit_rejections,
                    });
                } else {
                    // BLOCKED — inject feedback and retry
                    total_audit_rejections += 1;
                    consecutive_audit_rejections += 1;

                    tracing::warn!(
                        rejection = consecutive_audit_rejections,
                        category = %audit_result.failure_category,
                        "Observer BLOCKED — retrying"
                    );

                    // Bail-out mechanism
                    if consecutive_audit_rejections >= config.max_audit_rejections {
                        tracing::warn!(
                            rejections = consecutive_audit_rejections,
                            "Observer bail-out — forcing response delivery"
                        );

                        let _ = event_tx
                            .send(ReactEvent::ResponseReady {
                                text: reply_text.clone(),
                            })
                            .await;

                        return Ok(ReactResult {
                            response: reply_text,
                            turns: turn,
                            tool_results: all_tool_results,
                            audit_passes,
                            audit_rejections: total_audit_rejections,
                        });
                    }

                    // Inject feedback
                    if consecutive_audit_rejections >= 2 {
                        messages.push(Message {
                            role: "system".to_string(),
                            content: audit::format_bailout_override(consecutive_audit_rejections),
                            images: Vec::new(),
                        });
                    } else {
                        messages.push(Message {
                            role: "system".to_string(),
                            content: audit::format_rejection_feedback(&audit_result),
                            images: Vec::new(),
                        });
                    }

                    continue;
                }
            } else {
                // Observer disabled — deliver directly
                tracing::info!(
                    turns = turn,
                    tools = all_tool_results.len(),
                    "ReAct loop complete (no observer)"
                );

                let _ = event_tx
                    .send(ReactEvent::ResponseReady {
                        text: reply_text.clone(),
                    })
                    .await;

                return Ok(ReactResult {
                    response: reply_text,
                    turns: turn,
                    tool_results: all_tool_results,
                    audit_passes: 0,
                    audit_rejections: 0,
                });
            }
        }

        // NO tool calls and NO reply_request — inject error, continue loop
        if has_no_calls && !has_reply {
            tracing::warn!(
                turn = turn,
                "Model failed to call reply_request — injecting reminder"
            );

            // Add the model's raw response as assistant message for context
            if !response_text.is_empty() {
                messages.push(Message {
                    role: "assistant".to_string(),
                    content: response_text,
                    images: Vec::new(),
                });
            }

            messages.push(Message {
                role: "system".to_string(),
                content: "[ERROR: You did not call any tools and did not call reply_request. \
                          You MUST call the reply_request tool to deliver your response. \
                          Raw text content is NOT delivered to the user. Call reply_request \
                          with your response in the message field NOW.]".to_string(),
                images: Vec::new(),
            });

            continue;
        }

        // Has other tools but no reply — just continue the loop
        // (tool results were already injected above)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_react_event_variants() {
        let events = vec![
            ReactEvent::TurnStarted { turn: 1 },
            ReactEvent::Token("hello".to_string()),
            ReactEvent::Thinking("hmm".to_string()),
            ReactEvent::ToolExecuting { name: "search".to_string(), id: "t1".to_string() },
            ReactEvent::AuditRunning,
            ReactEvent::AuditCompleted { verdict: Verdict::Allowed, reason: "none".to_string() },
            ReactEvent::ResponseReady { text: "hi".to_string() },
            ReactEvent::Error("oops".to_string()),
        ];

        assert_eq!(events.len(), 8);
    }

    #[test]
    fn test_react_config() {
        let config = ReactConfig {
            observer_enabled: true,
            observer_model: None,
            max_audit_rejections: 3,
        };
        assert!(config.observer_enabled);
        assert_eq!(config.max_audit_rejections, 3);
    }

    #[test]
    fn test_react_result() {
        let result = ReactResult {
            response: "Hello!".to_string(),
            turns: 2,
            tool_results: Vec::new(),
            audit_passes: 1,
            audit_rejections: 0,
        };
        assert_eq!(result.turns, 2);
        assert_eq!(result.audit_passes, 1);
    }
}
