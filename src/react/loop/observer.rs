// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Observer audit pipeline and rejection handling.

use super::{ReactConfig, ReactEvent, ReactResult};
use super::learning::{self, LearningContext};
use crate::observer::audit;
use crate::provider::{Message, Provider};
use crate::tools::executor::ToolExecutor;
use crate::tools::schema::ToolResult;
use std::sync::Arc;
use tokio::sync::mpsc;

/// Outcome of a reply attempt.
pub(super) enum ReplyOutcome {
    Deliver(ReactResult),
    Retry,
}

/// Handle a reply_request: run observer audit and decide delivery.
pub(super) async fn handle_reply(
    reply_text: &str,
    provider: &Arc<dyn Provider>,
    model: &str,
    messages: &mut Vec<Message>,
    config: &ReactConfig,
    system_prompt: &str,
    _identity_prompt: &str,
    executor: &ToolExecutor,
    all_tool_results: &[ToolResult],
    turn: usize,
    audit_passes: &mut usize,
    total_audit_rejections: &mut usize,
    consecutive_audit_rejections: &mut usize,
    event_tx: &mpsc::Sender<ReactEvent>,
    learning_ctx: &mut LearningContext,
) -> ReplyOutcome {
    if !config.observer_enabled {
        tracing::info!(turns = turn, tools = all_tool_results.len(), "ReAct loop complete (no observer)");
        if let Some(ref buffers) = learning_ctx.buffers {
            learning::capture_golden(buffers, system_prompt, &learning_ctx.user_message, reply_text, &learning_ctx.session_id, model);
        }
        return deliver_response(reply_text, turn, all_tool_results, 0, 0, event_tx).await;
    }

    let audit_result = run_observer_audit(
        provider, model, messages, config, executor, all_tool_results, reply_text, event_tx,
        &learning_ctx.user_message,
    ).await;

    if audit_result.verdict.is_allowed() {
        *audit_passes += 1;
        tracing::info!(turns = turn, audit_passes = *audit_passes, "ReAct loop complete (audit passed)");

        if let Some(ref buffers) = learning_ctx.buffers {
            if *consecutive_audit_rejections == 0 {
                learning::capture_golden(buffers, system_prompt, &learning_ctx.user_message, reply_text, &learning_ctx.session_id, model);
            } else if let Some(ref rejected) = learning_ctx.last_rejected {
                let category = learning_ctx.last_failure_category.as_deref().unwrap_or("unknown");
                learning::capture_preference(
                    buffers, system_prompt, &learning_ctx.user_message,
                    rejected, reply_text, category, &learning_ctx.session_id, model,
                );
            }
        }
        learning_ctx.last_rejected = None;
        learning_ctx.last_failure_category = None;

        return deliver_response(
            reply_text, turn, all_tool_results, *audit_passes, *total_audit_rejections, event_tx,
        ).await;
    }

    learning_ctx.last_rejected = Some(reply_text.to_string());
    learning_ctx.last_failure_category = Some(audit_result.failure_category.clone());

    handle_audit_rejection(
        &audit_result, reply_text, messages, config, all_tool_results, turn,
        audit_passes, total_audit_rejections, consecutive_audit_rejections, event_tx,
    ).await
}

async fn deliver_response(
    text: &str,
    turn: usize,
    all_tool_results: &[ToolResult],
    audit_passes: usize,
    audit_rejections: usize,
    event_tx: &mpsc::Sender<ReactEvent>,
) -> ReplyOutcome {
    let _ = event_tx.send(ReactEvent::ResponseReady { text: text.to_string() }).await;
    ReplyOutcome::Deliver(ReactResult {
        response: text.to_string(),
        turns: turn,
        tool_results: all_tool_results.to_vec(),
        audit_passes,
        audit_rejections,
    })
}

async fn run_observer_audit(
    provider: &Arc<dyn Provider>,
    model: &str,
    messages: &[Message],
    config: &ReactConfig,
    executor: &ToolExecutor,
    all_tool_results: &[ToolResult],
    reply_text: &str,
    event_tx: &mpsc::Sender<ReactEvent>,
    user_message: &str,
) -> audit::AuditResult {
    let _ = event_tx.send(ReactEvent::AuditRunning).await;
    let audit_model = config.observer_model.as_deref().unwrap_or(model);
    let tool_context = ToolExecutor::format_tool_context(all_tool_results);
    let capabilities = executor.available_tools().join(", ");

    let result = audit::audit_response(
        provider, audit_model, messages, reply_text, &tool_context, &capabilities, user_message,
    ).await;

    let _ = event_tx.send(ReactEvent::AuditCompleted {
        verdict: result.verdict.clone(),
        reason: result.failure_category.clone(),
    }).await;

    result
}

async fn handle_audit_rejection(
    audit_result: &audit::AuditResult,
    _reply_text: &str,
    messages: &mut Vec<Message>,
    _config: &ReactConfig,
    _all_tool_results: &[ToolResult],
    _turn: usize,
    _audit_passes: &mut usize,
    total_audit_rejections: &mut usize,
    consecutive_audit_rejections: &mut usize,
    _event_tx: &mpsc::Sender<ReactEvent>,
) -> ReplyOutcome {
    *total_audit_rejections += 1;
    *consecutive_audit_rejections += 1;

    tracing::warn!(
        rejection = *consecutive_audit_rejections,
        category = %audit_result.failure_category,
        "Observer BLOCKED — retrying (no bail-out, unlimited retries)"
    );

    // Escalating recovery prompts based on rejection count
    let feedback = if *consecutive_audit_rejections >= 10 {
        // After 10 rejections, strip everything down to essentials
        format!(
            "[SYSTEM: CRITICAL — {} consecutive observer rejections. \
            Your responses keep failing audit. Strip your answer to the absolute minimum. \
            Answer ONLY what was asked. No preamble, no disclaimers, no elaboration. \
            Failure category: '{}'. Fix ONLY that and call reply_request.]",
            consecutive_audit_rejections,
            audit_result.failure_category,
        )
    } else if *consecutive_audit_rejections >= 5 {
        // After 5, add stronger directive
        format!(
            "[SYSTEM: {} consecutive observer rejections on category '{}'. \
            Your previous attempts were rejected. Re-read the user's original question carefully. \
            Provide a direct, factual answer. Do not speculate. Do not add unnecessary content. \
            The observer will keep rejecting until your response is correct.]",
            consecutive_audit_rejections,
            audit_result.failure_category,
        )
    } else if *consecutive_audit_rejections >= 2 {
        messages.push(Message {
            role: "system".to_string(),
            content: audit::format_bailout_override(*consecutive_audit_rejections),
            images: Vec::new(),
        });
        return ReplyOutcome::Retry;
    } else {
        messages.push(Message {
            role: "system".to_string(),
            content: audit::format_rejection_feedback(audit_result),
            images: Vec::new(),
        });
        return ReplyOutcome::Retry;
    };

    messages.push(Message {
        role: "system".to_string(),
        content: feedback,
        images: Vec::new(),
    });

    ReplyOutcome::Retry
}
