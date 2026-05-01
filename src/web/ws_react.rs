//! WebSocket ReAct loop execution — runs the full agentic tool loop.

use crate::inference::react_loop::{self, ReactContext, IterationResult};
use crate::observer;
use crate::provider::Message;
use crate::tools::schema;
use crate::web::state::AppState;
use crate::web::ws_learning::{ingest_assistant_turn, spawn_insight_extraction};
use crate::web::ws_stream::send_ws;
use axum::extract::ws::{Message as WsMessage, WebSocket};

/// Mutable state tracked across ReAct iterations.
struct LoopState {
    remaining_turns: usize,
    total_iterations: usize,
    budget_exhausted_prompted: bool,
    consecutive_rejections: usize,
    last_fail_signature: Option<String>,
    consecutive_fails: usize,
    empty_reply_retries: usize,
    progress: crate::inference::progress::ProgressTracker,
}

impl LoopState {
    fn new(planned_turns: usize, session_id: &str) -> Self {
        Self {
            remaining_turns: planned_turns,
            total_iterations: 0,
            budget_exhausted_prompted: false,
            consecutive_rejections: 0,
            last_fail_signature: None,
            consecutive_fails: 0,
            empty_reply_retries: 0,
            progress: crate::inference::progress::ProgressTracker::new(session_id),
        }
    }
}

/// Run the full ReAct loop with tool execution and observer audit.
pub async fn run_react_loop(
    state: &AppState,
    provider: &dyn crate::provider::Provider,
    base_messages: Vec<Message>,
    objective: &str,
    plan: Option<&str>,
    planned_turns: usize,
    user_query: &str,
    session_id: &str,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
    stop_flag: &std::sync::Arc<std::sync::atomic::AtomicBool>,
) {
    let mut ctx = ReactContext::new(objective, plan, base_messages);
    let mut ls = LoopState::new(planned_turns, session_id);

    loop {
        if handle_user_stop(stop_flag, &mut ctx, &ls, state, provider, sender, session_id).await {
            return;
        }
        handle_budget_exhaustion(&mut ctx, &mut ls);

        tracing::info!(iteration = ls.total_iterations, remaining = ls.remaining_turns, "ReAct iteration");
        state.cancel_flag.store(false, std::sync::atomic::Ordering::Relaxed);

        // After budget exhaustion, the assessment prompt is a directive (reply_request
        // or extend_turns) — not a reasoning task. Thinking is unnecessary and can cause
        // models to emit stop with zero content. Model-neutral per §7.1.
        let thinking = !ls.budget_exhausted_prompted;
        match react_loop::run_iteration_cancellable(provider, &ctx, thinking, Some(&state.cancel_flag)).await {
            Ok(IterationResult::Reply(reply, thinking)) => {
                if handle_reply(state, provider, &mut ctx, &mut ls, &reply, &thinking, user_query, session_id, sender).await {
                    return;
                }
            }
            Ok(IterationResult::Refuse(reason)) => {
                handle_refuse(&reason, &ls, sender).await;
                return;
            }
            Ok(IterationResult::ExtendTurns { additional, progress, remaining_work }) => {
                handle_extend_turns(&mut ctx, &mut ls, additional, &progress, &remaining_work, sender).await;
            }
            Ok(IterationResult::ToolCall(tc)) => {
                handle_single_tool(&mut ctx, &mut ls, state, &tc, sender).await;
            }
            Ok(IterationResult::ToolCalls(tcs)) => {
                handle_parallel_tools(&mut ctx, &mut ls, state, &tcs, sender).await;
            }
            Ok(IterationResult::ImplicitReply(text, thinking)) => {
                if handle_implicit_reply(&mut ctx, &mut ls, state, &text, &thinking, user_query, session_id, sender).await {
                    return;
                }
            }
            Err(e) => {
                handle_iteration_error(&e, &ls, sender).await;
                return;
            }
        }
    }
}

/// Handle user stop: if flagged, force a final reply and return true.
async fn handle_user_stop(
    stop_flag: &std::sync::Arc<std::sync::atomic::AtomicBool>,
    ctx: &mut ReactContext,
    ls: &LoopState,
    state: &AppState,
    provider: &dyn crate::provider::Provider,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
    session_id: &str,
) -> bool {
    if !stop_flag.load(std::sync::atomic::Ordering::Relaxed) {
        return false;
    }
    tracing::info!(iteration = ls.total_iterations, "ReAct loop stopped by user");
    ctx.messages.push(Message::text("system",
        "[USER INTERRUPT] The user has stopped this loop. \
         Summarize everything you have gathered so far and deliver your \
         best response using reply_request. Do NOT call any more tools."
    ));
    match react_loop::run_iteration(provider, ctx, true).await {
        Ok(IterationResult::Reply(reply, thinking)) | Ok(IterationResult::ImplicitReply(reply, thinking)) => {
            if let Some(ref t) = thinking {
                send_ws(sender, "thinking_delta", &serde_json::json!({"content": t})).await;
            }
            send_ws(sender, "text_delta", &serde_json::json!({"content": &reply})).await;
            ingest_assistant_turn(state, &reply, session_id).await;
        }
        _ => {
            send_ws(sender, "text_delta", &serde_json::json!({"content": "(Loop stopped by user)"})).await;
        }
    }
    send_ws(sender, "done", &serde_json::json!({})).await;
    true
}

/// Check if turn budget is exhausted and inject assessment prompt.
fn handle_budget_exhaustion(ctx: &mut ReactContext, ls: &mut LoopState) {
    if ls.remaining_turns != 0 || ls.budget_exhausted_prompted {
        return;
    }
    ls.budget_exhausted_prompted = true;
    tracing::info!(total = ls.total_iterations, "ReAct turns exhausted — forcing assessment");
    ctx.messages.push(Message::text("system",
        &format!(
            "[TURN BUDGET EXHAUSTED] You have used all {} planned turns.\n\n\
             STOP. Assess your progress:\n\
             - What have you accomplished so far?\n\
             - Do you have enough information to answer the user?\n\n\
             You have exactly TWO options:\n\
             1. Call `reply_request` with your complete response if you have enough.\n\
             2. Call `extend_turns` with a progress summary, what work remains, and a NEW turn estimate.\n\n\
             You MUST NOT call any other tool. Assess and decide.",
            ls.total_iterations
        )
    ));
}

/// Handle a Reply result with observer audit. Returns true if loop should exit.
async fn handle_reply(
    state: &AppState,
    provider: &dyn crate::provider::Provider,
    ctx: &mut ReactContext,
    ls: &mut LoopState,
    reply: &str,
    thinking: &Option<String>,
    user_query: &str,
    session_id: &str,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
) -> bool {
    tracing::info!(
        iteration = ls.total_iterations,
        reply_len = reply.len(),
        thinking_len = thinking.as_ref().map(|t| t.len()).unwrap_or(0),
        reply_preview = %reply.chars().take(200).collect::<String>(),
        "ReAct: Reply (via reply_request)"
    );
    if let Some(ref t) = thinking {
        send_ws(sender, "thinking_delta", &serde_json::json!({"content": t})).await;
    }
    if state.config.observer.enabled {
        if !run_observer_audit(state, provider, ctx, ls, reply, user_query, session_id, sender).await {
            return false; // Rejected — continue loop
        }
    }
    deliver_final_reply(state, reply, user_query, session_id, sender).await;
    true
}

/// Run observer audit on a reply. Returns true if approved, false if rejected.
/// No bailout — the model must produce an approved response.
async fn run_observer_audit(
    state: &AppState,
    provider: &dyn crate::provider::Provider,
    ctx: &mut ReactContext,
    ls: &mut LoopState,
    reply: &str,
    user_query: &str,
    session_id: &str,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
) -> bool {
    send_ws(sender, "audit_running", &serde_json::json!({})).await;
    use crate::inference::react_observer::{audit_reply, ObserverVerdict};
    match audit_reply(provider, &ctx.messages, reply, state.config.observer.enabled, user_query).await {
        Ok(ObserverVerdict::Approved) => {
            ls.consecutive_rejections = 0;
            send_ws(sender, "audit_completed", &serde_json::json!({"approved": true})).await;
            let tool_context = build_tool_context(&ctx.messages);
            if let Ok(output) = observer::audit_response(provider, &ctx.messages, reply, &tool_context, user_query).await {
                crate::web::ws_stream::save_conversation_stack(state, session_id, &output.result);
            }
            true
        }
        Ok(ObserverVerdict::Rejected { reason, guidance }) => {
            ls.consecutive_rejections += 1;
            tracing::warn!(
                reason = %reason,
                guidance = %guidance,
                consecutive = ls.consecutive_rejections,
                "ReAct reply rejected — re-inferring with feedback"
            );
            ctx.messages.push(Message::text("assistant", reply));
            let feedback = observer::format_rejection_feedback_from_reason(&reason, &guidance);
            ctx.messages.push(Message::text("system", &feedback));
            false
        }
        Err(e) => {
            // Infrastructure error — fail-open (observer itself is down)
            tracing::warn!(error = %e, "Observer failed — fail-open");
            true
        }
    }
}

/// Deliver a final reply to the user and clean up.
async fn deliver_final_reply(
    state: &AppState,
    reply: &str,
    user_query: &str,
    session_id: &str,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
) {
    send_ws(sender, "text_delta", &serde_json::json!({"content": reply})).await;
    ingest_assistant_turn(state, reply, session_id).await;
    spawn_insight_extraction(state, user_query, reply);
    spawn_skill_synthesis(state, user_query);
    send_ws(sender, "done", &serde_json::json!({})).await;
}

/// Background skill synthesis — delegated to ws_react_helpers.
fn spawn_skill_synthesis(state: &AppState, user_query: &str) {
    super::ws_react_helpers::spawn_skill_synthesis(state, user_query);
}

/// Handle a Refuse result.
async fn handle_refuse(
    reason: &str,
    ls: &LoopState,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
) {
    tracing::info!(iteration = ls.total_iterations, reason = %reason, "ReAct: Refuse");
    send_ws(sender, "text_delta", &serde_json::json!({"content": format!("I cannot complete this: {}", reason)})).await;
    send_ws(sender, "done", &serde_json::json!({})).await;
}

/// Handle an ExtendTurns result.
async fn handle_extend_turns(
    ctx: &mut ReactContext,
    ls: &mut LoopState,
    additional: usize,
    progress: &str,
    remaining_work: &str,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
) {
    ls.remaining_turns = additional;
    ls.budget_exhausted_prompted = false;
    tracing::info!(granted = additional, progress = %progress, remaining_work = %remaining_work, "ReAct extension granted");
    send_ws(sender, "status", &serde_json::json!({
        "message": format!("ReAct extended (+{} turns after assessment)", additional)
    })).await;
    ctx.messages.push(Message::text("system",
        &format!(
            "[EXTENSION GRANTED] You have been granted {} additional turns. \
             Continue with your plan. Remaining work: {}",
            additional, remaining_work
        )
    ));
}

/// Handle a single ToolCall result.
async fn handle_single_tool(
    ctx: &mut ReactContext,
    ls: &mut LoopState,
    state: &AppState,
    tc: &schema::ToolCall,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
) {
    tracing::info!(iteration = ls.total_iterations, tool = %tc.name, remaining = ls.remaining_turns, "ReAct: ToolCall");
    if reject_if_budget_exhausted(ctx, ls, tc) { return; }

    send_ws(sender, "tool_executing", &serde_json::json!({"name": &tc.name, "id": &tc.id})).await;
    let result = execute_tool_with_state(state, tc).await;
    send_ws(sender, "tool_completed", &serde_json::json!({
        "id": &tc.id, "name": &tc.name,
        "result": &result.output, "success": result.success,
    })).await;
    emit_artifact_card(&tc.name, &result, sender).await;
    track_spiral(ls, &tc.name, &result);
    ctx.add_tool_result(tc, result);
    crate::web::handlers::platform_context::enforce_context_budget(
        &mut ctx.messages, state.model_spec.context_length,
    );
    emit_auto_verify_hint(ctx, &tc.name);
    ls.remaining_turns = ls.remaining_turns.saturating_sub(1);
    ls.total_iterations += 1;
}

/// Reject a tool call if budget is exhausted. Returns true if rejected.
fn reject_if_budget_exhausted(ctx: &mut ReactContext, ls: &LoopState, tc: &schema::ToolCall) -> bool {
    if ls.remaining_turns > 0 { return false; }
    tracing::warn!(tool = %tc.name, "Model tried to call tool with 0 budget — rejecting");
    ctx.messages.push(Message::assistant_tool_call(&tc.id, &tc.name, &tc.arguments));
    ctx.messages.push(Message::tool_result(&tc.id,
        "[REJECTED] Your turn budget is exhausted. You must call `reply_request` \
         to deliver your response, or `extend_turns` to request more turns. \
         No other tools are available until you assess and decide."
    ));
    true
}

/// Handle parallel ToolCalls.
async fn handle_parallel_tools(
    ctx: &mut ReactContext,
    ls: &mut LoopState,
    state: &AppState,
    tcs: &[schema::ToolCall],
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
) {
    tracing::info!(
        iteration = ls.total_iterations,
        count = tcs.len(),
        remaining = ls.remaining_turns,
        tools = %tcs.iter().map(|t| t.name.as_str()).collect::<Vec<_>>().join(", "),
        "ReAct: ToolCalls (parallel)"
    );
    if reject_parallel_if_budget_exhausted(ctx, ls, tcs) { return; }

    for tc in tcs {
        send_ws(sender, "tool_executing", &serde_json::json!({"name": &tc.name, "id": &tc.id})).await;
    }
    let futs: Vec<_> = tcs.iter().map(|tc| execute_tool_with_state(state, tc)).collect();
    let results = futures::future::join_all(futs).await;
    for (i, result) in results.iter().enumerate() {
        send_ws(sender, "tool_completed", &serde_json::json!({
            "id": &result.tool_call_id, "name": &result.name,
            "result": &result.output, "success": result.success,
        })).await;
        emit_artifact_card(&tcs[i].name, result, sender).await;
    }
    ls.last_fail_signature = None;
    ls.consecutive_fails = 0;
    let pairs: Vec<_> = tcs.iter().zip(results.into_iter()).collect();
    ctx.add_tool_results(pairs);
    crate::web::handlers::platform_context::enforce_context_budget(
        &mut ctx.messages, state.model_spec.context_length,
    );
    ls.remaining_turns = ls.remaining_turns.saturating_sub(1);
    ls.total_iterations += 1;
}

/// Reject all parallel tool calls if budget is exhausted.
fn reject_parallel_if_budget_exhausted(ctx: &mut ReactContext, ls: &LoopState, tcs: &[schema::ToolCall]) -> bool {
    if ls.remaining_turns > 0 { return false; }
    tracing::warn!(count = tcs.len(), "Model tried parallel calls with 0 budget — rejecting all");
    let call_refs: Vec<_> = tcs.iter()
        .map(|tc| (tc.id.as_str(), tc.name.as_str(), tc.arguments.as_str()))
        .collect();
    ctx.messages.push(Message::assistant_tool_calls(&call_refs));
    for tc in tcs {
        ctx.messages.push(Message::tool_result(&tc.id,
            "[REJECTED] Turn budget exhausted. Call reply_request or extend_turns."
        ));
    }
    true
}

/// Handle ImplicitReply (model returned text without tool calls). Returns true if loop should exit.
async fn handle_implicit_reply(
    ctx: &mut ReactContext,
    ls: &mut LoopState,
    state: &AppState,
    text: &str,
    thinking: &Option<String>,
    user_query: &str,
    session_id: &str,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
) -> bool {
    tracing::info!(
        iteration = ls.total_iterations,
        text_len = text.len(),
        text_empty = text.trim().is_empty(),
        "ReAct: ImplicitReply"
    );
    if text.trim().is_empty() {
        ls.empty_reply_retries += 1;
        tracing::warn!(iteration = ls.total_iterations, retry = ls.empty_reply_retries, "ReAct: Empty ImplicitReply");
        if ls.empty_reply_retries <= 2 {
            ctx.messages.push(Message::text("system",
                "[EMPTY RESPONSE DETECTED] You returned no text and no tool calls. \
                 You MUST either call `reply_request` with your response, or call a tool. \
                 Do NOT return empty. Deliver your report now via `reply_request`."
            ));
            return false; // Retry
        }
        tracing::error!(iteration = ls.total_iterations, "ReAct: Empty reply after 2 retries — forcing completion");
    }
    if let Some(ref t) = thinking {
        send_ws(sender, "thinking_delta", &serde_json::json!({"content": t})).await;
    }
    deliver_final_reply(state, text, user_query, session_id, sender).await;
    true
}

/// Handle an iteration error.
async fn handle_iteration_error(
    e: &anyhow::Error,
    ls: &LoopState,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
) {
    tracing::error!(error = ?e, iteration = ls.total_iterations, "ReAct iteration inference failed");
    send_ws(sender, "error", &serde_json::json!({"message": format!("ReAct error: {}", e)})).await;
    send_ws(sender, "done", &serde_json::json!({})).await;
}

/// Track spiral detection: consecutive identical tool failures.
fn track_spiral(ls: &mut LoopState, tool_name: &str, result: &schema::ToolResult) {
    if !result.success {
        let sig = format!("{}:{}", tool_name, result.output);
        if ls.last_fail_signature.as_deref() == Some(&sig) {
            ls.consecutive_fails += 1;
            // This is a retry of the same failure — count as auto-fix attempt
            ls.progress.error_auto_fixed();
        } else {
            ls.last_fail_signature = Some(sig);
            ls.consecutive_fails = 1;
        }
        ls.progress.task_failed(tool_name);
    } else {
        ls.last_fail_signature = None;
        ls.consecutive_fails = 0;
        ls.progress.task_completed(tool_name);
    }
}

/// Emit artifact card if this was a successful create_artifact call.
async fn emit_artifact_card(
    tool_name: &str,
    result: &schema::ToolResult,
    sender: &mut futures_util::stream::SplitSink<WebSocket, WsMessage>,
) {
    if tool_name == "create_artifact" && result.success {
        if let Ok(artifact) = serde_json::from_str::<serde_json::Value>(&result.output) {
            send_ws(sender, "artifact_created", &artifact).await;
        }
    }
}

/// Inject auto-verify hint after code-modifying tools.
fn emit_auto_verify_hint(ctx: &mut ReactContext, tool_name: &str) {
    if matches!(tool_name, "codebase_edit" | "file_write") {
        ctx.messages.push(Message::text("system",
            "[AUTO-VERIFY HINT] You just modified code. Consider calling `verify_code` \
             to confirm build + tests pass before proceeding. If verify fails, fix errors \
             and re-verify until clean."
        ));
    }
}

/// Execute a tool call, dispatching memory tools through AppState.
async fn execute_tool_with_state(
    state: &AppState,
    tc: &schema::ToolCall,
) -> schema::ToolResult {
    crate::web::tool_dispatch::execute_tool_with_state(state, tc).await
}


/// Build tool context from message history — delegated to ws_react_helpers.
fn build_tool_context(messages: &[Message]) -> String {
    super::ws_react_helpers::build_tool_context(messages)
}

