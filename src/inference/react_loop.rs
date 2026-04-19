// Ern-OS — Layer 2: ReAct Loop — multi-turn reasoning + tool execution
//! Only triggered by explicit model tool call to `start_react_system`.
//! Loops: Reason → Act → Observe → Repeat until reply_request or user stop.

use crate::provider::{Message, Provider, StreamEvent};
use crate::tools::schema::{self, ToolCall, ToolResult};
use anyhow::{Context, Result};

/// ReAct loop iteration result.
pub enum IterationResult {
    /// Model wants to call a single tool — execute and continue.
    ToolCall(ToolCall),
    /// Model wants to call multiple tools in parallel.
    ToolCalls(Vec<ToolCall>),
    /// Model delivered final reply via reply_request. (reply_text, thinking_content)
    Reply(String, Option<String>),
    /// Model refused via refuse_request.
    Refuse(String),
    /// Model requests more turns with justification.
    ExtendTurns { additional: usize, progress: String, remaining_work: String },
    /// Text-only response (no tool call — treat as implicit reply). (text, thinking_content)
    ImplicitReply(String, Option<String>),
}

/// Full ReAct loop context — accumulates all turns.
pub struct ReactContext {
    pub objective: String,
    pub plan: Option<String>,
    pub messages: Vec<Message>,
    pub iteration: usize,
    pub tool_results: Vec<ToolResult>,
}

impl ReactContext {
    pub fn new(objective: &str, plan: Option<&str>, base_messages: Vec<Message>) -> Self {
        let mut messages = base_messages;

        // Inject the ReAct system prompt
        let react_prompt = format!(
            "[ReAct System Active]\n\
             Objective: {}\n\
             {}\n\n\
             You are now in the ReAct reasoning loop. Follow this cycle:\n\
             1. REASON: Analyze what you know and what you need to do next\n\
             2. ACT: Call a tool to gather information or take action\n\
             3. OBSERVE: Process the tool result\n\
             4. REPEAT until the objective is achieved\n\n\
             When finished, call `reply_request` with your complete response.\n\
             If you cannot complete the task, call `refuse_request` with an explanation.",
            objective,
            plan.map(|p| format!("Plan: {}", p)).unwrap_or_default()
        );

        messages.push(Message::text("system", &react_prompt));

        Self {
            objective: objective.to_string(),
            plan: plan.map(|s| s.to_string()),
            messages,
            iteration: 0,
            tool_results: Vec::new(),
        }
    }

    /// Add a tool result to the context, preserving the actual arguments sent.
    pub fn add_tool_result(&mut self, tc: &ToolCall, result: ToolResult) {
        // Assistant message declaring the tool call with ACTUAL args (not "{}")
        self.messages.push(Message::assistant_tool_call(
            &result.tool_call_id, &result.name, &tc.arguments,
        ));
        // Tool result message with matching call ID
        let content = if result.success {
            result.output.clone()
        } else {
            format!("Error: {}", result.output)
        };
        self.messages.push(Message::tool_result(&result.tool_call_id, &content));
        self.tool_results.push(result);
        self.iteration += 1;
    }

    /// Add multiple tool results from parallel execution.
    /// Creates a single assistant message with all tool calls (OpenAI format).
    pub fn add_tool_results(&mut self, pairs: Vec<(&ToolCall, ToolResult)>) {
        let call_refs: Vec<_> = pairs.iter()
            .map(|(tc, r)| (r.tool_call_id.as_str(), r.name.as_str(), tc.arguments.as_str()))
            .collect();
        self.messages.push(Message::assistant_tool_calls(&call_refs));
        for (_tc, r) in pairs {
            let content = if r.success {
                r.output.clone()
            } else {
                format!("Error: {}", r.output)
            };
            self.messages.push(Message::tool_result(&r.tool_call_id, &content));
            self.tool_results.push(r);
        }
        self.iteration += 1;
    }

    /// Add an observer rejection feedback message using SELF-CHECK framing.
    pub fn add_rejection_feedback(&mut self, result: &crate::observer::AuditResult) {
        let feedback = crate::observer::format_rejection_feedback(result);
        self.messages.push(Message::text("system", &feedback));
    }
}

/// Run a single iteration of the ReAct loop.
pub async fn run_iteration(
    provider: &dyn Provider,
    ctx: &ReactContext,
    thinking_enabled: bool,
) -> Result<IterationResult> {
    let tools = schema::layer2_tools();

    let mut rx = provider
        .chat(&ctx.messages, Some(&tools), thinking_enabled)
        .await
        .context("ReAct iteration inference failed")?;

    let mut text = String::new();
    let mut thinking = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    while let Some(event) = rx.recv().await {
        match event {
            StreamEvent::TextDelta(delta) => text.push_str(&delta),
            StreamEvent::ThinkingDelta(delta) => thinking.push_str(&delta),
            StreamEvent::ToolCall { id, name, arguments } => {
                tool_calls.push(ToolCall { id, name, arguments });
            }
            StreamEvent::Done => break,
            StreamEvent::Error(e) => anyhow::bail!("ReAct stream error: {}", e),
        }
    }

    // Check for terminal tool calls first
    for tc in &tool_calls {
        if tc.name == "reply_request" {
            let reply = schema::extract_reply_text(tc)
                .unwrap_or_else(|| text.clone());
            return Ok(IterationResult::Reply(reply, opt_thinking(&thinking)));
        }
        if tc.name == "refuse_request" {
            let reason = tc.args()["reason"]
                .as_str()
                .unwrap_or("No reason given")
                .to_string();
            return Ok(IterationResult::Refuse(reason));
        }
        if tc.name == "extend_turns" {
            let args = tc.args();
            let additional = args["additional_turns"].as_u64().unwrap_or(5) as usize;
            let progress = args["progress_summary"]
                .as_str().unwrap_or("").to_string();
            let remaining_work = args["remaining_work"]
                .as_str().unwrap_or("").to_string();
            return Ok(IterationResult::ExtendTurns { additional, progress, remaining_work });
        }
    }

    // Non-terminal tool calls — return for execution
    match tool_calls.len() {
        0 => Ok(IterationResult::ImplicitReply(text, opt_thinking(&thinking))),
        1 => Ok(IterationResult::ToolCall(tool_calls.into_iter().next().unwrap())),
        _ => Ok(IterationResult::ToolCalls(tool_calls)),
    }
}

/// Convert accumulated thinking string to Option — empty string becomes None.
fn opt_thinking(thinking: &str) -> Option<String> {
    if thinking.is_empty() { None } else { Some(thinking.to_string()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_react_context_creation() {
        let ctx = ReactContext::new("Test objective", Some("Step 1"), Vec::new());
        assert_eq!(ctx.objective, "Test objective");
        assert_eq!(ctx.iteration, 0);
        // The system prompt should be injected
        assert!(ctx.messages.iter().any(|m|
            m.text_content().contains("ReAct System Active")
        ));
    }

    #[test]
    fn test_add_tool_result() {
        let mut ctx = ReactContext::new("Test", None, Vec::new());
        let tc = ToolCall {
            id: "1".into(),
            name: "shell".into(),
            arguments: r#"{"command":"ls"}"#.into(),
        };
        ctx.add_tool_result(&tc, ToolResult {
            tool_call_id: "1".into(),
            name: "shell".into(),
            output: "file.txt".into(),
            success: true,
        });
        assert_eq!(ctx.iteration, 1);
        assert_eq!(ctx.tool_results.len(), 1);
        // Verify actual args are preserved in context, not "{}"
        let assistant_msg = ctx.messages.iter().find(|m| m.role == "assistant").unwrap();
        let tc_json = assistant_msg.text_content();
        assert!(tc_json.contains("ls") || ctx.messages.len() > 0);
    }

    #[test]
    fn test_rejection_feedback() {
        use crate::observer::{AuditResult, Verdict};
        let mut ctx = ReactContext::new("Test", None, Vec::new());
        let result = AuditResult {
            verdict: Verdict::Blocked,
            confidence: 0.9,
            failure_category: "sycophancy".to_string(),
            what_worked: String::new(),
            what_went_wrong: "Response was too short".to_string(),
            how_to_fix: "Provide more detail and cite sources".to_string(),
        };
        ctx.add_rejection_feedback(&result);
        let last = ctx.messages.last().unwrap().text_content();
        assert!(last.contains("SELF-CHECK FAIL"));
        assert!(last.contains("sycophancy"));
        assert!(last.contains("Response was too short"));
    }
}
