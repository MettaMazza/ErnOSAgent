//! Sub-agent execution — isolated ReAct loops with restricted tool sets.
//! Prevents context pollution; parent only receives a summary.

use crate::inference::react_loop::{self, ReactContext, IterationResult};
use crate::provider::{Message, Provider};
use crate::tools::schema::{self, ToolResult};
use crate::web::state::AppState;
use anyhow::{Context, Result};

/// Configuration for a sub-agent spawn.
pub struct SubAgentConfig {
    pub task: String,
    pub allowed_tools: Vec<String>,
    pub max_turns: usize,
}

/// Result returned to the parent context.
#[derive(Debug)]
pub struct SubAgentResult {
    pub summary: String,
    pub success: bool,
    pub turns_used: usize,
    pub tool_calls_made: Vec<String>,
}

/// Run an isolated sub-agent loop with restricted tools.
pub async fn run_sub_agent(
    provider: &dyn Provider,
    config: SubAgentConfig,
    state: &AppState,
) -> Result<SubAgentResult> {
    let system_prompt = format!(
        "[Sub-Agent Task]\n\
         You are a focused sub-agent with a specific task.\n\
         Task: {}\n\n\
         Available tools: {}\n\n\
         Complete this task efficiently. When done, call `reply_request` \
         with a concise summary of your findings or actions.\n\
         Do NOT call tools outside your allowed set — they will be rejected.",
        config.task,
        config.allowed_tools.join(", ")
    );

    let base_messages = vec![Message::text("system", &system_prompt)];
    let mut ctx = ReactContext::new(&config.task, None, base_messages);
    let mut turns_used = 0usize;
    let mut tool_names = Vec::new();

    for _ in 0..config.max_turns {
        match react_loop::run_iteration(provider, &ctx, true).await
            .context("Sub-agent iteration failed")?
        {
            IterationResult::Reply(reply, _thinking) => {
                return Ok(SubAgentResult {
                    summary: reply,
                    success: true,
                    turns_used,
                    tool_calls_made: tool_names,
                });
            }
            IterationResult::Refuse(reason) => {
                return Ok(SubAgentResult {
                    summary: format!("Sub-agent refused: {}", reason),
                    success: false,
                    turns_used,
                    tool_calls_made: tool_names,
                });
            }
            IterationResult::ToolCall(tc) => {
                if !is_tool_allowed(&tc.name, &config.allowed_tools) {
                    tracing::warn!(
                        tool = %tc.name,
                        "Sub-agent tried to call restricted tool — rejecting"
                    );
                    ctx.messages.push(Message::assistant_tool_call(&tc.id, &tc.name, &tc.arguments));
                    ctx.messages.push(Message::tool_result(&tc.id,
                        &format!("[REJECTED] Tool '{}' is not in your allowed set: {}",
                            tc.name, config.allowed_tools.join(", "))
                    ));
                    continue;
                }
                tool_names.push(tc.name.clone());
                let result = execute_tool(state, &tc).await;
                ctx.add_tool_result(&tc, result);
                turns_used += 1;
            }
            IterationResult::ToolCalls(tcs) => {
                let mut pairs: Vec<(&schema::ToolCall, ToolResult)> = Vec::new();
                for tc in &tcs {
                    if !is_tool_allowed(&tc.name, &config.allowed_tools) {
                        pairs.push((tc, ToolResult {
                            tool_call_id: tc.id.clone(),
                            name: tc.name.clone(),
                            output: format!("[REJECTED] Tool '{}' not allowed", tc.name),
                            success: false,
                        }));
                    } else {
                        tool_names.push(tc.name.clone());
                        pairs.push((tc, execute_tool(state, tc).await));
                    }
                }
                ctx.add_tool_results(pairs);
                turns_used += 1;
            }
            IterationResult::ExtendTurns { .. } => {
                // Sub-agents don't get extensions — force reply
                ctx.messages.push(Message::text("system",
                    "[Sub-agent budget is fixed. Deliver your reply now via reply_request.]"
                ));
            }
            IterationResult::ImplicitReply(text, _thinking) => {
                return Ok(SubAgentResult {
                    summary: text,
                    success: true,
                    turns_used,
                    tool_calls_made: tool_names,
                });
            }
        }
    }

    Ok(SubAgentResult {
        summary: format!(
            "Sub-agent exhausted {} turns without delivering a reply. \
             Tools called: {}",
            config.max_turns,
            tool_names.join(", ")
        ),
        success: false,
        turns_used,
        tool_calls_made: tool_names,
    })
}

/// Check if a tool name is in the allowed list.
fn is_tool_allowed(name: &str, allowed: &[String]) -> bool {
    // Terminal tools are always allowed
    if schema::is_loop_terminator(name) {
        return true;
    }
    allowed.iter().any(|a| a == name)
}

/// Execute a tool call through the standard dispatch.
async fn execute_tool(state: &AppState, tc: &schema::ToolCall) -> ToolResult {
    crate::web::tool_dispatch::execute_tool_with_state(state, tc).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_allowed() {
        let allowed = vec!["web_search".to_string(), "file_read".to_string()];
        assert!(is_tool_allowed("web_search", &allowed));
        assert!(is_tool_allowed("file_read", &allowed));
        assert!(!is_tool_allowed("run_bash_command", &allowed));
        // Terminal tools always allowed
        assert!(is_tool_allowed("reply_request", &allowed));
        assert!(is_tool_allowed("refuse_request", &allowed));
    }

    #[test]
    fn test_sub_agent_config() {
        let config = SubAgentConfig {
            task: "Search for Rust web frameworks".to_string(),
            allowed_tools: vec!["web_search".to_string()],
            max_turns: 5,
        };
        assert_eq!(config.max_turns, 5);
        assert_eq!(config.allowed_tools.len(), 1);
    }
}
