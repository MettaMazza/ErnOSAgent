// Ern-OS — Observer audit module
//! 19-rule battle-tested audit system. Uses the same model via
//! chat_sync (thinking disabled) for fast verdicts.
//! Ported from ErnOSAgent's production observer with full 7-section audit prompt.

pub mod rules;
pub mod parser;
pub mod insights;
pub mod skills;

use crate::provider::{Message, Provider};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// The verdict: ALLOWED or BLOCKED.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Verdict {
    Allowed,
    Blocked,
}

impl Verdict {
    pub fn is_allowed(&self) -> bool {
        matches!(self, Verdict::Allowed)
    }
}

impl std::fmt::Display for Verdict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Verdict::Allowed => write!(f, "ALLOWED"),
            Verdict::Blocked => write!(f, "BLOCKED"),
        }
    }
}

fn default_confidence() -> f32 {
    0.5
}

/// The result of an observer audit — 6-field structured verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditResult {
    pub verdict: Verdict,

    #[serde(default = "default_confidence")]
    pub confidence: f32,

    #[serde(default)]
    pub failure_category: String,

    #[serde(default)]
    pub what_worked: String,

    #[serde(default)]
    pub what_went_wrong: String,

    #[serde(default)]
    pub how_to_fix: String,
}

impl AuditResult {
    /// Create an infrastructure-error pass-through result (fail-open).
    pub fn infrastructure_error(error: &str) -> Self {
        Self {
            verdict: Verdict::Allowed,
            confidence: 0.0,
            failure_category: "infrastructure_error".to_string(),
            what_worked: String::new(),
            what_went_wrong: format!("Observer unavailable: {}", error),
            how_to_fix: String::new(),
        }
    }

    /// Create a parse-error pass-through result (fail-open).
    ///
    /// A parse error is an infrastructure problem (the Observer's JSON was
    /// garbled), NOT evidence that the candidate response is bad. Fail-open
    /// is the correct policy here — blocking a valid response because the
    /// auditor produced broken output is worse than passing it through.
    pub fn parse_error(error: &str) -> Self {
        Self {
            verdict: Verdict::Allowed,
            confidence: 0.0,
            failure_category: "parse_error".to_string(),
            what_worked: String::new(),
            what_went_wrong: format!("Failed to parse observer verdict: {}", error),
            how_to_fix: "Observer returned malformed JSON — response passed through.".to_string(),
        }
    }
}

/// The full output of an observer audit, including data needed for training.
pub struct AuditOutput {
    /// The parsed audit result (verdict, confidence, etc.).
    pub result: AuditResult,
    /// The Observer's raw text response (for SFT training).
    pub raw_response: String,
    /// The audit instruction sent to the Observer (for SFT training).
    pub audit_instruction: String,
}

/// Audit a response through the observer system.
///
/// Uses 1-to-1 context parity — same messages as the main inference.
/// The last user message is replaced with a 7-section structured audit prompt.
///
/// Returns `AuditOutput` containing the parsed result plus raw data for training.
///
/// Error handling:
/// - Infrastructure error (provider down) → fail-OPEN (pass through)
/// - Parse error (no JSON extractable) → fail-OPEN (pass through)
pub async fn audit_response(
    provider: &dyn Provider,
    conversation: &[Message],
    reply: &str,
    tool_context: &str,
    user_message: &str,
) -> Result<AuditOutput> {
    let start = std::time::Instant::now();

    tracing::info!(
        candidate_len = reply.len(),
        context_msgs = conversation.len(),
        "Observer audit starting (1-to-1 context)"
    );

    // Build 7-section observer messages
    let (messages, audit_instruction) = build_observer_messages(
        conversation, reply, tool_context, user_message,
    );

    let response = provider
        .chat_sync(&messages, None)
        .await
        .context("Observer audit inference failed")?;

    let result = parser::parse_verdict(&response);

    tracing::info!(
        verdict = %result.verdict,
        confidence = result.confidence,
        category = %result.failure_category,
        duration_ms = start.elapsed().as_millis() as u64,
        "Observer audit complete"
    );

    Ok(AuditOutput {
        result,
        raw_response: response,
        audit_instruction,
    })
}

/// Build the observer message list from the live context.
///
/// Strategy:
///   1. Keep the system message verbatim (identical to main chat)
///   2. Keep all messages up to the last user message verbatim
///   3. Replace the last user message with the structured audit instruction
///      (7 sections: rules + user message + capabilities + tool context + candidate + JSON)
///
/// This gives 100% context parity.
fn build_observer_messages(
    conversation: &[Message],
    candidate_response: &str,
    tool_context: &str,
    user_message: &str,
) -> (Vec<Message>, String) {
    // Load the observer rules
    let observer_rules = rules::OBSERVER_SYSTEM_PROMPT;

    let tool_display = if tool_context.is_empty() {
        "[No tools were executed in THIS TURN. \
         The candidate may correctly reference tools from PREVIOUS turns \
         visible in the conversation history above — that is NOT ghost tooling.]"
    } else {
        tool_context
    };

    let audit_instruction = format!(
        "{rules}\n\n\
         ## USER'S ORIGINAL MESSAGE\n{user_message}\n\n\
         ## TOOL EXECUTION CONTEXT (THIS TURN ONLY)\n{tool_display}\n\n\
         ## CANDIDATE RESPONSE TO AUDIT\n{candidate_response}\n\n\
         Respond with ONLY a JSON object matching the audit schema above.",
        rules = observer_rules,
    );

    // Find the index of the last user message
    let last_user_idx = conversation
        .iter()
        .rposition(|m| m.role == "user");

    let messages = match last_user_idx {
        Some(idx) => {
            // Keep all messages up to last user (exclusive), then the audit instruction
            let mut msgs: Vec<Message> = conversation[..idx].to_vec();
            msgs.push(Message::text("user", &audit_instruction));
            msgs
        }
        None => {
            // No user message — fallback to minimal 2-message form
            tracing::warn!("Observer: no user message found in context — using minimal fallback");
            vec![
                Message::text("system", "You are a strict quality auditor. Respond ONLY with the requested JSON."),
                Message::text("user", &audit_instruction),
            ]
        }
    };

    (messages, audit_instruction)
}

/// Format rejection feedback for injection into the agent's context.
/// Uses "SELF-CHECK FAIL" framing so the model treats it as internal
/// self-correction, not an external authority.
pub fn format_rejection_feedback(result: &AuditResult) -> String {
    format!(
        "[SELF-CHECK FAIL: INVISIBLE TO USER] Your output did not meet your own standards.\n\
         Category: {}\n\
         Why it failed: {}\n\
         How to fix it: {}\n\
         \n\
         You MUST rewrite your response immediately incorporating this feedback.",
        result.failure_category,
        result.what_went_wrong,
        result.how_to_fix,
    )
}

/// Format the bail-out critical override message.
/// Used when the observer has blocked N consecutive responses to break infinite loops.
pub fn format_bailout_override(rejections: usize) -> String {
    format!(
        "[CRITICAL — OBSERVER BLOCKED {} TIMES. The observer audit has blocked your \
         response {} times and may be incorrect. You MUST respond to the user NOW. \
         Explain what you did, what tools you used, and their results. \
         Be honest about any issues. Do NOT retry the same approach.]",
        rejections, rejections
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verdict_display() {
        assert_eq!(Verdict::Allowed.to_string(), "ALLOWED");
        assert_eq!(Verdict::Blocked.to_string(), "BLOCKED");
    }

    #[test]
    fn test_verdict_is_allowed() {
        assert!(Verdict::Allowed.is_allowed());
        assert!(!Verdict::Blocked.is_allowed());
    }

    #[test]
    fn test_verdict_serde_uppercase() {
        let json = r#""ALLOWED""#;
        let v: Verdict = serde_json::from_str(json).unwrap();
        assert_eq!(v, Verdict::Allowed);

        let json = r#""BLOCKED""#;
        let v: Verdict = serde_json::from_str(json).unwrap();
        assert_eq!(v, Verdict::Blocked);
    }

    #[test]
    fn test_audit_result_defaults() {
        let json = r#"{"verdict":"ALLOWED"}"#;
        let result: AuditResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.confidence, 0.5); // default
        assert!(result.failure_category.is_empty()); // default
    }

    #[test]
    fn test_infrastructure_error_is_allowed() {
        let result = AuditResult::infrastructure_error("timeout");
        assert!(result.verdict.is_allowed());
        assert_eq!(result.confidence, 0.0);
        assert_eq!(result.failure_category, "infrastructure_error");
    }

    #[test]
    fn test_parse_error_is_allowed() {
        let result = AuditResult::parse_error("no JSON found");
        assert!(result.verdict.is_allowed());
        assert_eq!(result.failure_category, "parse_error");
    }

    #[test]
    fn test_observer_messages_preserve_system_and_history() {
        let live = vec![
            Message::text("system", "You are Ernos."),
            Message::text("user", "Turn 1 question"),
            Message::text("assistant", "Turn 1 answer"),
            Message::text("user", "Turn 2 question"),
        ];
        let (msgs, _instruction) = build_observer_messages(&live, "candidate reply", "", "Turn 2 question");

        // System message must be identical
        assert_eq!(msgs[0].role, "system");

        // Prior turns preserved verbatim
        assert_eq!(msgs[1].role, "user");
        assert_eq!(msgs[2].role, "assistant");

        // Last message is the audit instruction (replaces the original user turn)
        let last = msgs.last().unwrap();
        assert_eq!(last.role, "user");
        assert!(last.content.as_str().unwrap_or("").contains("CANDIDATE RESPONSE TO AUDIT"));
        assert!(last.content.as_str().unwrap_or("").contains("candidate reply"));
        assert!(last.content.as_str().unwrap_or("").contains("USER'S ORIGINAL MESSAGE"));
        assert!(last.content.as_str().unwrap_or("").contains("Turn 2 question"));
        assert_eq!(msgs.len(), 4);
    }

    #[test]
    fn test_observer_messages_no_tools_marker() {
        let live = vec![
            Message::text("system", "sys"),
            Message::text("user", "hi"),
        ];
        let (msgs, _) = build_observer_messages(&live, "hello", "", "hi");
        let last = msgs.last().unwrap();
        assert!(last.content.as_str().unwrap_or("").contains("[No tools were executed in THIS TURN"));
    }

    #[test]
    fn test_observer_messages_fallback_when_no_user_message() {
        let live = vec![
            Message::text("system", "sys"),
        ];
        let (msgs, _) = build_observer_messages(&live, "candidate", "", "");
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, "system");
        assert_eq!(msgs[1].role, "user");
        assert!(msgs[1].content.as_str().unwrap_or("").contains("CANDIDATE RESPONSE TO AUDIT"));
    }

    #[test]
    fn test_format_rejection_feedback() {
        let result = AuditResult {
            verdict: Verdict::Blocked,
            confidence: 0.9,
            failure_category: "ghost_tooling".to_string(),
            what_worked: "Structure was clear".to_string(),
            what_went_wrong: "Claimed search without evidence".to_string(),
            how_to_fix: "Execute web_search first".to_string(),
        };

        let feedback = format_rejection_feedback(&result);
        assert!(feedback.contains("SELF-CHECK FAIL"));
        assert!(feedback.contains("ghost_tooling"));
        assert!(feedback.contains("Claimed search without evidence"));
        assert!(feedback.contains("Execute web_search first"));
        assert!(feedback.contains("MUST rewrite"));
    }

    #[test]
    fn test_format_bailout_override() {
        let msg = format_bailout_override(2);
        assert!(msg.contains("CRITICAL"));
        assert!(msg.contains("BLOCKED 2 TIMES"));
        assert!(msg.contains("respond to the user NOW"));
    }
}
