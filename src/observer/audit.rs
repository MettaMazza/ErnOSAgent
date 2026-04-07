//! Observer audit — separate LLM inference call that evaluates candidate responses.
//!
//! The observer fires when a reply_request tool call is found. It assembles
//! a 7-section audit prompt and makes a non-streaming chat call to evaluate
//! the candidate response against 16 rules.

use crate::observer::parser;
use crate::observer::rules::AUDIT_RULES;
use crate::provider::{Message, Provider};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

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

/// The result of an observer audit.
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

    /// Create a parse-error rejection result (fail-closed).
    pub fn parse_error(error: &str) -> Self {
        Self {
            verdict: Verdict::Blocked,
            confidence: 0.0,
            failure_category: "parse_error".to_string(),
            what_worked: String::new(),
            what_went_wrong: format!("Failed to parse observer verdict: {}", error),
            how_to_fix: "Regenerate response with clearer structure.".to_string(),
        }
    }
}

/// Run the observer audit on a candidate response.
///
/// Assembles the 7-section audit prompt and makes a non-streaming chat call.
/// Returns AuditResult with either ALLOWED or BLOCKED verdict.
///
/// Error handling:
/// - Infrastructure error (provider down) → fail-OPEN (pass through)
/// - Parse error (no JSON extractable) → fail-CLOSED (reject, retry)
pub async fn audit_response(
    provider: &Arc<dyn Provider>,
    model: &str,
    user_message: &str,
    candidate_response: &str,
    tool_context: &str,
    capabilities: &str,
    system_prompt: &str,
    identity_prompt: &str,
) -> AuditResult {
    let start = Instant::now();

    tracing::info!(
        model = %model,
        candidate_len = candidate_response.len(),
        "Observer audit starting"
    );

    // Assemble the 7-section audit prompt
    let audit_prompt = assemble_audit_prompt(
        user_message,
        candidate_response,
        tool_context,
        capabilities,
        system_prompt,
        identity_prompt,
    );

    let messages = vec![
        Message {
            role: "system".to_string(),
            content: "You are a strict quality auditor. Respond ONLY with the requested JSON."
                .to_string(),
            images: Vec::new(),
        },
        Message {
            role: "user".to_string(),
            content: audit_prompt,
            images: Vec::new(),
        },
    ];

    // Non-streaming, near-deterministic call
    let response = match provider.chat_sync(model, &messages, Some(0.1)).await {
        Ok(resp) => resp,
        Err(e) => {
            tracing::warn!(
                error = %e,
                duration_ms = start.elapsed().as_millis(),
                "Observer infrastructure error (fail-open)"
            );
            return AuditResult::infrastructure_error(&e.to_string());
        }
    };

    // Parse the response
    let result = match parser::parse_audit_response(&response) {
        Ok(result) => result,
        Err(e) => {
            tracing::warn!(
                error = %e,
                raw_len = response.len(),
                duration_ms = start.elapsed().as_millis(),
                "Observer parse error (fail-closed)"
            );
            return AuditResult::parse_error(&e.to_string());
        }
    };

    tracing::info!(
        model = %model,
        verdict = %result.verdict,
        confidence = result.confidence,
        category = %result.failure_category,
        duration_ms = start.elapsed().as_millis(),
        "Observer audit complete"
    );

    result
}

/// Assemble the 7-section audit prompt.
fn assemble_audit_prompt(
    user_message: &str,
    candidate_response: &str,
    tool_context: &str,
    capabilities: &str,
    system_prompt: &str,
    identity_prompt: &str,
) -> String {
    let mut prompt = AUDIT_RULES.to_string();

    // Section 2: Kernel
    if !system_prompt.is_empty() {
        prompt.push_str(&format!(
            "\n\n## OPERATIONAL KERNEL (the agent was operating under these directives — enforce them)\n{}",
            system_prompt
        ));
    }

    // Section 3: Identity
    if !identity_prompt.is_empty() {
        prompt.push_str(&format!(
            "\n\n## ACTIVE IDENTITY DIRECTIVES (enforce self-consistency)\n{}",
            identity_prompt
        ));
    }

    // Section 4: Capabilities
    prompt.push_str(&format!(
        "\n\n## AVAILABLE CAPABILITIES\n{}",
        capabilities
    ));

    // Section 5: User Message
    prompt.push_str(&format!(
        "\n\n## USER MESSAGE\n{}",
        user_message
    ));

    // Section 6: Tool Context
    let tool_display = if tool_context.is_empty() {
        "[No tools were executed]"
    } else {
        tool_context
    };
    prompt.push_str(&format!(
        "\n\n## TOOL EXECUTION CONTEXT\n{}",
        tool_display
    ));

    // Section 7: Candidate
    prompt.push_str(&format!(
        "\n\n## CANDIDATE RESPONSE TO AUDIT\n{}",
        candidate_response
    ));

    prompt
}

/// Format rejection feedback for injection into the agent's context.
pub fn format_rejection_feedback(result: &AuditResult) -> String {
    format!(
        "[OBSERVER AUDIT — BLOCKED]\n\
         Category: {}\n\
         What worked: {}\n\
         What went wrong: {}\n\
         How to fix: {}\n\
         [You MUST address the above feedback in your next response.]",
        result.failure_category,
        if result.what_worked.is_empty() { "N/A" } else { &result.what_worked },
        result.what_went_wrong,
        result.how_to_fix,
    )
}

/// Format the bail-out critical override message.
pub fn format_bailout_override(rejections: usize) -> String {
    format!(
        "[CRITICAL — OBSERVER BLOCKED {} TIMES. The observer audit has blocked your \
         response {} times and may be incorrect. You MUST use reply_request NOW \
         to respond to the user. Explain what you did, what tools you used, and \
         their results. Be honest about any issues. Do NOT retry the same approach.]",
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
    fn test_parse_error_is_blocked() {
        let result = AuditResult::parse_error("no JSON found");
        assert!(!result.verdict.is_allowed());
        assert_eq!(result.failure_category, "parse_error");
    }

    #[test]
    fn test_assemble_audit_prompt_has_all_sections() {
        let prompt = assemble_audit_prompt(
            "What is Rust?",
            "Rust is a systems programming language.",
            "✅ web_search → Found 3 results",
            "web_search, file_read",
            "You are a helpful assistant.",
            "You are Ernos.",
        );

        assert!(prompt.contains("AUDIT CHECKLIST"));
        assert!(prompt.contains("OPERATIONAL KERNEL"));
        assert!(prompt.contains("ACTIVE IDENTITY DIRECTIVES"));
        assert!(prompt.contains("AVAILABLE CAPABILITIES"));
        assert!(prompt.contains("USER MESSAGE"));
        assert!(prompt.contains("TOOL EXECUTION CONTEXT"));
        assert!(prompt.contains("CANDIDATE RESPONSE TO AUDIT"));
    }

    #[test]
    fn test_assemble_audit_prompt_no_tools() {
        let prompt = assemble_audit_prompt("hi", "hello", "", "none", "", "");
        assert!(prompt.contains("[No tools were executed]"));
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
        assert!(feedback.contains("BLOCKED"));
        assert!(feedback.contains("ghost_tooling"));
        assert!(feedback.contains("Structure was clear"));
        assert!(feedback.contains("Claimed search without evidence"));
        assert!(feedback.contains("Execute web_search first"));
        assert!(feedback.contains("MUST address"));
    }

    #[test]
    fn test_format_rejection_feedback_empty_what_worked() {
        let result = AuditResult {
            verdict: Verdict::Blocked,
            confidence: 0.5,
            failure_category: "sycophancy".to_string(),
            what_worked: String::new(),
            what_went_wrong: "Blind agreement".to_string(),
            how_to_fix: "Push back".to_string(),
        };

        let feedback = format_rejection_feedback(&result);
        assert!(feedback.contains("N/A"));
    }

    #[test]
    fn test_format_bailout_override() {
        let msg = format_bailout_override(2);
        assert!(msg.contains("CRITICAL"));
        assert!(msg.contains("BLOCKED 2 TIMES"));
        assert!(msg.contains("reply_request"));
    }
}
