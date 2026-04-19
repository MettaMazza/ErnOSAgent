// Ern-OS — Observer verdict parser
//! Parses the 6-field ALLOWED/BLOCKED verdict from observer JSON output.

use crate::observer::{AuditResult, Verdict};

/// Parse an observer verdict from the model's JSON response.
/// Handles ALLOWED/BLOCKED format with 6 fields.
/// Falls back to fail-open on parse failure (infrastructure problem, not bad candidate).
pub fn parse_verdict(response: &str) -> AuditResult {
    let json_str = extract_json(response);

    match serde_json::from_str::<serde_json::Value>(&json_str) {
        Ok(v) => {
            // Parse verdict — support both formats for robustness
            let verdict = match v["verdict"].as_str() {
                Some("BLOCKED") => Verdict::Blocked,
                Some("ALLOWED") => Verdict::Allowed,
                _ => {
                    // Fallback: try legacy "approved" field
                    if v["approved"].as_bool().unwrap_or(true) {
                        Verdict::Allowed
                    } else {
                        Verdict::Blocked
                    }
                }
            };

            let confidence = v["confidence"].as_f64().unwrap_or(0.5) as f32;
            let failure_category = v["failure_category"].as_str().unwrap_or("none").to_string();
            let what_worked = v["what_worked"].as_str().unwrap_or("").to_string();
            let what_went_wrong = v["what_went_wrong"].as_str().unwrap_or("").to_string();
            let how_to_fix = v["how_to_fix"].as_str().unwrap_or("").to_string();

            tracing::debug!(
                verdict = %verdict,
                confidence,
                category = %failure_category,
                "Observer verdict parsed"
            );

            AuditResult {
                verdict,
                confidence,
                failure_category,
                what_worked,
                what_went_wrong,
                how_to_fix,
            }
        }
        Err(e) => {
            // If we can't parse JSON, fail-open (infrastructure problem)
            tracing::warn!(
                error = %e,
                response_len = response.len(),
                "Observer: failed to parse verdict JSON — fail-open (ALLOWED)"
            );
            AuditResult::parse_error(&e.to_string())
        }
    }
}

/// Extract JSON from a response that might contain surrounding text.
fn extract_json(text: &str) -> String {
    // Try raw first
    if text.trim().starts_with('{') {
        return text.trim().to_string();
    }

    // Try to find JSON in code blocks
    if let Some(start) = text.find("```json") {
        if let Some(end) = text[start + 7..].find("```") {
            return text[start + 7..start + 7 + end].trim().to_string();
        }
    }
    if let Some(start) = text.find("```") {
        if let Some(end) = text[start + 3..].find("```") {
            let content = text[start + 3..start + 3 + end].trim();
            if content.starts_with('{') {
                return content.to_string();
            }
        }
    }

    // Try to find raw JSON braces
    if let Some(start) = text.find('{') {
        if let Some(end) = text.rfind('}') {
            return text[start..=end].to_string();
        }
    }

    text.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_allowed() {
        let response = r#"{"verdict": "ALLOWED", "confidence": 0.95, "failure_category": "none", "what_worked": "Good response", "what_went_wrong": "", "how_to_fix": ""}"#;
        let result = parse_verdict(response);
        assert!(result.verdict.is_allowed());
        assert!((result.confidence - 0.95).abs() < f32::EPSILON);
        assert_eq!(result.failure_category, "none");
    }

    #[test]
    fn test_parse_blocked() {
        let response = r#"{"verdict": "BLOCKED", "confidence": 0.9, "failure_category": "sycophancy", "what_worked": "Structure", "what_went_wrong": "Blind agreement", "how_to_fix": "Push back"}"#;
        let result = parse_verdict(response);
        assert!(!result.verdict.is_allowed());
        assert_eq!(result.failure_category, "sycophancy");
        assert_eq!(result.what_went_wrong, "Blind agreement");
    }

    #[test]
    fn test_parse_legacy_approved_format() {
        // Backwards compatibility: support old {approved: true} format
        let response = r#"{"approved": true, "score": 9, "reason": "Good"}"#;
        let result = parse_verdict(response);
        assert!(result.verdict.is_allowed());
    }

    #[test]
    fn test_parse_legacy_rejected_format() {
        let response = r#"{"approved": false, "score": 3, "reason": "Bad"}"#;
        let result = parse_verdict(response);
        assert!(!result.verdict.is_allowed());
    }

    #[test]
    fn test_parse_from_code_block() {
        let response = "Here's my verdict:\n```json\n{\"verdict\": \"ALLOWED\", \"confidence\": 0.8}\n```";
        let result = parse_verdict(response);
        assert!(result.verdict.is_allowed());
    }

    #[test]
    fn test_fail_open_on_garbage() {
        let result = parse_verdict("This is not JSON at all");
        assert!(result.verdict.is_allowed()); // Fail-open
        assert_eq!(result.failure_category, "parse_error");
    }

    #[test]
    fn test_extract_json_raw() {
        assert_eq!(extract_json(r#"{"a": 1}"#), r#"{"a": 1}"#);
    }

    #[test]
    fn test_parse_defaults() {
        let response = r#"{"verdict": "ALLOWED"}"#;
        let result = parse_verdict(response);
        assert!(result.verdict.is_allowed());
        assert_eq!(result.confidence, 0.5); // default
        assert!(result.what_worked.is_empty());
    }
}
