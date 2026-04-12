// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! GRPO Reward Functions — composable reward signals for self-play scoring.
//!
//! Each RewardFn returns a scalar score in [0.0, 1.0] for a given response.
//! Rewards are combined via weighted sum in CompositeReward.

/// A single reward function that scores a response.
pub trait RewardFn: Send + Sync {
    /// Score a response given the prompt. Returns [0.0, 1.0].
    fn score(&self, prompt: &str, response: &str) -> f64;

    /// Human-readable name for logging.
    fn name(&self) -> &str;
}

/// Weighted composite of multiple reward functions.
pub struct CompositeReward {
    components: Vec<(Box<dyn RewardFn>, f64)>,
}

impl CompositeReward {
    pub fn new() -> Self {
        Self { components: Vec::new() }
    }

    /// Add a reward function with a weight.
    pub fn add(mut self, reward: Box<dyn RewardFn>, weight: f64) -> Self {
        self.components.push((reward, weight));
        self
    }

    /// Score a response with all components.
    pub fn score(&self, prompt: &str, response: &str) -> f64 {
        let total_weight: f64 = self.components.iter().map(|(_, w)| w).sum();
        if total_weight == 0.0 {
            return 0.0;
        }

        let weighted_sum: f64 = self.components
            .iter()
            .map(|(rf, w)| rf.score(prompt, response) * w)
            .sum();

        weighted_sum / total_weight
    }

    /// Score with per-component breakdown for logging.
    pub fn score_detailed(&self, prompt: &str, response: &str) -> Vec<(String, f64)> {
        self.components
            .iter()
            .map(|(rf, _)| (rf.name().to_string(), rf.score(prompt, response)))
            .collect()
    }
}

// ── Built-in Reward Functions ──────────────────────────────────────

/// Rewards responses that reference tool usage.
pub struct ToolUsageReward;

impl RewardFn for ToolUsageReward {
    fn score(&self, _prompt: &str, response: &str) -> f64 {
        let tool_patterns = ["✅", "❌", "tool_call", "Tool:", "[TOOL", "```"];
        let matches: usize = tool_patterns.iter()
            .map(|p| response.matches(p).count())
            .sum();
        (matches as f64 / 3.0).min(1.0)
    }

    fn name(&self) -> &str { "tool_usage" }
}

/// Rewards properly formatted responses (no raw JSON/XML leak).
pub struct FormatReward;

impl RewardFn for FormatReward {
    fn score(&self, _prompt: &str, response: &str) -> f64 {
        let bad_patterns = [
            "```json\n{\"tool_call",
            "<tool_call>",
            "{\"name\":",
            "\"arguments\":{",
        ];

        let violations: usize = bad_patterns.iter()
            .filter(|p| response.contains(**p))
            .count();

        if violations == 0 { 1.0 }
        else if violations == 1 { 0.3 }
        else { 0.0 }
    }

    fn name(&self) -> &str { "format" }
}

/// Rewards responses in the goldilocks length zone (200-2000 chars).
pub struct LengthReward;

impl RewardFn for LengthReward {
    fn score(&self, _prompt: &str, response: &str) -> f64 {
        let len = response.len();
        if (200..=2000).contains(&len) { 1.0 }
        else if (100..200).contains(&len) || (2000..3000).contains(&len) { 0.5 }
        else if len < 50 { 0.0 }
        else { 0.3 }
    }

    fn name(&self) -> &str { "length" }
}

/// Rewards responses that directly address the prompt (keyword overlap).
pub struct CoherenceReward;

impl RewardFn for CoherenceReward {
    fn score(&self, prompt: &str, response: &str) -> f64 {
        let prompt_words: Vec<String> = prompt.split_whitespace()
            .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
            .filter(|w| w.len() > 3)
            .collect();

        if prompt_words.is_empty() {
            return 0.5;
        }

        let response_lower = response.to_lowercase();
        let matches = prompt_words.iter()
            .filter(|w| response_lower.contains(w.as_str()))
            .count();

        (matches as f64 / prompt_words.len() as f64).min(1.0)
    }

    fn name(&self) -> &str { "coherence" }
}

/// Build the default composite reward function.
pub fn default_rewards() -> CompositeReward {
    CompositeReward::new()
        .add(Box::new(ToolUsageReward), 1.0)
        .add(Box::new(FormatReward), 1.5)
        .add(Box::new(LengthReward), 0.5)
        .add(Box::new(CoherenceReward), 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_usage_reward_with_tools() {
        let r = ToolUsageReward;
        let score = r.score("", "I used the ✅ tool and got ❌ results");
        assert!(score > 0.0);
    }

    #[test]
    fn test_tool_usage_reward_no_tools() {
        let r = ToolUsageReward;
        let score = r.score("", "Hello, how can I help you?");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_format_reward_clean() {
        let r = FormatReward;
        assert_eq!(r.score("", "This is a clean response."), 1.0);
    }

    #[test]
    fn test_format_reward_leaked_json() {
        let r = FormatReward;
        let score = r.score("", "{\"name\": \"tool\", \"arguments\":{\"x\":1}}");
        assert!(score < 1.0);
    }

    #[test]
    fn test_length_reward_goldilocks() {
        let r = LengthReward;
        assert_eq!(r.score("", &"a".repeat(500)), 1.0);
    }

    #[test]
    fn test_length_reward_too_short() {
        let r = LengthReward;
        assert_eq!(r.score("", "hi"), 0.0);
    }

    #[test]
    fn test_coherence_reward_matching() {
        let r = CoherenceReward;
        let score = r.score("What is Rust programming?", "Rust is a systems programming language.");
        assert!(score > 0.5);
    }

    #[test]
    fn test_composite_reward() {
        let composite = default_rewards();
        let score = composite.score("How do I use tools?", &"a".repeat(500));
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_composite_detailed_scoring() {
        let composite = default_rewards();
        let details = composite.score_detailed("test prompt", "test response");
        assert_eq!(details.len(), 4);
        assert!(details.iter().all(|(_, s)| *s >= 0.0 && *s <= 1.0));
    }
}
