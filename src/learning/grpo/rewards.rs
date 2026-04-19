// Ern-OS — GRPO reward scoring
//! Multi-signal reward model using text quality heuristics.

/// Score a group of candidate responses using multiple quality signals.
pub fn score_group(candidates: &[String], query: &str) -> Vec<f32> {
    candidates.iter().map(|c| score_single(c, query)).collect()
}

/// Score a single response on multiple quality dimensions.
fn score_single(response: &str, query: &str) -> f32 {
    let mut score = 0.0f32;
    let len = response.len();

    // Length reward: prefer substantive responses (100-800 chars)
    score += if len < 20 { -1.0 }
        else if len < 100 { 0.2 }
        else if len < 800 { 0.8 }
        else if len < 2000 { 0.5 }
        else { 0.3 };

    // Relevance: check if response contains query terms
    let query_words: Vec<&str> = query.split_whitespace().collect();
    let response_lower = response.to_lowercase();
    let relevance = query_words.iter()
        .filter(|w| response_lower.contains(&w.to_lowercase()))
        .count() as f32 / query_words.len().max(1) as f32;
    score += relevance * 0.5;

    // Structure: reward paragraphs, lists, code blocks
    if response.contains('\n') { score += 0.2; }
    if response.contains("- ") || response.contains("1.") { score += 0.2; }
    if response.contains("```") { score += 0.1; }

    // Penalise: empty, error messages, repetition
    if response.trim().is_empty() { score = -2.0; }
    if response.contains("[error]") || response.contains("Error:") { score -= 0.5; }

    score
}

/// Normalize scores to advantages (subtract group mean, divide by std).
pub fn compute_advantages(scores: &[f32]) -> Vec<f32> {
    if scores.is_empty() { return Vec::new(); }
    let mean: f32 = scores.iter().sum::<f32>() / scores.len() as f32;
    let variance: f32 = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32;
    let std = variance.sqrt().max(1e-8);
    scores.iter().map(|s| (s - mean) / std).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advantages_zero_mean() {
        let scores = vec![1.0, 2.0, 3.0, 4.0];
        let advantages = compute_advantages(&scores);
        let sum: f32 = advantages.iter().sum();
        assert!(sum.abs() < 1e-5);
    }

    #[test]
    fn test_score_quality() {
        let good = "Rust is a systems programming language focused on safety, speed, and concurrency.";
        let bad = "idk";
        assert!(score_single(good, "Rust language") > score_single(bad, "Rust language"));
    }

    #[test]
    fn test_score_group() {
        let candidates = vec!["Good answer here".into(), "Bad".into()];
        let scores = score_group(&candidates, "test");
        assert_eq!(scores.len(), 2);
    }
}
