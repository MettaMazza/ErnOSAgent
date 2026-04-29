//! Thought spiral detection algorithms for the SSE stream parser.
//!
//! Uses three strategies to detect repetitive thinking patterns:
//! 1. Exact consecutive duplicate lines (3+ = spiral)
//! 2. Chunk-level repetition via FNV hashing (catches verbatim paragraph loops)
//! 3. Sliding window n-gram overlap (catches rephrased repetition)

/// Detect thought spirals — repetitive thinking patterns.
/// Returns true if the thinking content shows repetition.
///
/// Uses two strategies:
/// 1. Exact consecutive duplicate lines (3+ = spiral)
/// 2. Sliding window n-gram overlap (catches rephrased repetition)
pub fn detect_thought_spiral(thinking: &str) -> bool {
    if thinking.len() < 800 {
        return false;
    }

    let lines: Vec<&str> = thinking.lines().collect();
    if lines.len() < 6 {
        return false;
    }

    // Strategy 1: exact consecutive duplicates
    if has_consecutive_duplicates(&lines) {
        return true;
    }

    // Strategy 2: chunk-level repetition — catches large repeating paragraph blocks
    // The model often loops with 200-500 char blocks repeating verbatim
    if has_chunk_repetition(thinking) {
        return true;
    }

    // Strategy 3: sliding window n-gram overlap for rephrased loops
    has_ngram_repetition(thinking)
}

/// Detect repeating chunks of text — catches the common failure mode where
/// the model repeats entire paragraph blocks verbatim.
fn has_chunk_repetition(text: &str) -> bool {
    use std::collections::HashMap;

    // Only check once we have enough text
    if text.len() < 800 {
        return false;
    }

    // Hash 150-char chunks with 50-char step — if any chunk appears 3+ times,
    // the model is looping. This catches the exact pattern from the user's log.
    let chunk_size = 150;
    let step = 50;
    let mut chunk_counts: HashMap<u64, u32> = HashMap::new();

    let bytes = text.as_bytes();
    let mut i = 0;
    while i + chunk_size <= bytes.len() {
        // Simple FNV-like hash of the chunk for speed
        let mut hash: u64 = 14695981039346656037;
        for &b in &bytes[i..i + chunk_size] {
            hash ^= b as u64;
            hash = hash.wrapping_mul(1099511628211);
        }
        let count = chunk_counts.entry(hash).or_insert(0);
        *count += 1;
        if *count >= 3 {
            return true;
        }
        i += step;
    }

    false
}

/// Check for 3+ consecutive duplicate lines.
fn has_consecutive_duplicates(lines: &[&str]) -> bool {
    let mut consecutive_dupes = 0;
    for window in lines.windows(2) {
        if window[0].trim() == window[1].trim() && !window[0].trim().is_empty() {
            consecutive_dupes += 1;
            if consecutive_dupes >= 3 { return true; }
        } else {
            consecutive_dupes = 0;
        }
    }
    false
}

/// Detect rephrased repetition via word-level n-gram overlap.
/// Splits thinking into windows and compares n-gram sets.
fn has_ngram_repetition(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 100 { return false; }

    let window_size = 50; // 50-word sliding windows
    let step = 40; // overlap of 10 words
    let mut windows: Vec<std::collections::HashSet<String>> = Vec::new();

    let mut i = 0;
    while i + window_size <= words.len() {
        let ngrams = build_trigrams(&words[i..i + window_size]);
        windows.push(ngrams);
        i += step;
    }

    // Count how many windows have >60% overlap with a non-adjacent window
    let mut high_overlap_pairs = 0;
    for (i, a) in windows.iter().enumerate() {
        for b in windows.iter().skip(i + 2) {
            let overlap = jaccard_similarity(a, b);
            if overlap > 0.7 { high_overlap_pairs += 1; }
        }
    }

    // 4+ high-overlap pairs = spiral
    high_overlap_pairs >= 4
}

/// Build word trigrams from a slice.
fn build_trigrams(words: &[&str]) -> std::collections::HashSet<String> {
    words.windows(3)
        .map(|w| format!("{} {} {}", w[0].to_lowercase(), w[1].to_lowercase(), w[2].to_lowercase()))
        .collect()
}

/// Jaccard similarity between two sets.
fn jaccard_similarity(a: &std::collections::HashSet<String>, b: &std::collections::HashSet<String>) -> f64 {
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 { 0.0 } else { intersection as f64 / union as f64 }
}
