//! String utilities for safe text manipulation.

/// Truncates a string to a maximum number of characters, adding an ellipsis if truncated.
/// This function is safe for UTF-8 strings.
pub fn safe_truncate(text: &str, max_chars: usize) -> String {
    let char_count = text.chars().count();
    if char_count <= max_chars {
        text.to_string()
    } else {
        let truncated: String = text.chars().take(max_chars).collect();
        format!("{}...", truncated)
    }
}

/// Truncates a string to a maximum number of bytes, ensuring it stays on a character boundary.
/// If the byte index falls in the middle of a character, it truncates to the previous valid boundary.
pub fn truncate_to_boundary(text: &str, max_bytes: usize) -> String {
    if text.len() <= max_bytes {
        return text.to_string();
    }

    let mut end = max_bytes;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }

    format!("{}...", &text[..end])
}
