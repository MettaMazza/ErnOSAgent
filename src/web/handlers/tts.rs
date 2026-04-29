//! TTS proxy handler — proxies requests to local Kokoro ONNX TTS server.
//! Endpoint: POST /api/tts
//! The Kokoro server runs on localhost:8880 (OpenAI-compatible /v1/audio/speech).

use crate::web::state::AppState;
use axum::{extract::State, response::IntoResponse, Json};

/// POST /api/tts — Generate speech from text via local Kokoro server.
pub async fn synthesize(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> impl IntoResponse {
    let text = body["text"].as_str().unwrap_or("");
    let voice = body["voice"].as_str().unwrap_or("am_michael");
    let speed = body["speed"].as_f64().unwrap_or(1.0);

    if text.is_empty() {
        return axum::response::Response::builder()
            .status(400)
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                serde_json::json!({"error": "Missing text"}).to_string(),
            ))
            .expect("valid response builder");
    }

    let clean_text = sanitise_for_tts(text);
    if clean_text.is_empty() {
        return axum::response::Response::builder()
            .status(400)
            .header("content-type", "application/json")
            .body(axum::body::Body::from(
                serde_json::json!({"error": "Text empty after sanitisation"}).to_string(),
            ))
            .expect("valid response builder");
    }

    let kokoro_url = format!(
        "http://127.0.0.1:{}/v1/audio/speech",
        state.config.general.kokoro_port.unwrap_or(8880)
    );

    let payload = serde_json::json!({
        "model": "kokoro",
        "input": clean_text,
        "voice": voice,
        "response_format": "wav",
        "speed": speed,
    });

    let client = reqwest::Client::new();
    match client.post(&kokoro_url).json(&payload).send().await {
        Ok(resp) if resp.status().is_success() => {
            let bytes = resp.bytes().await.unwrap_or_default();
            axum::response::Response::builder()
                .status(200)
                .header("content-type", "audio/wav")
                .body(axum::body::Body::from(bytes))
                .expect("valid response builder")
        }
        Ok(resp) => {
            let status = resp.status().as_u16();
            let body = resp.text().await.unwrap_or_default();
            tracing::warn!(status, body = %body, "Kokoro TTS error");
            axum::response::Response::builder()
                .status(502)
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({"error": "Kokoro TTS error", "detail": body}).to_string(),
                ))
                .expect("valid response builder")
        }
        Err(e) => {
            tracing::warn!(error = %e, "Kokoro TTS unreachable");
            axum::response::Response::builder()
                .status(503)
                .header("content-type", "application/json")
                .body(axum::body::Body::from(
                    serde_json::json!({"error": "Kokoro TTS server unreachable", "hint": "Start with: python start-kokoro.py"}).to_string(),
                ))
                .expect("valid response builder")
        }
    }
}

/// GET /api/tts/status — Check if Kokoro TTS is available.
pub async fn tts_status(State(state): State<AppState>) -> impl IntoResponse {
    let port = state.config.general.kokoro_port.unwrap_or(8880);
    let url = format!("http://127.0.0.1:{}/v1/models", port);
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap_or_default();
    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            Json(serde_json::json!({"available": true, "port": port}))
        }
        _ => Json(serde_json::json!({"available": false, "port": port})),
    }
}

// ─── TTS Text Sanitiser ───────────────────────────────────────────────────────
// Strips markdown formatting, emojis, code blocks, URLs, and other artefacts
// that produce unlistenable output when read aloud by Kokoro TTS.

/// Clean text for TTS consumption — strip markdown, emojis, URLs, code blocks.
/// Public so voice.rs and video.rs can use it too.
pub fn sanitise_for_tts(text: &str) -> String {
    let mut s = text.to_string();

    // 1. Remove fenced code blocks entirely → "(code block omitted)"
    let code_block = regex::Regex::new(r"(?s)```[\w]*\n?.*?```").expect("code_block regex");
    s = code_block.replace_all(&s, " (code block omitted) ").to_string();

    // 2. Remove inline code backticks, replace underscores with spaces
    let inline_code = regex::Regex::new(r"`([^`]+)`").expect("inline_code regex");
    s = inline_code
        .replace_all(&s, |caps: &regex::Captures| caps[1].replace('_', " "))
        .to_string();

    // 3. Remove markdown images → "(image: alt text)"
    let images = regex::Regex::new(r"!\[([^\]]*)\]\([^)]+\)").expect("images regex");
    s = images.replace_all(&s, "(image: $1)").to_string();

    // 4. Remove markdown links → keep link text only
    let links = regex::Regex::new(r"\[([^\]]+)\]\([^)]+\)").expect("links regex");
    s = links.replace_all(&s, "$1").to_string();

    // 5. Remove bare URLs
    let urls = regex::Regex::new(r"https?://\S+").expect("urls regex");
    s = urls.replace_all(&s, "").to_string();

    // 6. Remove markdown heading markers (# ## ### ####)
    let headings = regex::Regex::new(r"(?m)^#{1,6}\s+").expect("headings regex");
    s = headings.replace_all(&s, "").to_string();

    // 7. Remove bold/italic markers
    s = s.replace("***", "");
    s = s.replace("**", "");
    s = s.replace("__", "");
    // Single asterisks used as italic — remove carefully (not multiplication)
    let single_star = regex::Regex::new(r"\*([^*\n]+)\*").expect("single_star regex");
    s = single_star.replace_all(&s, "$1").to_string();

    // 8. Remove horizontal rules (---, ***, ___)
    let hr = regex::Regex::new(r"(?m)^[\s]*[-*_]{3,}\s*$").expect("hr regex");
    s = hr.replace_all(&s, "").to_string();

    // 9. Remove bullet/list markers (- item, * item)
    let bullets = regex::Regex::new(r"(?m)^[\s]*[-*]\s+").expect("bullets regex");
    s = bullets.replace_all(&s, "").to_string();

    // 10. Remove numbered list markers (1. item)
    let numbered = regex::Regex::new(r"(?m)^[\s]*\d+\.\s+").expect("numbered regex");
    s = numbered.replace_all(&s, "").to_string();

    // 11. Remove blockquote markers (> text)
    let blockquote = regex::Regex::new(r"(?m)^>\s*").expect("blockquote regex");
    s = blockquote.replace_all(&s, "").to_string();

    // 12. Remove table separator rows (|---|---|)
    let table_sep = regex::Regex::new(r"(?m)^\|[-:\s|]+\|\s*$").expect("table_sep regex");
    s = table_sep.replace_all(&s, "").to_string();

    // 13. Convert table pipes to commas for natural speech
    s = s.replace(" | ", ", ");
    s = s.replace("| ", "");
    s = s.replace(" |", "");
    s = s.replace('|', "");

    // 14. Remove HTML tags
    let html_tags = regex::Regex::new(r"<[^>]+>").expect("html_tags regex");
    s = html_tags.replace_all(&s, "").to_string();

    // 15. Remove emojis and common Unicode symbols
    // Covers emoji presentation forms, extended pictographics, and common symbols
    let emoji = regex::Regex::new(
        r"[\x{1F600}-\x{1F64F}\x{1F300}-\x{1F5FF}\x{1F680}-\x{1F6FF}\x{1F1E0}-\x{1F1FF}\x{2600}-\x{26FF}\x{2700}-\x{27BF}\x{FE00}-\x{FE0F}\x{1F900}-\x{1F9FF}\x{1FA00}-\x{1FA6F}\x{1FA70}-\x{1FAFF}\x{200D}\x{20E3}\x{E0020}-\x{E007F}\x{2B50}\x{2B55}\x{231A}-\x{231B}\x{23E9}-\x{23F3}\x{23F8}-\x{23FA}\x{25AA}-\x{25AB}\x{25B6}\x{25C0}\x{25FB}-\x{25FE}]"
    ).expect("emoji regex");
    s = emoji.replace_all(&s, "").to_string();

    // 16. Remove remaining standalone special characters that sound bad spoken
    // (tilde, backtick — asterisks/underscores already handled above)
    s = s.replace('~', "");
    s = s.replace('`', "");

    // 17. Replace underscores in remaining identifiers with spaces
    // (e.g. function_name → function name)
    let underscored = regex::Regex::new(r"\b(\w+)_(\w+)\b").expect("underscored regex");
    // Apply multiple times to handle chained underscores (a_b_c → a b c)
    for _ in 0..5 {
        let prev = s.clone();
        s = underscored.replace_all(&s, "$1 $2").to_string();
        if s == prev {
            break;
        }
    }

    // 18. Collapse multiple newlines → period + space (natural pause)
    let multi_newline = regex::Regex::new(r"\n{2,}").expect("multi_newline regex");
    s = multi_newline.replace_all(&s, ". ").to_string();

    // 19. Replace single newlines with spaces
    s = s.replace('\n', " ");

    // 20. Collapse excessive whitespace
    let multi_space = regex::Regex::new(r"[ \t]{2,}").expect("multi_space regex");
    s = multi_space.replace_all(&s, " ").to_string();

    // 21. Clean up awkward punctuation artefacts (". ." → ".", ",," → ",")
    let double_period = regex::Regex::new(r"\.(\s*\.)+").expect("double_period regex");
    s = double_period.replace_all(&s, ".").to_string();
    s = s.replace(",,", ",");
    s = s.replace(", ,", ",");

    s.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::sanitise_for_tts;

    #[test]
    fn strips_headings() {
        assert_eq!(sanitise_for_tts("### Chapter Five"), "Chapter Five");
        assert_eq!(sanitise_for_tts("# Title"), "Title");
        assert_eq!(sanitise_for_tts("#### Deep heading"), "Deep heading");
    }

    #[test]
    fn strips_bold_italic() {
        assert_eq!(sanitise_for_tts("**bold text**"), "bold text");
        assert_eq!(sanitise_for_tts("*italic text*"), "italic text");
        assert_eq!(
            sanitise_for_tts("This is **bold** and *italic*"),
            "This is bold and italic"
        );
    }

    #[test]
    fn strips_code_blocks() {
        let input = "Before\n```rust\nfn main() {\n    println!(\"hello\");\n}\n```\nAfter";
        let result = sanitise_for_tts(input);
        assert!(result.contains("(code block omitted)"));
        assert!(!result.contains("fn main"));
        assert!(result.contains("Before"));
        assert!(result.contains("After"));
    }

    #[test]
    fn strips_inline_code() {
        assert_eq!(sanitise_for_tts("Use `variable_name` here"), "Use variable name here");
    }

    #[test]
    fn strips_links_keeps_text() {
        assert_eq!(
            sanitise_for_tts("[click here](https://example.com)"),
            "click here"
        );
    }

    #[test]
    fn strips_images() {
        assert_eq!(
            sanitise_for_tts("![a cat](image.png)"),
            "(image: a cat)"
        );
    }

    #[test]
    fn strips_bare_urls() {
        let result = sanitise_for_tts("Visit https://example.com/long/path?q=test for more");
        assert!(!result.contains("https"));
        assert!(result.contains("Visit"));
        assert!(result.contains("for more"));
    }

    #[test]
    fn strips_horizontal_rules() {
        let result = sanitise_for_tts("Above\n\n---\n\nBelow");
        assert!(!result.contains("---"));
        assert!(result.contains("Above"));
        assert!(result.contains("Below"));
    }

    #[test]
    fn strips_bullet_markers() {
        let input = "- First item\n- Second item\n- Third item";
        let result = sanitise_for_tts(input);
        assert!(!result.starts_with('-'));
        assert!(result.contains("First item"));
        assert!(result.contains("Second item"));
    }

    #[test]
    fn strips_numbered_list_markers() {
        let input = "1. First\n2. Second\n3. Third";
        let result = sanitise_for_tts(input);
        assert!(result.contains("First"));
        assert!(!result.contains("1."));
    }

    #[test]
    fn strips_blockquotes() {
        assert_eq!(sanitise_for_tts("> quoted text"), "quoted text");
    }

    #[test]
    fn strips_table_formatting() {
        let input = "| Name | Age |\n|---|---|\n| Maria | 64 |";
        let result = sanitise_for_tts(input);
        assert!(!result.contains('|'));
        assert!(result.contains("Maria"));
    }

    #[test]
    fn strips_html_tags() {
        assert_eq!(sanitise_for_tts("Hello <br> world"), "Hello world");
    }

    #[test]
    fn replaces_underscores_in_identifiers() {
        assert_eq!(sanitise_for_tts("file_read_tool"), "file read tool");
    }

    #[test]
    fn collapses_whitespace() {
        let input = "Too    many     spaces";
        assert_eq!(sanitise_for_tts(input), "Too many spaces");
    }

    #[test]
    fn handles_complex_markdown() {
        let input = "### Overview\n\n\
                      Here is **important** info:\n\n\
                      - First [link](https://x.com)\n\
                      - Second `code_thing`\n\n\
                      ```python\nprint('hello')\n```\n\n\
                      > A wise quote\n\n\
                      Visit https://example.com for more.";
        let result = sanitise_for_tts(input);
        assert!(!result.contains('#'));
        assert!(!result.contains("**"));
        assert!(!result.contains("https"));
        assert!(!result.contains('`'));
        assert!(!result.contains("print('hello')"));
        assert!(result.contains("Overview"));
        assert!(result.contains("important"));
        assert!(result.contains("link"));
        assert!(result.contains("code thing"));
        assert!(result.contains("A wise quote"));
    }

    #[test]
    fn empty_after_stripping_returns_empty() {
        assert_eq!(sanitise_for_tts("---"), "");
        assert_eq!(sanitise_for_tts("```\ncode\n```"), "(code block omitted)");
    }
}
