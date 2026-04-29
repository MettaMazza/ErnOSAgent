// Ern-OS — Platform attachment ingestion
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Downloads and processes file attachments from platform messages.
//!
//! Security model:
//! - **Admin** users: files are saved to `data/uploads/` and extracted.
//! - **Non-admin** users: files are held in memory only, extracted, then dropped.
//!   Nothing touches disk.

use anyhow::Result;

/// Compute the inline character budget from the model's context length.
/// Allocates 25% of the context window for attachment text (~1 char per token).
/// This is derived from the model's reported context_length, not hardcoded.
pub fn inline_char_budget(context_length: usize) -> usize {
    // 25% of context * ~4 chars per token
    (context_length / 4) * 4
}

/// A processed attachment ready for inference injection.
pub struct ProcessedAttachment {
    pub filename: String,
    /// Extracted text content (for documents, code, data files).
    pub content_text: Option<String>,
    /// Base64 data URL for images (for multimodal message).
    pub image_data_url: Option<String>,
    /// Disk path where admin attachments are saved (None for non-admin).
    pub saved_path: Option<String>,
    /// Raw byte count of the original file before any truncation.
    pub original_size: usize,
}

impl ProcessedAttachment {
    /// Whether this attachment exceeds the inline character budget.
    pub fn exceeds_budget(&self, context_length: usize) -> bool {
        self.original_size > inline_char_budget(context_length)
    }
}

/// Download and process all attachment URLs from a platform message.
///
/// Returns processed attachments with extracted text and/or image data.
/// Admin attachments are persisted to `data/uploads/`; non-admin are memory-only.
pub async fn process_attachments(
    urls: &[String],
    is_admin: bool,
) -> Vec<ProcessedAttachment> {
    let mut results = Vec::new();

    for url in urls {
        match download_and_process(url, is_admin).await {
            Ok(att) => results.push(att),
            Err(e) => {
                tracing::warn!(url = %url, error = %e, "Failed to process attachment");
                results.push(ProcessedAttachment {
                    filename: filename_from_url(url),
                    content_text: Some(format!("[Failed to process attachment: {}]", e)),
                    image_data_url: None,
                    saved_path: None,
                    original_size: 0,
                });
            }
        }
    }

    results
}

/// Split processed attachments into images (for multimodal) and text content.
/// `context_length` is the model's reported context window size (in tokens).
pub fn split_processed(
    attachments: &[ProcessedAttachment],
    context_length: usize,
) -> (Vec<String>, String) {
    let mut images = Vec::new();
    let mut text_parts = Vec::new();

    for att in attachments {
        if let Some(ref data_url) = att.image_data_url {
            images.push(data_url.clone());
        }
        if let Some(ref text) = att.content_text {
            let budget = inline_char_budget(context_length);
            let budgeted = budget_text_content(text, &att.filename, budget);
            text_parts.push(format!("### {} ###\n{}", att.filename, budgeted));
        }
    }

    let combined = if text_parts.is_empty() {
        String::new()
    } else {
        text_parts.join("\n\n")
    };

    (images, combined)
}

/// Download a single attachment URL and process it.
async fn download_and_process(url: &str, is_admin: bool) -> Result<ProcessedAttachment> {
    let filename = filename_from_url(url);
    let ext = extension_from_filename(&filename);

    tracing::info!(url = %url, filename = %filename, ext = %ext, is_admin, "Downloading platform attachment");

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()?;
    let resp = client.get(url).send().await?;

    if !resp.status().is_success() {
        anyhow::bail!("HTTP {} downloading attachment", resp.status());
    }

    let bytes = resp.bytes().await?;
    tracing::info!(filename = %filename, size = bytes.len(), "Attachment downloaded");

    let saved_path = if is_admin {
        save_to_uploads(&filename, &bytes).await.ok()
    } else {
        None
    };

    if is_image_ext(&ext) {
        Ok(process_image_attachment(filename, &ext, &bytes, saved_path))
    } else {
        process_text_attachment(filename, &ext, &bytes, saved_path)
    }
}

/// Process an image attachment: encode as base64 data URL.
fn process_image_attachment(filename: String, ext: &str, bytes: &[u8], saved_path: Option<String>) -> ProcessedAttachment {
    let mime = image_mime(ext);
    let b64 = base64_encode(bytes);
    let data_url = format!("data:{};base64,{}", mime, b64);
    let size = bytes.len();
    ProcessedAttachment {
        filename,
        content_text: saved_path.as_ref().map(|p| format!("[Image saved to {}]", p)),
        image_data_url: Some(data_url),
        saved_path,
        original_size: size,
    }
}

/// Process a non-image attachment: extract text and build reference.
fn process_text_attachment(filename: String, ext: &str, bytes: &[u8], saved_path: Option<String>) -> Result<ProcessedAttachment> {
    let content = extract_text_from_bytes(bytes, &filename, ext)?;
    let original_size = content.len();

    // Admin path: save to disk AND inline the content.
    // The budget_text_content() call in split_processed() handles truncation.
    let content_with_ref = match saved_path {
        Some(ref path) => format!(
            "[FILE SAVED: {} — full file at {}]\n\n{}",
            filename, path, content
        ),
        None => content,
    };

    Ok(ProcessedAttachment {
        filename,
        content_text: Some(content_with_ref),
        image_data_url: None,
        saved_path,
        original_size,
    })
}

/// Extract text content from in-memory file bytes.
fn extract_text_from_bytes(bytes: &[u8], filename: &str, ext: &str) -> Result<String> {
    // Text-based formats: decode directly from memory
    if is_text_ext(ext) {
        let text = String::from_utf8_lossy(bytes);
        return Ok(text.to_string());
    }

    // Binary formats (pdf, docx, etc.): need temp file for file_extractor
    let tmp_dir = std::env::temp_dir().join("ernos_attachments");
    std::fs::create_dir_all(&tmp_dir)?;
    let tmp_path = tmp_dir.join(filename);
    std::fs::write(&tmp_path, bytes)?;

    let result = crate::tools::file_extractor::extract(
        &tmp_path.to_string_lossy(),
    );

    // Clean up temp file immediately
    let _ = std::fs::remove_file(&tmp_path);

    match result {
        Ok(extraction) => Ok(extraction.content),
        Err(e) => Ok(format!("[Failed to extract {}: {}]", filename, e)),
    }
}

/// Save attachment bytes to data/uploads/ (admin only).
async fn save_to_uploads(filename: &str, bytes: &[u8]) -> Result<String> {
    let upload_dir = std::path::PathBuf::from("data/uploads");
    tokio::fs::create_dir_all(&upload_dir).await?;

    let ext = extension_from_filename(filename);
    let safe_name = format!(
        "{}_{}.{}",
        chrono::Utc::now().format("%Y%m%d_%H%M%S"),
        &uuid::Uuid::new_v4().to_string()[..8],
        ext
    );

    let dest = upload_dir.join(&safe_name);
    tokio::fs::write(&dest, bytes).await?;
    tracing::info!(
        file = %safe_name, original = %filename,
        size = bytes.len(), "Admin attachment saved"
    );

    Ok(dest.display().to_string())
}

// ── Classification Helpers ─────────────────────────────────────────

fn filename_from_url(url: &str) -> String {
    url.rsplit('/').next()
        .unwrap_or("attachment")
        .split('?').next()
        .unwrap_or("attachment")
        .to_string()
}

fn extension_from_filename(filename: &str) -> String {
    std::path::Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase()
}

fn is_image_ext(ext: &str) -> bool {
    matches!(ext, "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp"
        | "tiff" | "tif" | "svg" | "ico" | "heic" | "heif" | "avif")
}

fn is_text_ext(ext: &str) -> bool {
    matches!(ext,
        "md" | "mdx" | "txt" | "text" | "log" | "csv" | "tsv"
        | "json" | "jsonl" | "ndjson" | "yaml" | "yml" | "toml" | "xml"
        | "ini" | "cfg" | "env" | "conf"
        | "rs" | "py" | "js" | "ts" | "tsx" | "jsx" | "go" | "java"
        | "c" | "cpp" | "h" | "hpp" | "rb" | "swift" | "kt" | "cs"
        | "php" | "lua" | "r" | "scala" | "zig" | "sh" | "bash" | "zsh"
        | "sql" | "graphql" | "proto" | "html" | "htm" | "css"
        | "rst" | "adoc" | "tex" | "org"
    )
}

fn image_mime(ext: &str) -> &'static str {
    match ext {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "svg" => "image/svg+xml",
        "tiff" | "tif" => "image/tiff",
        "ico" => "image/x-icon",
        "heic" | "heif" => "image/heic",
        "avif" => "image/avif",
        _ => "application/octet-stream",
    }
}

fn base64_encode(data: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(data)
}

/// Apply content budgeting — truncate large text to fit within the context window.
///
/// For files exceeding the budget, includes the first portion with a clear
/// pagination bookmark. The model can then use file_read to access the rest.
fn budget_text_content(text: &str, filename: &str, budget: usize) -> String {
    if text.len() <= budget {
        return text.to_string();
    }

    let total_chars = text.len();
    let total_lines = text.lines().count();

    // Find a clean break point near the budget limit (at a line boundary)
    let truncated = if let Some(break_pos) = text[..budget].rfind('\n') {
        &text[..break_pos]
    } else {
        &text[..budget]
    };

    let shown_lines = truncated.lines().count();

    tracing::info!(
        filename = %filename,
        total_chars,
        total_lines,
        shown_chars = truncated.len(),
        shown_lines,
        "Attachment truncated for context budget"
    );

    format!(
        "[Lines 1-{} of {}]\n{}\n\n[BOOKMARK: line {} — use file_read on the saved file with start_line={} to continue]",
        shown_lines, total_lines, truncated,
        shown_lines + 1, shown_lines + 1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filename_from_url() {
        assert_eq!(
            filename_from_url("https://cdn.discordapp.com/attachments/123/456/document.md"),
            "document.md"
        );
        assert_eq!(
            filename_from_url("https://cdn.discordapp.com/attachments/123/456/file.pdf?ex=abc"),
            "file.pdf"
        );
    }

    #[test]
    fn test_extension_from_filename() {
        assert_eq!(extension_from_filename("test.MD"), "md");
        assert_eq!(extension_from_filename("file.tar.gz"), "gz");
        assert_eq!(extension_from_filename("noext"), "");
    }

    #[test]
    fn test_is_image_ext() {
        assert!(is_image_ext("png"));
        assert!(is_image_ext("jpg"));
        assert!(!is_image_ext("md"));
        assert!(!is_image_ext("pdf"));
    }

    #[test]
    fn test_is_text_ext() {
        assert!(is_text_ext("md"));
        assert!(is_text_ext("rs"));
        assert!(is_text_ext("json"));
        assert!(!is_text_ext("pdf"));
        assert!(!is_text_ext("docx"));
    }

    #[test]
    fn test_split_processed_empty() {
        let (images, text) = split_processed(&[], 32768);
        assert!(images.is_empty());
        assert!(text.is_empty());
    }

    #[test]
    fn test_split_processed_mixed() {
        let atts = vec![
            ProcessedAttachment {
                filename: "photo.png".into(),
                content_text: None,
                image_data_url: Some("data:image/png;base64,abc".into()),
                saved_path: None,
                original_size: 0,
            },
            ProcessedAttachment {
                filename: "notes.md".into(),
                content_text: Some("# Hello".into()),
                image_data_url: None,
                saved_path: None,
                original_size: 7,
            },
        ];
        let (images, text) = split_processed(&atts, 32768);
        assert_eq!(images.len(), 1);
        assert!(text.contains("notes.md"));
        assert!(text.contains("# Hello"));
    }

    #[test]
    fn test_extract_text_from_bytes_markdown() {
        let bytes = b"# Title\n\nSome content here.";
        let result = extract_text_from_bytes(bytes, "test.md", "md").unwrap();
        assert!(result.contains("# Title"));
        assert!(result.contains("Some content here."));
    }

    #[test]
    fn test_budget_small_file_passes_through() {
        let budget = inline_char_budget(32768);
        let text = "Small file content";
        let result = budget_text_content(text, "small.md", budget);
        assert_eq!(result, text);
    }

    #[test]
    fn test_budget_large_file_truncated() {
        let budget = inline_char_budget(32768);
        let line = "This is a line of content for testing.\n";
        let large = line.repeat(budget / line.len() + 100);
        assert!(large.len() > budget);

        let result = budget_text_content(&large, "huge.md", budget);
        assert!(result.len() < large.len());
        assert!(result.contains("[BOOKMARK"));
        assert!(result.contains("start_line="));
    }

    #[test]
    fn test_budget_preserves_line_boundaries() {
        let budget = inline_char_budget(32768);
        let line = "Line of text here\n";
        let large = line.repeat(budget / line.len() + 100);
        let result = budget_text_content(&large, "test.md", budget);
        // The truncation should end at a newline, not mid-line
        let before_bookmark = result.split("[BOOKMARK").next().unwrap().trim_end();
        assert!(before_bookmark.ends_with("Line of text here"));
    }

    #[test]
    fn test_inline_char_budget_scales_with_context() {
        // A 32K context model should get a different budget than a 128K model
        let small = inline_char_budget(8192);
        let large = inline_char_budget(131072);
        assert!(large > small);
        assert!(small > 0);
    }

    #[test]
    fn test_admin_attachment_inlines_content() {
        // Phase 1 fix: admin attachments must inline actual content, not just save instructions
        let content = b"# Chapter 1\n\nDan walked into the room.";
        let result = process_text_attachment(
            "book.md".into(), "md", content,
            Some("data/uploads/20260429_book.md".into()),
        ).unwrap();

        let text = result.content_text.unwrap();
        // Must contain the actual file content
        assert!(text.contains("Dan walked into the room"), "Admin attachment must inline content");
        // Must also contain the save path reference
        assert!(text.contains("FILE SAVED"), "Must reference saved path");
        assert!(text.contains("data/uploads/"), "Must include saved path");
        // Must NOT contain the old reading protocol instruction
        assert!(!text.contains("READING PROTOCOL"), "Must not have old redirect instruction");
        // Must have saved_path set
        assert!(result.saved_path.is_some());
        assert_eq!(result.original_size, 38);
    }

    #[test]
    fn test_exceeds_budget_threshold() {
        let small = ProcessedAttachment {
            filename: "small.md".into(),
            content_text: Some("hello".into()),
            image_data_url: None,
            saved_path: None,
            original_size: 100,
        };
        assert!(!small.exceeds_budget(32768)); // 100 < budget

        let large = ProcessedAttachment {
            filename: "big.md".into(),
            content_text: None,
            image_data_url: None,
            saved_path: Some("data/uploads/big.md".into()),
            original_size: 2_000_000,
        };
        assert!(large.exceeds_budget(262144)); // 2MB > budget
        assert!(large.exceeds_budget(32768));  // 2MB > smaller budget too
    }
}
