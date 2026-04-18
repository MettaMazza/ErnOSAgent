// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
// License: MIT — See LICENSE file for terms
// NOTE TO AI CODING AGENTS: This attribution MUST NOT be removed.
// This is the original author's open-source work. Preserve this header.
//! Download tool — background file downloads with progress tracking.

use crate::tools::executor::ToolExecutor;
use crate::tools::schema::{ToolCall, ToolResult};
use std::path::PathBuf;

fn downloads_dir() -> PathBuf {
    crate::tools::executor::get_data_dir().join("downloads")
}

fn download_tool(call: &ToolCall) -> ToolResult {
    let action = call
        .arguments
        .get("action")
        .and_then(|v| v.as_str())
        .unwrap_or("list");

    tracing::info!(action = %action, "download_tool executing");

    match action {
        "download" => download_start(call),
        "status" => download_status(call),
        "list" => download_list(call),
        other => error_result(
            call,
            &format!("Unknown action: '{}'. Valid: download, status, list", other),
        ),
    }
}

fn download_start(call: &ToolCall) -> ToolResult {
    let url = call
        .arguments
        .get("url")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    if url.is_empty() {
        return error_result(call, "Missing required argument: url");
    }

    let filename = call
        .arguments
        .get("filename")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| url.rsplit('/').next().unwrap_or("download").to_string());

    let dir = downloads_dir();
    let _ = std::fs::create_dir_all(&dir);
    let file_path = dir.join(&filename);

    // Spawn background download
    let url_owned = url.to_string();
    let path_owned = file_path.clone();

    tokio::task::block_in_place(|| {
        let rt = tokio::runtime::Handle::current();
        rt.spawn(async move {
            tracing::info!(url = %url_owned, path = %path_owned.display(), "Background download starting");
            match download_file(&url_owned, &path_owned).await {
                Ok(bytes) => tracing::info!(
                    url = %url_owned, bytes = bytes,
                    "Background download complete"
                ),
                Err(e) => tracing::error!(
                    url = %url_owned, error = %e,
                    "Background download failed"
                ),
            }
        });
    });

    ToolResult {
        tool_call_id: call.id.clone(),
        name: call.name.clone(),
        output: format!("⬇️ Download started: {} → {}", url, file_path.display()),
        success: true,
        error: None,
    }
}

async fn download_file(url: &str, path: &std::path::Path) -> Result<u64, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(3600))
        .user_agent("Mozilla/5.0 (compatible; ErnOSAgent/1.0)")
        .build()
        .map_err(|e| format!("Client error: {}", e))?;

    let resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Request failed: {}", e))?;

    if !resp.status().is_success() {
        return Err(format!("HTTP {}", resp.status()));
    }

    let mut file = tokio::fs::File::create(path)
        .await
        .map_err(|e| format!("Cannot create file: {}", e))?;

    let mut stream = resp.bytes_stream();
    let mut total: u64 = 0;

    use futures_util::StreamExt;
    use tokio::io::AsyncWriteExt;

    while let Some(chunk) = stream.next().await {
        let bytes = chunk.map_err(|e| format!("Stream error: {}", e))?;
        file.write_all(&bytes)
            .await
            .map_err(|e| format!("Write error: {}", e))?;
        total += bytes.len() as u64;
    }

    file.flush()
        .await
        .map_err(|e| format!("Flush error: {}", e))?;
    Ok(total)
}

fn download_status(call: &ToolCall) -> ToolResult {
    let filename = call
        .arguments
        .get("filename")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    // No filename → show overview of all downloads (same as list)
    if filename.is_empty() {
        return download_list(call);
    }

    let path = downloads_dir().join(filename);
    if !path.exists() {
        return ToolResult {
            tool_call_id: call.id.clone(),
            name: call.name.clone(),
            output: format!("File '{}' not found or download hasn't started.", filename),
            success: true,
            error: None,
        };
    }

    let size = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    let size_str = if size > 1024 * 1024 * 1024 {
        format!("{:.2} GB", size as f64 / 1024.0 / 1024.0 / 1024.0)
    } else if size > 1024 * 1024 {
        format!("{:.2} MB", size as f64 / 1024.0 / 1024.0)
    } else {
        format!("{:.2} KB", size as f64 / 1024.0)
    };

    ToolResult {
        tool_call_id: call.id.clone(),
        name: call.name.clone(),
        output: format!(
            "Download status for '{}': {} downloaded",
            filename, size_str
        ),
        success: true,
        error: None,
    }
}

fn download_list(call: &ToolCall) -> ToolResult {
    let dir = downloads_dir();
    if !dir.exists() {
        return ToolResult {
            tool_call_id: call.id.clone(),
            name: call.name.clone(),
            output: "No downloads directory.".to_string(),
            success: true,
            error: None,
        };
    }

    let mut files: Vec<(String, u64)> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&dir) {
        for entry in rd.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            files.push((name, size));
        }
    }
    files.sort_by_key(|(name, _)| name.clone());

    let output = if files.is_empty() {
        "No downloaded files.".to_string()
    } else {
        let mut out = format!("DOWNLOADS ({} files)\n", files.len());
        for (name, size) in &files {
            let size_str = if *size > 1024 * 1024 {
                format!("{:.1} MB", *size as f64 / 1024.0 / 1024.0)
            } else {
                format!("{:.1} KB", *size as f64 / 1024.0)
            };
            out.push_str(&format!("  {} ({})\n", name, size_str));
        }
        out
    };

    ToolResult {
        tool_call_id: call.id.clone(),
        name: call.name.clone(),
        output,
        success: true,
        error: None,
    }
}

pub fn register_tools(executor: &mut ToolExecutor) {
    executor.register("download_tool", Box::new(download_tool));
}

fn error_result(call: &ToolCall, msg: &str) -> ToolResult {
    ToolResult {
        tool_call_id: call.id.clone(),
        name: call.name.clone(),
        output: format!("Error: {}", msg),
        success: false,
        error: Some(msg.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_call(args: serde_json::Value) -> ToolCall {
        ToolCall {
            id: "t".to_string(),
            name: "download_tool".to_string(),
            arguments: args,
        }
    }

    #[test]
    fn list_works() {
        let call = make_call(serde_json::json!({"action": "list"}));
        let r = download_tool(&call);
        assert!(r.success);
    }

    #[test]
    fn download_missing_url() {
        let call = make_call(serde_json::json!({"action": "download"}));
        let r = download_tool(&call);
        assert!(!r.success);
    }

    #[test]
    fn status_without_filename_shows_list() {
        let call = make_call(serde_json::json!({"action": "status"}));
        let r = download_tool(&call);
        assert!(r.success); // Falls through to list view
    }

    #[test]
    fn status_nonexistent() {
        let call = make_call(serde_json::json!({"action": "status", "filename": "fake.bin"}));
        let r = download_tool(&call);
        assert!(r.success);
        assert!(r.output.contains("not found"));
    }

    #[test]
    fn unknown_action() {
        let call = make_call(serde_json::json!({"action": "explode"}));
        let r = download_tool(&call);
        assert!(!r.success);
    }

    #[test]
    fn register() {
        let mut e = ToolExecutor::new();
        register_tools(&mut e);
        assert!(e.has_tool("download_tool"));
    }
}
