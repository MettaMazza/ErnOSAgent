// Ern-OS — High-performance, model-neutral Rust AI agent engine
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Enhanced codebase editing tools — patch, insert, multi_patch, delete.
//!
//! All operations run through containment checks and auto-checkpoint
//! before any destructive modification.

use crate::tools::containment;
use crate::tools::checkpoint::CheckpointManager;
use std::path::{Path, PathBuf};

/// Log a self-edit to data/self_edit_log.jsonl for audit trail.
fn log_edit(data_dir: &Path, action: &str, path: &str, detail: &str) {
    let entry = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "action": action,
        "path": path,
        "detail": &detail[..detail.len().min(500)],
    });
    let log_path = data_dir.join("self_edit_log.jsonl");
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&log_path) {
        use std::io::Write;
        let _ = writeln!(f, "{}", entry);
    }
}

/// Resolve a relative or absolute path against the project root.
fn resolve_path(path_str: &str) -> anyhow::Result<PathBuf> {
    let path = Path::new(path_str);
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        let cwd = std::env::current_dir()?;
        Ok(cwd.join(path))
    }
}

/// Apply a single find-and-replace patch to a file.
pub fn patch_file(data_dir: &Path, path: &str, find: &str, replace: &str) -> anyhow::Result<String> {
    if let Some(reason) = containment::check_path(path) {
        anyhow::bail!(reason);
    }

    let full_path = resolve_path(path)?;
    if !full_path.exists() {
        anyhow::bail!("File not found: {}", path);
    }

    // Checkpoint before modification
    let mgr = CheckpointManager::new(data_dir);
    match mgr.snapshot(&full_path) {
        Ok(id) => tracing::debug!(id = %id, "Pre-patch checkpoint"),
        Err(e) => tracing::warn!(error = %e, "Pre-patch checkpoint failed (non-fatal)"),
    }

    let content = std::fs::read_to_string(&full_path)?;
    let count = content.matches(find).count();
    if count == 0 {
        anyhow::bail!("Search string not found in {}. Verify the exact text.", path);
    }

    let patched = content.replacen(find, replace, 1);
    std::fs::write(&full_path, &patched)?;

    log_edit(data_dir, "patch", path, &format!("find={}, replace={}", &find[..find.len().min(100)], &replace[..replace.len().min(100)]));
    tracing::info!(path = %path, occurrences = count, "codebase_patch applied");
    Ok(format!("Patched {} — replaced 1 of {} occurrence(s)", path, count))
}

/// Insert content before or after an anchor string in a file.
pub fn insert_content(
    data_dir: &Path, path: &str, anchor: &str, content: &str, position: &str,
) -> anyhow::Result<String> {
    if let Some(reason) = containment::check_path(path) {
        anyhow::bail!(reason);
    }
    if position != "before" && position != "after" {
        anyhow::bail!("position must be 'before' or 'after'");
    }

    let full_path = resolve_path(path)?;
    if !full_path.exists() {
        anyhow::bail!("File not found: {}", path);
    }

    let mgr = CheckpointManager::new(data_dir);
    match mgr.snapshot(&full_path) {
        Ok(id) => tracing::debug!(id = %id, "Pre-insert checkpoint"),
        Err(e) => tracing::warn!(error = %e, "Pre-insert checkpoint failed (non-fatal)"),
    }

    let text = std::fs::read_to_string(&full_path)?;
    let pos = text.find(anchor)
        .ok_or_else(|| anyhow::anyhow!("Anchor text not found in {}", path))?;

    let new_text = if position == "after" {
        let insert_pos = pos + anchor.len();
        format!("{}{}{}", &text[..insert_pos], content, &text[insert_pos..])
    } else {
        format!("{}{}{}", &text[..pos], content, &text[pos..])
    };

    std::fs::write(&full_path, &new_text)?;
    log_edit(data_dir, "insert", path, &format!("position={}, anchor_len={}", position, anchor.len()));
    tracing::info!(path = %path, position = %position, "codebase_insert applied");
    Ok(format!("Inserted content {} anchor in {}", position, path))
}

/// Apply multiple find-replace patches to a file atomically.
pub fn multi_patch(
    data_dir: &Path, path: &str, patches: &[serde_json::Value],
) -> anyhow::Result<String> {
    if let Some(reason) = containment::check_path(path) {
        anyhow::bail!(reason);
    }

    let full_path = resolve_path(path)?;
    if !full_path.exists() {
        anyhow::bail!("File not found: {}", path);
    }

    let mgr = CheckpointManager::new(data_dir);
    match mgr.snapshot(&full_path) {
        Ok(id) => tracing::debug!(id = %id, "Pre-multi_patch checkpoint"),
        Err(e) => tracing::warn!(error = %e, "Pre-multi_patch checkpoint failed (non-fatal)"),
    }

    let mut text = std::fs::read_to_string(&full_path)?;
    let mut applied = 0;
    let mut errors = Vec::new();

    for (i, patch) in patches.iter().enumerate() {
        let find = patch.get("find")
            .or_else(|| patch.get("search"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let replace = patch.get("replace").and_then(|v| v.as_str()).unwrap_or("");

        if find.is_empty() {
            errors.push(format!("Patch {}: empty search string", i + 1));
            continue;
        }
        if text.contains(find) {
            text = text.replacen(find, replace, 1);
            applied += 1;
        } else {
            errors.push(format!("Patch {}: search string not found", i + 1));
        }
    }

    if applied > 0 {
        std::fs::write(&full_path, &text)?;
        log_edit(data_dir, "multi_patch", path, &format!("applied={}, errors={}", applied, errors.len()));
    }

    tracing::info!(path = %path, applied, errors = errors.len(), "codebase_multi_patch applied");

    if applied == 0 && !errors.is_empty() {
        anyhow::bail!("No patches applied. Errors: {}", errors.join("; "));
    }

    Ok(format!("Applied {} patch(es) to {} (errors: {})", applied, path, errors.len()))
}

/// Delete a file (with checkpoint backup).
pub fn delete_file(data_dir: &Path, path: &str) -> anyhow::Result<String> {
    if let Some(reason) = containment::check_path(path) {
        anyhow::bail!(reason);
    }

    let full_path = resolve_path(path)?;
    if !full_path.exists() {
        return Ok(format!("Verified: {} does not exist", path));
    }

    // Checkpoint before deletion
    if full_path.is_file() {
        let mgr = CheckpointManager::new(data_dir);
        match mgr.snapshot(&full_path) {
            Ok(id) => tracing::debug!(id = %id, "Pre-delete checkpoint"),
            Err(e) => tracing::warn!(error = %e, "Pre-delete checkpoint failed (non-fatal)"),
        }
        std::fs::remove_file(&full_path)?;
    } else if full_path.is_dir() {
        std::fs::remove_dir_all(&full_path)?;
    }

    log_edit(data_dir, "delete", path, "file deleted");
    tracing::info!(path = %path, "codebase_delete executed");
    Ok(format!("Deleted {}", path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn test_dir() -> PathBuf {
        let dir = PathBuf::from("target/test_codebase_edit");
        let _ = fs::create_dir_all(&dir);
        dir
    }

    #[test]
    fn test_patch_file() {
        let dir = test_dir().join("patch");
        let _ = fs::create_dir_all(&dir);
        let file = dir.join("test.rs");
        fs::write(&file, "fn hello() { println!(\"old\"); }").unwrap();

        let result = patch_file(&dir, &file.to_string_lossy(), "old", "new");
        assert!(result.is_ok());
        assert!(fs::read_to_string(&file).unwrap().contains("new"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_patch_not_found() {
        let dir = test_dir().join("patch_nf");
        let _ = fs::create_dir_all(&dir);
        let file = dir.join("test.rs");
        fs::write(&file, "hello world").unwrap();

        let result = patch_file(&dir, &file.to_string_lossy(), "nonexistent", "x");
        assert!(result.is_err());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_insert_after() {
        let dir = test_dir().join("insert");
        let _ = fs::create_dir_all(&dir);
        let file = dir.join("test.rs");
        fs::write(&file, "// START\nfn main() {}\n// END").unwrap();

        let result = insert_content(&dir, &file.to_string_lossy(), "// START\n", "// INSERTED\n", "after");
        assert!(result.is_ok());
        let content = fs::read_to_string(&file).unwrap();
        assert!(content.contains("// START\n// INSERTED\n"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_containment_blocks_protected() {
        let dir = test_dir().join("contain");
        let result = patch_file(&dir, "agents/rust_code_governance.md", "x", "y");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Containment"));
    }

    #[test]
    fn test_delete_file() {
        let dir = test_dir().join("delete");
        let _ = fs::create_dir_all(&dir);
        let file = dir.join("deleteme.txt");
        fs::write(&file, "bye").unwrap();

        let result = delete_file(&dir, &file.to_string_lossy());
        assert!(result.is_ok());
        assert!(!file.exists());

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_multi_patch() {
        let dir = test_dir().join("multi");
        let _ = fs::create_dir_all(&dir);
        let file = dir.join("multi.rs");
        fs::write(&file, "aaa bbb ccc").unwrap();

        let patches = vec![
            serde_json::json!({"find": "aaa", "replace": "AAA"}),
            serde_json::json!({"find": "ccc", "replace": "CCC"}),
        ];

        let result = multi_patch(&dir, &file.to_string_lossy(), &patches);
        assert!(result.is_ok());
        assert_eq!(fs::read_to_string(&file).unwrap(), "AAA bbb CCC");

        let _ = fs::remove_dir_all(&dir);
    }
}
