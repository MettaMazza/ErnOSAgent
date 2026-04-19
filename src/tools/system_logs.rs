// Ern-OS — High-performance, model-neutral Rust AI agent engine
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! System logs tool — read-only access to engine logs for self-healing.
//!
//! The agent can read its own error/warn output to diagnose and fix
//! issues proactively. Write access is NOT provided.

use std::path::Path;

/// Execute a system_logs action.
pub fn execute(args: &serde_json::Value, data_dir: &Path) -> anyhow::Result<String> {
    tracing::info!(tool = "system_logs", "tool START");
    let action = args["action"].as_str().unwrap_or("tail");
    match action {
        "tail" => tail_logs(data_dir, args),
        "errors" => grep_errors(data_dir, args),
        "search" => search_logs(data_dir, args),
        "self_edits" => list_self_edits(data_dir, args),
        other => Ok(format!("Unknown system_logs action: {}", other)),
    }
}

/// Return the last N lines from the upgrade/runtime log.
fn tail_logs(data_dir: &Path, args: &serde_json::Value) -> anyhow::Result<String> {
    let n = args["lines"].as_u64().unwrap_or(50) as usize;
    let log_path = data_dir.join("logs/upgrade.log");

    if !log_path.exists() {
        return Ok("No log file found. The engine may not have generated logs yet.".into());
    }

    let content = std::fs::read_to_string(&log_path)?;
    let lines: Vec<&str> = content.lines().collect();
    let start = lines.len().saturating_sub(n);
    let tail: Vec<&str> = lines[start..].to_vec();

    Ok(format!(
        "Last {} lines (of {}):\n{}",
        tail.len(), lines.len(), tail.join("\n")
    ))
}

/// Grep for ERROR and WARN lines across log files.
fn grep_errors(data_dir: &Path, args: &serde_json::Value) -> anyhow::Result<String> {
    let n = args["max"].as_u64().unwrap_or(30) as usize;
    let mut errors = Vec::new();

    // Check upgrade log
    let log_path = data_dir.join("logs/upgrade.log");
    if log_path.exists() {
        if let Ok(content) = std::fs::read_to_string(&log_path) {
            for line in content.lines() {
                let lower = line.to_lowercase();
                if lower.contains("error") || lower.contains("warn")
                    || lower.contains("panic") || lower.contains("failed")
                {
                    errors.push(line.to_string());
                }
            }
        }
    }

    // Check activity log
    let activity = data_dir.join("activity.jsonl");
    if activity.exists() {
        if let Ok(content) = std::fs::read_to_string(&activity) {
            for line in content.lines() {
                if line.contains("error") || line.contains("ERROR") {
                    errors.push(line.to_string());
                }
            }
        }
    }

    // Trim to max and return latest first
    errors.reverse();
    errors.truncate(n);

    if errors.is_empty() {
        Ok("No errors or warnings found in logs.".into())
    } else {
        Ok(format!("{} error/warning line(s):\n{}", errors.len(), errors.join("\n")))
    }
}

/// Search logs for a specific pattern.
fn search_logs(data_dir: &Path, args: &serde_json::Value) -> anyhow::Result<String> {
    let pattern = args["pattern"].as_str().unwrap_or("");
    if pattern.is_empty() {
        anyhow::bail!("Missing 'pattern' parameter");
    }
    let n = args["max"].as_u64().unwrap_or(20) as usize;
    let lower_pattern = pattern.to_lowercase();
    let mut matches = Vec::new();

    let log_files = [
        data_dir.join("logs/upgrade.log"),
        data_dir.join("activity.jsonl"),
        data_dir.join("recompile_log.md"),
        data_dir.join("self_edit_log.jsonl"),
    ];

    for path in &log_files {
        if !path.exists() { continue; }
        if let Ok(content) = std::fs::read_to_string(path) {
            let fname = path.file_name().unwrap_or_default().to_string_lossy();
            for (i, line) in content.lines().enumerate() {
                if line.to_lowercase().contains(&lower_pattern) {
                    matches.push(format!("[{}:{}] {}", fname, i + 1, line));
                    if matches.len() >= n { break; }
                }
            }
        }
    }

    if matches.is_empty() {
        Ok(format!("No matches for '{}' in logs.", pattern))
    } else {
        Ok(format!("{} match(es) for '{}':\n{}", matches.len(), pattern, matches.join("\n")))
    }
}

/// List recent self-edit audit entries.
fn list_self_edits(data_dir: &Path, args: &serde_json::Value) -> anyhow::Result<String> {
    let n = args["max"].as_u64().unwrap_or(20) as usize;
    let path = data_dir.join("self_edit_log.jsonl");

    if !path.exists() {
        return Ok("No self-edit log found. No codebase edits have been made.".into());
    }

    let content = std::fs::read_to_string(&path)?;
    let lines: Vec<&str> = content.lines().collect();
    let start = lines.len().saturating_sub(n);
    let recent: Vec<&str> = lines[start..].to_vec();

    Ok(format!(
        "{} self-edit(s) (last {} of {}):\n{}",
        recent.len(), n, lines.len(), recent.join("\n")
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn test_dir(name: &str) -> std::path::PathBuf {
        let d = std::path::PathBuf::from(format!("target/test_system_logs_{}", name));
        let _ = fs::create_dir_all(d.join("logs"));
        d
    }

    #[test]
    fn test_tail_empty() {
        let dir = test_dir("tail_empty");
        // Ensure no log file exists
        let _ = fs::remove_file(dir.join("logs/upgrade.log"));
        let args = serde_json::json!({"action": "tail"});
        let result = execute(&args, &dir);
        assert!(result.is_ok());
        assert!(result.unwrap().contains("No log file"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_tail_with_content() {
        let dir = test_dir("tail_content");
        let log = dir.join("logs/upgrade.log");
        fs::write(&log, "line1\nline2\nline3\nline4\nline5\n").unwrap();

        let args = serde_json::json!({"action": "tail", "lines": 3});
        let result = execute(&args, &dir).unwrap();
        assert!(result.contains("line3"));
        assert!(result.contains("line5"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_errors_grep() {
        let dir = test_dir("errors_grep");
        let log = dir.join("logs/upgrade.log");
        fs::write(&log, "INFO: all good\nERROR: something broke\nWARN: caution\nINFO: ok\n").unwrap();

        let args = serde_json::json!({"action": "errors"});
        let result = execute(&args, &dir).unwrap();
        assert!(result.contains("something broke"));
        assert!(result.contains("caution"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_search_pattern() {
        let dir = test_dir("search");
        let log = dir.join("logs/upgrade.log");
        fs::write(&log, "Starting server\nListening on 3000\nConnection established\n").unwrap();

        let args = serde_json::json!({"action": "search", "pattern": "3000"});
        let result = execute(&args, &dir).unwrap();
        assert!(result.contains("Listening on 3000"));

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_self_edits_empty() {
        let dir = test_dir("edits_empty");
        let _ = fs::remove_file(dir.join("self_edit_log.jsonl"));
        let args = serde_json::json!({"action": "self_edits"});
        let result = execute(&args, &dir).unwrap();
        assert!(result.contains("No self-edit log"));
        let _ = fs::remove_dir_all(&dir);
    }
}
