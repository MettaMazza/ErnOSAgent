// Ern-OS — High-performance, model-neutral Rust AI agent engine
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! System recompile tool — safety-gated self-compilation with hot-swap.
//!
//! Pipeline: change gate → test gate → warning gate → build → changelog
//!           → resume state → binary stage → activity log → hot-swap.

use std::path::PathBuf;

/// Filter warnings that belong to our own code (not dependencies).
fn is_own_warning(line: &str) -> bool {
    (line.starts_with("warning:") || line.contains("warning["))
        && !line.contains("generated")
        && !line.contains("future-incompat")
}

/// Extract warning context lines from compiler stderr.
fn extract_warning_context(stderr: &str) -> String {
    let lines: Vec<&str> = stderr.lines().collect();
    let mut context = String::new();
    for (i, line) in lines.iter().enumerate() {
        if is_own_warning(line) {
            context.push_str(line);
            context.push('\n');
            for j in 1..=4 {
                if i + j < lines.len() {
                    context.push_str(lines[i + j]);
                    context.push('\n');
                }
            }
            context.push('\n');
        }
    }
    context
}

/// Run the full self-recompile pipeline.
pub async fn run_recompile() -> Result<String, String> {
    let project_root = std::env::current_dir()
        .map_err(|e| format!("Failed to get project root: {}", e))?;

    tracing::info!(cwd = %project_root.display(), "system_recompile: starting");

    // Stage 0: Change gate — block no-op recompiles
    check_for_changes(&project_root).await?;

    // Stage 1-2: Test + warning gate
    run_test_gate(&project_root).await?;

    // Stage 3: Build
    run_build_gate(&project_root).await?;

    // Stage 4-7: Changelog, resume, binary stage, activity log
    write_changelog(&project_root).await;
    write_resume_state(&project_root);
    let staged = stage_binary(&project_root)?;
    write_activity_log(&project_root);

    // Stage 8: Hot-swap
    try_hot_swap(&project_root, &staged).await
}

/// Stage 0 — Verify source files actually changed before allowing recompile.
async fn check_for_changes(root: &PathBuf) -> Result<(), String> {
    tracing::info!("system_recompile: STAGE 0 — change gate");

    let diff = tokio::process::Command::new("git")
        .args(["diff", "--stat", "HEAD"])
        .current_dir(root)
        .output()
        .await
        .map_err(|e| format!("Failed to run git diff: {}", e))?;

    let diff_text = String::from_utf8_lossy(&diff.stdout);
    let has_uncommitted = !diff_text.trim().is_empty();

    if !has_uncommitted {
        // Also check for untracked .rs files
        let status = tokio::process::Command::new("git")
            .args(["status", "--porcelain", "--", "*.rs", "src/"])
            .current_dir(root)
            .output()
            .await
            .map_err(|e| format!("Failed to run git status: {}", e))?;
        let status_text = String::from_utf8_lossy(&status.stdout);
        if status_text.trim().is_empty() {
            return Err(
                "RECOMPILE BLOCKED: No source files changed. \
                Recompilation without code changes is not allowed. \
                Make changes with codebase_edit first, then recompile."
                    .to_string(),
            );
        }
    }

    tracing::info!("system_recompile: changes detected, proceeding");
    Ok(())
}

/// Stages 1-2 — Run tests and check for warnings.
async fn run_test_gate(root: &PathBuf) -> Result<(), String> {
    tracing::info!("system_recompile: STAGE 1 — test suite");

    let test_out = run_cargo(root, &["test", "--release", "--lib"], 600).await?;
    let stderr = String::from_utf8_lossy(&test_out.stderr);

    if !test_out.status.success() {
        let stdout = String::from_utf8_lossy(&test_out.stdout);
        return Err(format!(
            "RECOMPILE BLOCKED: Tests failed.\n\n\
            Fix with codebase_edit, then call system_recompile again.\n\n\
            Output:\n{}\n{}",
            &stdout[..stdout.len().min(3000)],
            &stderr[..stderr.len().min(3000)]
        ));
    }

    let warnings: Vec<&str> = stderr.lines().filter(|l| is_own_warning(l)).collect();
    if !warnings.is_empty() {
        let context = extract_warning_context(&stderr);
        return Err(format!(
            "RECOMPILE BLOCKED: {} warning(s). Fix with codebase_edit and retry.\n\n{}",
            warnings.len(), context
        ));
    }

    tracing::info!("system_recompile: tests passed, zero warnings");
    Ok(())
}

/// Stage 3 — Build release binary.
async fn run_build_gate(root: &PathBuf) -> Result<(), String> {
    tracing::info!("system_recompile: STAGE 3 — cargo build --release");

    let build_out = run_cargo(root, &["build", "--release"], 600).await?;
    let stderr = String::from_utf8_lossy(&build_out.stderr);

    if !build_out.status.success() {
        return Err(format!("Build failed:\n{}", stderr));
    }

    let warns: Vec<&str> = stderr.lines().filter(|l| is_own_warning(l)).collect();
    if !warns.is_empty() {
        return Err(format!(
            "BUILD BLOCKED: {} warning(s). Fix and retry.\n\n{}",
            warns.len(), warns.join("\n")
        ));
    }

    tracing::info!("system_recompile: build successful");
    Ok(())
}

/// Stage 6 — Copy release binary to staging location.
fn stage_binary(root: &PathBuf) -> Result<PathBuf, String> {
    let binary_name = "ern-os";
    let release = root.join(format!("target/release/{}", binary_name));
    let staged = root.join(format!("{}_next", binary_name));

    if !release.exists() {
        return Err(format!("Release binary not found: {}", release.display()));
    }

    std::fs::copy(&release, &staged)
        .map_err(|e| format!("Failed to stage binary: {}", e))?;

    tracing::info!(staged = %staged.display(), "Binary staged for hot-swap");
    Ok(staged)
}

/// Stage 8 — Spawn upgrade.sh for hot-swap, or report manual restart needed.
async fn try_hot_swap(root: &PathBuf, staged: &PathBuf) -> Result<String, String> {
    let upgrade_script = root.join("scripts/upgrade.sh");
    if upgrade_script.exists() {
        tracing::warn!("system_recompile: spawning upgrade.sh, exiting in 5s");

        let _ = tokio::process::Command::new("bash")
            .arg(&upgrade_script)
            .current_dir(root)
            .spawn();

        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        std::process::exit(0);
    }

    Ok(format!(
        "✅ Compilation successful! Binary staged at {}.\n\
        No scripts/upgrade.sh — hot-swap skipped. Restart manually.",
        staged.display()
    ))
}

/// Run a cargo command with timeout.
async fn run_cargo(
    root: &PathBuf, args: &[&str], timeout_secs: u64,
) -> Result<std::process::Output, String> {
    tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        tokio::process::Command::new("cargo")
            .args(args)
            .current_dir(root)
            .output(),
    )
    .await
    .map_err(|_| format!("cargo {} timed out after {}s", args.join(" "), timeout_secs))?
    .map_err(|e| format!("Failed to invoke cargo: {}", e))
}

/// Write changelog from git diff/log.
async fn write_changelog(root: &PathBuf) {
    let diff = tokio::process::Command::new("git")
        .args(["diff", "--stat", "HEAD"])
        .current_dir(root)
        .output().await;
    let log = tokio::process::Command::new("git")
        .args(["log", "--oneline", "-5"])
        .current_dir(root)
        .output().await;

    let diff_text = diff.map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();
    let log_text = log.map(|o| String::from_utf8_lossy(&o.stdout).to_string()).unwrap_or_default();

    let entry = format!(
        "\n## Recompile — {}\n\n**Changes:**\n{}\n\n**Recent commits:**\n{}\n\n---\n",
        chrono::Utc::now().format("%Y-%m-%d %H:%M UTC"),
        if diff_text.trim().is_empty() { "None" } else { diff_text.trim() },
        log_text.trim(),
    );

    let log_path = root.join("data/recompile_log.md");
    let existing = std::fs::read_to_string(&log_path)
        .unwrap_or_else(|_| "# Self-Recompilation Log\n".to_string());
    let _ = std::fs::write(&log_path, format!("{}{}", existing, entry));
    tracing::info!("Changelog written");
}

/// Save resume state for post-restart context.
fn write_resume_state(root: &PathBuf) {
    let resume = serde_json::json!({
        "message": "System recompiled successfully. Resuming operations.",
        "compiled_at": chrono::Utc::now().to_rfc3339(),
    });
    let _ = std::fs::write(
        root.join("data/resume.json"),
        serde_json::to_string_pretty(&resume).unwrap_or_default(),
    );
    tracing::info!("Resume state saved");
}

/// Append to activity log.
fn write_activity_log(root: &PathBuf) {
    let entry = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "tool": "system_recompile",
        "summary": "[SELF-RECOMPILE] Binary staged. Tests passed, zero warnings."
    });
    let path = root.join("data/activity.jsonl");
    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        use std::io::Write;
        let _ = writeln!(f, "{}", entry);
    }
    tracing::info!("Activity logged");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_own_warning() {
        assert!(is_own_warning("warning: unused variable `x`"));
        assert!(is_own_warning("some context warning[E0001]"));
        assert!(!is_own_warning("warning: 1 warning generated"));
        assert!(!is_own_warning("note: this is just a note"));
    }

    #[test]
    fn test_extract_warning_context() {
        let stderr = "Compiling ern-os v0.1.0\n\
            warning: unused variable `x`\n\
            --> src/main.rs:10:9\n\
            |\n\
            10 |     let x = 5;\n\
            |         ^ help: prefix with `_`\n\
            \n\
            warning: 1 warning generated\n";

        let context = extract_warning_context(stderr);
        assert!(context.contains("unused variable"));
        assert!(context.contains("src/main.rs:10:9"));
        assert!(!context.contains("1 warning generated"));
    }

    #[test]
    fn test_extract_warning_context_clean() {
        let stderr = "Compiling ern-os v0.1.0\nFinished in 2.5s\n";
        let context = extract_warning_context(stderr);
        assert!(context.is_empty());
    }
}
