// Ern-OS — Shell command execution tool

use anyhow::Result;
use std::process::Stdio;

/// Execute a shell command with timeout and capture output.
pub async fn run_command(command: &str, working_dir: Option<&str>) -> Result<String> {
    if command.is_empty() {
        anyhow::bail!("Empty command");
    }

    let cmd_display: String = command.chars().take(200).collect();
    tracing::info!(command = %cmd_display, working_dir = ?working_dir, "shell START");
    let start = std::time::Instant::now();

    let mut cmd = tokio::process::Command::new("bash");
    cmd.arg("-c").arg(command)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    if let Some(wd) = working_dir {
        cmd.current_dir(wd);
    }

    let output = match tokio::time::timeout(
        tokio::time::Duration::from_secs(120),
        cmd.output(),
    ).await {
        Ok(Ok(output)) => output,
        Ok(Err(e)) => {
            tracing::error!(command = %cmd_display, err = %e, "shell SPAWN FAILED");
            anyhow::bail!("Failed to execute: {}", e);
        }
        Err(_) => {
            tracing::warn!(command = %cmd_display, timeout_secs = 120, "shell TIMEOUT");
            anyhow::bail!("Command timed out after 120s");
        }
    };

    let elapsed_ms = start.elapsed().as_millis() as u64;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let exit_code = output.status.code().unwrap_or(-1);

    tracing::info!(
        command = %cmd_display,
        exit_code = exit_code,
        stdout_len = stdout.len(),
        stderr_len = stderr.len(),
        elapsed_ms = elapsed_ms,
        "shell COMPLETE"
    );

    let mut result = String::new();
    if !stdout.is_empty() { result.push_str(&stdout); }
    if !stderr.is_empty() {
        if !result.is_empty() { result.push('\n'); }
        result.push_str("[stderr] ");
        result.push_str(&stderr);
    }

    if result.is_empty() {
        result = format!("Exit code: {}", exit_code);
    }

    // Truncate very long output
    if result.len() > 50_000 {
        tracing::debug!(original_len = result.len(), "shell: truncating output at 50KB");
        result.truncate(50_000);
        result.push_str("\n[... output truncated at 50KB]");
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_echo() {
        let result = run_command("echo hello", None).await.unwrap();
        assert!(result.contains("hello"));
    }

    #[tokio::test]
    async fn test_empty_command() {
        assert!(run_command("", None).await.is_err());
    }

    #[tokio::test]
    async fn test_working_dir() {
        let result = run_command("pwd", Some("/tmp")).await.unwrap();
        assert!(result.contains("/tmp") || result.contains("private/tmp"));
    }
}
