//! Compile & test verification — runs build and test commands, parses results.

use anyhow::{Context, Result};
use std::path::Path;

/// Result of a compilation or test run.
#[derive(Debug, Clone)]
pub struct CompileResult {
    pub success: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub test_summary: Option<TestSummary>,
    pub raw_output: String,
}

/// Parsed test results.
#[derive(Debug, Clone)]
pub struct TestSummary {
    pub passed: usize,
    pub failed: usize,
    pub ignored: usize,
    pub failures: Vec<String>,
}

/// Run `cargo build --release --features metal` and parse output.
pub async fn check_build(project_root: &Path) -> Result<CompileResult> {
    let output = tokio::process::Command::new("cargo")
        .args(["build", "--release", "--features", "metal"])
        .current_dir(project_root)
        .output()
        .await
        .context("Failed to execute cargo build")?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let raw = format!("{}\n{}", stdout, stderr);

    let errors = parse_cargo_errors(&stderr);
    let warnings = parse_cargo_warnings(&stderr);

    tracing::info!(
        success = output.status.success(),
        errors = errors.len(),
        warnings = warnings.len(),
        "Build check complete"
    );

    Ok(CompileResult {
        success: output.status.success(),
        errors,
        warnings,
        test_summary: None,
        raw_output: raw,
    })
}

/// Run `cargo test --features metal` and parse output.
pub async fn check_tests(project_root: &Path) -> Result<CompileResult> {
    let output = tokio::process::Command::new("cargo")
        .args(["test", "--features", "metal"])
        .current_dir(project_root)
        .output()
        .await
        .context("Failed to execute cargo test")?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let raw = format!("{}\n{}", stdout, stderr);

    let errors = parse_cargo_errors(&stderr);
    let warnings = parse_cargo_warnings(&stderr);
    let test_summary = parse_test_summary(&stdout);

    tracing::info!(
        success = output.status.success(),
        passed = test_summary.as_ref().map(|t| t.passed).unwrap_or(0),
        failed = test_summary.as_ref().map(|t| t.failed).unwrap_or(0),
        "Test check complete"
    );

    Ok(CompileResult {
        success: output.status.success(),
        errors,
        warnings,
        test_summary,
        raw_output: raw,
    })
}

/// Run an arbitrary shell command and check its exit code.
pub async fn check_command(cmd: &str, cwd: &Path) -> Result<CompileResult> {
    let output = tokio::process::Command::new("sh")
        .args(["-c", cmd])
        .current_dir(cwd)
        .output()
        .await
        .with_context(|| format!("Failed to execute: {}", cmd))?;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let raw = format!("{}\n{}", stdout, stderr);

    Ok(CompileResult {
        success: output.status.success(),
        errors: if output.status.success() { vec![] } else { vec![stderr] },
        warnings: vec![],
        test_summary: None,
        raw_output: raw,
    })
}

/// Parse `error[E...]` lines from cargo stderr.
fn parse_cargo_errors(stderr: &str) -> Vec<String> {
    stderr.lines()
        .filter(|l| l.starts_with("error"))
        .map(|l| l.to_string())
        .collect()
}

/// Parse `warning:` lines from cargo stderr (own crate only).
fn parse_cargo_warnings(stderr: &str) -> Vec<String> {
    stderr.lines()
        .filter(|l| l.contains("warning:") && l.contains("ern-os"))
        .map(|l| l.trim().to_string())
        .collect()
}

/// Parse `test result: ok. N passed; N failed; N ignored` from test output.
fn parse_test_summary(stdout: &str) -> Option<TestSummary> {
    let mut total_passed = 0usize;
    let mut total_failed = 0usize;
    let mut total_ignored = 0usize;
    let mut found = false;

    for line in stdout.lines() {
        if line.starts_with("test result:") {
            found = true;
            total_passed += extract_count(line, "passed");
            total_failed += extract_count(line, "failed");
            total_ignored += extract_count(line, "ignored");
        }
    }

    if !found { return None; }

    let failures = parse_test_failures(stdout);

    Some(TestSummary {
        passed: total_passed,
        failed: total_failed,
        ignored: total_ignored,
        failures,
    })
}

/// Extract a count number before a label (e.g., "42 passed").
fn extract_count(line: &str, label: &str) -> usize {
    line.split_whitespace()
        .zip(line.split_whitespace().skip(1))
        .find(|(_, next)| next.trim_end_matches(';') == label)
        .and_then(|(num, _)| num.parse().ok())
        .unwrap_or(0)
}

/// Parse failed test names from cargo test output.
fn parse_test_failures(stdout: &str) -> Vec<String> {
    let mut failures = Vec::new();
    let mut in_failures = false;

    for line in stdout.lines() {
        if line.contains("failures:") && !line.contains("---") {
            in_failures = true;
            continue;
        }
        if in_failures {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with("---") {
                in_failures = false;
                continue;
            }
            failures.push(trimmed.to_string());
        }
    }
    failures
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_cargo_errors() {
        let stderr = "error[E0308]: mismatched types\n  --> src/main.rs:10:5\nwarning: unused variable";
        let errors = parse_cargo_errors(stderr);
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("E0308"));
    }

    #[test]
    fn test_parse_cargo_warnings() {
        let stderr = "warning: `ern-os` (lib) generated 2 warnings\nwarning: other crate";
        let warnings = parse_cargo_warnings(stderr);
        assert_eq!(warnings.len(), 1);
    }

    #[test]
    fn test_parse_test_summary() {
        let stdout = "running 5 tests\ntest foo ... ok\ntest result: ok. 4 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out";
        let summary = parse_test_summary(stdout).unwrap();
        assert_eq!(summary.passed, 4);
        assert_eq!(summary.failed, 1);
        assert_eq!(summary.ignored, 0);
    }

    #[test]
    fn test_parse_test_summary_multiple_suites() {
        let stdout = "test result: ok. 100 passed; 0 failed; 0 ignored;\ntest result: ok. 50 passed; 2 failed; 1 ignored;";
        let summary = parse_test_summary(stdout).unwrap();
        assert_eq!(summary.passed, 150);
        assert_eq!(summary.failed, 2);
        assert_eq!(summary.ignored, 1);
    }

    #[test]
    fn test_extract_count() {
        assert_eq!(extract_count("test result: ok. 42 passed; 0 failed;", "passed"), 42);
        assert_eq!(extract_count("test result: ok. 42 passed; 3 failed;", "failed"), 3);
    }

    #[test]
    fn test_parse_test_failures() {
        let stdout = "failures:\n    tests::test_foo\n    tests::test_bar\n\ntest result:";
        let failures = parse_test_failures(stdout);
        assert_eq!(failures.len(), 2);
        assert!(failures[0].contains("test_foo"));
    }

    #[test]
    fn test_no_test_summary() {
        let summary = parse_test_summary("no test output here");
        assert!(summary.is_none());
    }
}
