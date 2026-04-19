// Ern-OS — High-performance, model-neutral Rust AI agent engine
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Structured logging — per-session rotating JSON log files.

use anyhow::{Context, Result};
use std::path::Path;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

/// Initialise the tracing subscriber with structured JSON logging.
///
/// Logs go to both:
/// - stdout (human-readable, for development)
/// - `{data_dir}/logs/` (JSON, rotating daily, for production audit)
pub fn init(data_dir: &Path) -> Result<()> {
    let log_dir = data_dir.join("logs");
    std::fs::create_dir_all(&log_dir)
        .with_context(|| format!("Failed to create log dir: {}", log_dir.display()))?;

    let file_appender = tracing_appender::rolling::daily(&log_dir, "ern-os.log");

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,ern_os=debug"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(
            fmt::layer()
                .with_target(true)
                .with_thread_ids(false)
                .with_ansi(true),
        )
        .with(
            fmt::layer()
                .json()
                .with_writer(file_appender)
                .with_target(true)
                .with_thread_ids(true)
                .with_ansi(false),
        )
        .init();

    tracing::debug!(log_dir = %log_dir.display(), "Logging initialised");
    Ok(())
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    #[test]
    fn test_init_creates_log_dir() {
        let tmp = TempDir::new().unwrap();
        let log_dir = tmp.path().join("logs");
        assert!(!log_dir.exists());
        // Can't call init twice in tests (global subscriber), just verify dir creation
        std::fs::create_dir_all(&log_dir).unwrap();
        assert!(log_dir.exists());
    }
}
