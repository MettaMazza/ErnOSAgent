// Ern-OS — Service startup helpers
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Startup routines for optional services (Kokoro TTS, Flux, code-server)
//! and platform-neutral browser open. Extracted from main.rs for §1.1 compliance.

use anyhow::Result;

/// Auto-start Kokoro TTS server if not already running.
pub async fn maybe_start_kokoro(config: &crate::config::AppConfig) {
    let port = config.general.kokoro_port.unwrap_or(8880);
    let url = format!("http://127.0.0.1:{}/v1/models", port);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap_or_default();
    if client.get(&url).send().await.map_or(false, |r| r.status().is_success()) {
        tracing::info!(port, "Kokoro TTS already running");
        return;
    }

    let script_path = match find_kokoro_script() {
        Some(p) => p,
        None => {
            tracing::debug!("Kokoro TTS script not found — TTS disabled");
            return;
        }
    };

    tracing::info!(script = %script_path.display(), port, "Starting Kokoro TTS server");

    let python = find_python312();
    match tokio::process::Command::new(&python)
        .arg(&script_path)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(child) => {
            tracing::info!(pid = child.id().unwrap_or(0), python = %python, "Kokoro TTS server spawned");
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to start Kokoro TTS — TTS disabled");
        }
    }
}

/// Auto-start Flux image generation server if not already running.
pub async fn maybe_start_flux(config: &crate::config::AppConfig) {
    let port = config.general.flux_port.unwrap_or(8890);
    let url = format!("http://127.0.0.1:{}/health", port);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap_or_default();
    if client.get(&url).send().await.map_or(false, |r| r.status().is_success()) {
        tracing::info!(port, "Flux image server already running");
        return;
    }

    let script_path = match find_flux_script() {
        Some(p) => p,
        None => {
            tracing::debug!("Flux server script not found — image generation disabled");
            return;
        }
    };

    tracing::info!(script = %script_path.display(), port, "Starting Flux image server");

    let (cmd_bin, cmd_args) = find_flux_launch_command(&script_path);
    tracing::info!(cmd = %cmd_bin, "Flux launch command");

    match tokio::process::Command::new(&cmd_bin)
        .args(&cmd_args)
        .env("FLUX_PORT", port.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(mut child) => {
            let pid = child.id().unwrap_or(0);
            tracing::info!(pid, cmd = %cmd_bin, "Flux image server spawned — waiting for health check");
            wait_for_flux_health(&mut child, &client, &url, pid).await;
        }
        Err(e) => {
            tracing::warn!(error = %e, cmd = %cmd_bin, "Failed to start Flux — image generation disabled");
        }
    }
}

/// Wait up to 60s for Flux server health, checking for early crashes.
async fn wait_for_flux_health(
    child: &mut tokio::process::Child,
    client: &reqwest::Client,
    url: &str,
    pid: u32,
) {
    for i in 0..60 {
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        if let Ok(Some(status)) = child.try_wait() {
            let stderr = if let Some(mut err) = child.stderr.take() {
                let mut buf = String::new();
                let _ = tokio::io::AsyncReadExt::read_to_string(&mut err, &mut buf).await;
                buf
            } else {
                String::new()
            };
            tracing::error!(
                pid, exit = %status, stderr = %stderr.trim(),
                "Flux server crashed during startup"
            );
            return;
        }

        if client.get(url).send().await.map_or(false, |r| r.status().is_success()) {
            tracing::info!(pid, seconds = i + 1, "Flux image server ready");
            return;
        }
    }
    tracing::warn!(pid, "Flux server spawned but not healthy after 60s — may still be loading");
}

/// Auto-start code-server (VS Code IDE) if enabled and not already running.
pub async fn maybe_start_code_server(config: &crate::config::AppConfig) {
    if !config.codes.enabled { return; }
    let port = config.codes.port;
    let url = format!("http://127.0.0.1:{}/healthz", port);

    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap_or_default();
    if client.get(&url).send().await.is_ok() {
        tracing::info!(port, "code-server already running");
        return;
    }

    let binary_path = match find_code_server_binary() {
        Some(p) => p,
        None => {
            tracing::debug!("code-server binary not found — Codes IDE disabled");
            return;
        }
    };

    let workspace = std::path::PathBuf::from(&config.codes.workspace)
        .canonicalize()
        .unwrap_or_else(|_| std::path::PathBuf::from("."));

    tracing::info!(binary = %binary_path, port, workspace = %workspace.display(), "Starting code-server");
    match tokio::process::Command::new(&binary_path)
        .args([
            "--port", &port.to_string(),
            "--auth", "none",
            "--disable-telemetry",
            "--disable-update-check",
            &workspace.to_string_lossy(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(child) => {
            tracing::info!(pid = child.id().unwrap_or(0), "code-server started");
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to start code-server");
        }
    }
}

/// Check for post-recompile resume state. Returns the resume message if found.
pub fn check_recompile_resume(config: &crate::config::AppConfig) -> Option<String> {
    let resume_path = config.general.data_dir.join("resume.json");
    if !resume_path.exists() {
        return None;
    }

    let result = match std::fs::read_to_string(&resume_path) {
        Ok(content) => {
            if let Ok(resume) = serde_json::from_str::<serde_json::Value>(&content) {
                let msg = resume["message"].as_str().unwrap_or("Recompile complete").to_string();
                let at = resume["compiled_at"].as_str().unwrap_or("unknown");
                tracing::info!(
                    compiled_at = %at,
                    "POST-RECOMPILE RESUME: {}",
                    msg
                );
                Some(msg)
            } else {
                None
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to read resume state");
            None
        }
    };

    let _ = std::fs::remove_file(&resume_path);
    if result.is_some() {
        tracing::info!("Resume state consumed and deleted — will deliver to first WebSocket client");
    }
    result
}

/// Open the default browser — platform-neutral.
pub fn open_browser(url: &str) -> Result<()> {
    #[cfg(target_os = "macos")]
    { std::process::Command::new("open").arg(url).spawn()?; }
    #[cfg(target_os = "linux")]
    { std::process::Command::new("xdg-open").arg(url).spawn()?; }
    #[cfg(target_os = "windows")]
    { std::process::Command::new("cmd").args(["/C", "start", url]).spawn()?; }
    Ok(())
}

// ── Script/Binary Discovery Helpers ────────────────────────────────

fn find_kokoro_script() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok().map(std::path::PathBuf::from);
    let candidates = [
        home.as_ref().map(|h| h.join(".ernos/sandbox/scripts/start-kokoro.py")),
        Some(std::path::PathBuf::from("scripts/start-kokoro.py")),
    ];
    candidates.into_iter().flatten().find(|p| p.exists())
}

fn find_python312() -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        format!("{home}/.ernos/kokoro-venv/bin/python"),
        format!("{home}/.ernos/python/bin/python3.12"),
        "/opt/homebrew/bin/python3.12".to_string(),
        "/opt/homebrew/bin/python3.11".to_string(),
        "python3".to_string(),
    ];
    for c in &candidates {
        if std::path::Path::new(c).exists() { return c.clone(); }
    }
    "python3".to_string()
}

fn find_flux_script() -> Option<std::path::PathBuf> {
    let home = dirs::home_dir();
    let candidates = [
        Some(std::path::PathBuf::from("scripts/flux_server.py")),
        home.as_ref().map(|h| h.join(".ernos/sandbox/scripts/flux_server.py")),
    ];
    candidates.into_iter().flatten().find(|p| p.exists())
}

fn find_flux_launch_command(script: &std::path::Path) -> (String, Vec<String>) {
    let home = std::env::var("HOME").unwrap_or_default();
    let uv_candidates = [
        format!("{home}/.local/bin/uv"),
        format!("{home}/.cargo/bin/uv"),
        "/opt/homebrew/bin/uv".to_string(),
        "uv".to_string(),
    ];
    for uv in &uv_candidates {
        if std::path::Path::new(uv).exists() || uv == "uv" {
            if std::process::Command::new(uv).arg("--version").output().is_ok() {
                return (uv.clone(), vec!["run".to_string(), script.display().to_string()]);
            }
        }
    }

    let python = find_flux_python();
    tracing::warn!(python = %python, "uv not found — falling back to raw python (may lack deps)");
    (python, vec![script.display().to_string()])
}

fn find_flux_python() -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        format!("{home}/.ernos/flux-venv/bin/python"),
        format!("{home}/.ernos/python/bin/python3.12"),
        "/opt/homebrew/bin/python3.12".to_string(),
        "/opt/homebrew/bin/python3.11".to_string(),
        "python3".to_string(),
    ];
    for c in &candidates {
        if std::path::Path::new(c).exists() { return c.clone(); }
    }
    "python3".to_string()
}

fn find_code_server_binary() -> Option<String> {
    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        format!("{home}/.ernos/code-server-4.116.0-macos-arm64/bin/code-server"),
        "code-server".to_string(),
    ];
    for c in &candidates {
        if c.contains('/') {
            if std::path::Path::new(c).exists() { return Some(c.clone()); }
        } else {
            if std::process::Command::new(c).arg("--version")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status().is_ok()
            { return Some(c.clone()); }
        }
    }
    None
}
