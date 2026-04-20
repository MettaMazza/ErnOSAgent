//! Service launchers — external subprocess management for Ern-OS.
//! Manages Kokoro TTS, Flux image gen, and code-server (VS Code IDE).

/// Auto-start Kokoro TTS server if not already running.
pub async fn maybe_start_kokoro(config: &ern_os::config::AppConfig) {
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
pub async fn maybe_start_flux(config: &ern_os::config::AppConfig) {
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

    match spawn_flux_process(&cmd_bin, &cmd_args, port).await {
        Some(mut child) => {
            wait_for_flux_health(&mut child, &client, &url, &cmd_bin).await;
        }
        None => {}
    }
}

/// Spawn the Flux server subprocess.
async fn spawn_flux_process(
    cmd_bin: &str,
    cmd_args: &[String],
    port: u16,
) -> Option<tokio::process::Child> {
    match tokio::process::Command::new(cmd_bin)
        .args(cmd_args)
        .env("FLUX_PORT", port.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(child) => {
            tracing::info!(pid = child.id().unwrap_or(0), cmd = %cmd_bin, "Flux image server spawned");
            Some(child)
        }
        Err(e) => {
            tracing::warn!(error = %e, cmd = %cmd_bin, "Failed to start Flux — image generation disabled");
            None
        }
    }
}

/// Wait up to 60s for Flux server health check.
async fn wait_for_flux_health(
    child: &mut tokio::process::Child,
    client: &reqwest::Client,
    url: &str,
    cmd_bin: &str,
) {
    let pid = child.id().unwrap_or(0);
    for i in 0..60 {
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;

        if let Ok(Some(status)) = child.try_wait() {
            let stderr = read_child_stderr(child).await;
            tracing::error!(pid, exit = %status, stderr = %stderr.trim(), "Flux server crashed during startup");
            return;
        }

        if client.get(url).send().await.map_or(false, |r| r.status().is_success()) {
            tracing::info!(pid, seconds = i + 1, "Flux image server ready");
            return;
        }
    }
    tracing::warn!(pid, cmd = %cmd_bin, "Flux server spawned but not healthy after 60s — may still be loading");
}

/// Read stderr from a child process.
async fn read_child_stderr(child: &mut tokio::process::Child) -> String {
    if let Some(mut err) = child.stderr.take() {
        let mut buf = String::new();
        let _ = tokio::io::AsyncReadExt::read_to_string(&mut err, &mut buf).await;
        buf
    } else {
        String::new()
    }
}

/// Search for the Flux server script.
fn find_flux_script() -> Option<std::path::PathBuf> {
    let home = dirs::home_dir();
    let candidates = [
        Some(std::path::PathBuf::from("scripts/flux_server.py")),
        home.as_ref().map(|h| h.join(".ernos/sandbox/scripts/flux_server.py")),
    ];
    candidates.into_iter().flatten().find(|p| p.exists())
}

/// Determine how to launch the Flux server script.
/// Prefers `uv run` (manages its own venv with correct deps) over raw python.
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

/// Find Python binary — fallback only, prefer uv run.
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

/// Find Python 3.10+ binary for Kokoro TTS.
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

/// Auto-start code-server (VS Code IDE) if enabled and not already running.
pub async fn maybe_start_code_server(config: &ern_os::config::AppConfig) {
    if !config.codes.enabled { return; }
    let port = config.codes.port;

    if is_service_running(&format!("http://127.0.0.1:{}/healthz", port)).await {
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

    install_bundled_extensions();
    install_code_server_settings();

    tracing::info!(binary = %binary_path, port, workspace = %workspace.display(), "Starting code-server");
    let ext_dir = bundled_extensions_dir();
    let user_data_dir = code_server_user_data_dir();
    match tokio::process::Command::new(&binary_path)
        .args([
            "--port", &port.to_string(),
            "--auth", "none",
            "--disable-telemetry",
            "--disable-update-check",
            "--extensions-dir", &ext_dir,
            "--user-data-dir", &user_data_dir,
            &workspace.to_string_lossy(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(child) => tracing::info!(pid = child.id().unwrap_or(0), "code-server started"),
        Err(e) => tracing::warn!(error = %e, "Failed to start code-server"),
    }
}

/// Check if a service is responding at a given URL.
async fn is_service_running(url: &str) -> bool {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap_or_default();
    client.get(url).send().await.is_ok()
}

/// Find code-server binary in known locations.
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

/// Directory where bundled extensions are installed for code-server.
fn bundled_extensions_dir() -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    format!("{home}/.ernos/code-server-extensions")
}

/// User data directory for code-server (settings, keybindings, etc.)
fn code_server_user_data_dir() -> String {
    let home = std::env::var("HOME").unwrap_or_default();
    format!("{home}/.ernos/code-server-data")
}

/// Install settings.json that disables built-in Copilot Chat and configures Ern-OS defaults.
fn install_code_server_settings() {
    let data_dir = code_server_user_data_dir();
    let user_dir = format!("{data_dir}/User");
    if let Err(e) = std::fs::create_dir_all(&user_dir) {
        tracing::warn!(error = %e, "Failed to create code-server User dir");
        return;
    }

    let settings_path = format!("{user_dir}/settings.json");
    let settings = serde_json::json!({
        // Disable built-in GitHub Copilot Chat features
        "github.copilot.enable": { "*": false },
        "github.copilot.editor.enableAutoCompletions": false,
        "chat.experimental.enabled": false,
        "inlineChat.enabled": false,
        // Hide the built-in Chat panel
        "workbench.panel.chat.hidden": true,
        // Ern-OS panel visible by default
        "workbench.panel.defaultViewContainerVisibility": "visible",
        // Theme and font defaults
        "editor.fontFamily": "'SF Mono', 'Fira Code', Menlo, monospace",
        "editor.fontSize": 13,
        "editor.fontLigatures": true,
        "editor.minimap.enabled": false,
        "workbench.colorTheme": "Default Dark Modern",
        "workbench.startupEditor": "none",
        "telemetry.telemetryLevel": "off",
    });

    match std::fs::write(&settings_path, serde_json::to_string_pretty(&settings).unwrap_or_default()) {
        Ok(_) => tracing::debug!("code-server settings installed"),
        Err(e) => tracing::warn!(error = %e, "Failed to write code-server settings"),
    }
}

/// Copy bundled extensions from the project into the code-server extensions dir.
fn install_bundled_extensions() {
    let src = std::path::Path::new("extensions");
    if !src.exists() { return; }

    let dest = std::path::PathBuf::from(bundled_extensions_dir());
    if let Err(e) = std::fs::create_dir_all(&dest) {
        tracing::warn!(error = %e, "Failed to create extensions dir");
        return;
    }

    if let Ok(entries) = std::fs::read_dir(src) {
        for entry in entries.flatten() {
            let ext_name = entry.file_name();
            let target = dest.join(&ext_name);
            if entry.path().is_dir() {
                copy_dir_recursive(&entry.path(), &target);
                tracing::info!(ext = ?ext_name, "Bundled extension installed");
            }
        }
    }
}

/// Recursively copy a directory.
fn copy_dir_recursive(src: &std::path::Path, dst: &std::path::Path) {
    let _ = std::fs::create_dir_all(dst);
    if let Ok(entries) = std::fs::read_dir(src) {
        for entry in entries.flatten() {
            let target = dst.join(entry.file_name());
            if entry.path().is_dir() {
                copy_dir_recursive(&entry.path(), &target);
            } else {
                let _ = std::fs::copy(entry.path(), &target);
            }
        }
    }
}

/// Search for the Kokoro startup script in known locations.
fn find_kokoro_script() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok().map(std::path::PathBuf::from);
    let candidates = [
        home.as_ref().map(|h| h.join(".ernos/sandbox/scripts/start-kokoro.py")),
        Some(std::path::PathBuf::from("scripts/start-kokoro.py")),
    ];
    for candidate in candidates.into_iter().flatten() {
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// Open the default browser — platform-neutral.
pub fn open_browser(url: &str) -> anyhow::Result<()> {
    #[cfg(target_os = "macos")]
    { std::process::Command::new("open").arg(url).spawn()?; }
    #[cfg(target_os = "linux")]
    { std::process::Command::new("xdg-open").arg(url).spawn()?; }
    #[cfg(target_os = "windows")]
    { std::process::Command::new("cmd").args(["/C", "start", url]).spawn()?; }
    Ok(())
}

/// Check for post-recompile resume state. Returns the resume message if found.
pub fn check_recompile_resume(config: &ern_os::config::AppConfig) -> Option<String> {
    let resume_path = config.general.data_dir.join("resume.json");
    if !resume_path.exists() { return None; }

    let result = match std::fs::read_to_string(&resume_path) {
        Ok(content) => {
            if let Ok(resume) = serde_json::from_str::<serde_json::Value>(&content) {
                let msg = resume["message"].as_str().unwrap_or("Recompile complete").to_string();
                tracing::info!(compiled_at = %resume["compiled_at"].as_str().unwrap_or("unknown"), "POST-RECOMPILE RESUME: {}", msg);
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
        tracing::info!("Resume state consumed and deleted");
    }
    result
}
