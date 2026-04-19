// Ern-OS — High-performance, model-neutral Rust AI agent engine
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Entry point — initialises config, logging, provider, memory, launches
//! llama-server subprocess, and starts the WebUI hub.

use anyhow::{Context, Result};
use std::sync::Arc;
use tokio::sync::RwLock;

#[tokio::main]
async fn main() -> Result<()> {
    let config = ern_os::config::AppConfig::load()
        .context("Failed to load configuration")?;

    ern_os::logging::init(&config.general.data_dir)
        .context("Failed to initialise logging")?;

    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        data_dir = %config.general.data_dir.display(),
        provider = %config.general.active_provider,
        "Ern-OS starting"
    );

    let _server_handle = maybe_start_llama_server(&config).await?;
    let provider = create_and_verify_provider(&config).await?;
    let model_spec = detect_model_spec(&provider).await?;
    let state = build_app_state(&config, provider, model_spec)?;

    let _scheduler = ern_os::scheduler::start(state.clone());

    // Auto-start Kokoro TTS server if configured and not already running
    maybe_start_kokoro(&config).await;

    // Auto-start Flux image generation server
    maybe_start_flux(&config).await;

    // Auto-start code-server (VS Code IDE) if enabled
    maybe_start_code_server(&config).await;

    // Check for post-recompile resume state
    check_recompile_resume(&config);

    // Start platform router (forwards platform messages to hub)
    let router_registry = state.platforms.clone();
    let hub_port = config.web.port;
    tokio::spawn(async move {
        ern_os::platform::router::start_platform_router(router_registry, hub_port).await;
    });

    launch_webui(state, &config).await
}

/// Conditionally start llama-server subprocess.
async fn maybe_start_llama_server(
    config: &ern_os::config::AppConfig,
) -> Result<Option<tokio::process::Child>> {
    if config.general.active_provider != "llamacpp" {
        return Ok(None);
    }

    let llama_config = &config.llamacpp;
    let provider = ern_os::provider::llamacpp::LlamaCppProvider::new(llama_config);
    let args = provider.build_server_args();

    tracing::info!(
        binary = %llama_config.server_binary,
        model = %llama_config.model_path,
        port = llama_config.port,
        "Starting llama-server"
    );

    let child = tokio::process::Command::new(&llama_config.server_binary)
        .args(&args)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .context(format!(
            "Failed to start llama-server at '{}'. Is it installed?",
            llama_config.server_binary
        ))?;

    tracing::info!(pid = child.id().unwrap_or(0), "llama-server started");
    Ok(Some(child))
}

/// Create provider and verify health with retries.
async fn create_and_verify_provider(
    config: &ern_os::config::AppConfig,
) -> Result<Arc<dyn ern_os::provider::Provider>> {
    let provider: Arc<dyn ern_os::provider::Provider> = Arc::from(
        ern_os::provider::create_provider(config)
            .context("Failed to create provider")?
    );

    tracing::info!("Waiting for provider health check...");
    let mut retries = 0;
    while !provider.health().await {
        retries += 1;
        if retries > 60 {
            anyhow::bail!("Provider failed health check after 60 attempts. Is the server running?");
        }
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        if retries % 10 == 0 {
            tracing::info!(retries, "Still waiting for provider...");
        }
    }
    tracing::info!("Provider healthy");
    Ok(provider)
}

/// Detect model specification from the active provider.
async fn detect_model_spec(
    provider: &Arc<dyn ern_os::provider::Provider>,
) -> Result<ern_os::model::ModelSpec> {
    let spec = provider.get_model_spec().await
        .context("Failed to get model spec from provider")?;
    tracing::info!(
        model = %spec.name, context_length = spec.context_length,
        vision = spec.supports_vision, "Model detected"
    );
    Ok(spec)
}

/// Build the shared application state from all initialised components.
fn build_app_state(
    config: &ern_os::config::AppConfig,
    provider: Arc<dyn ern_os::provider::Provider>,
    model_spec: ern_os::model::ModelSpec,
) -> Result<ern_os::web::state::AppState> {
    let data_dir = config.general.data_dir.clone();
    let memory = ern_os::memory::MemoryManager::new(&data_dir)
        .context("Failed to initialise memory system")?;
    tracing::info!(status = %memory.status_summary(), "Memory system initialised");

    let golden_buffer = ern_os::learning::buffers::GoldenBuffer::open(
        &data_dir.join("golden_buffer.json"), 500,
    ).context("Failed to initialise golden buffer")?;

    let rejection_buffer = ern_os::learning::buffers_rejection::RejectionBuffer::open(
        &data_dir.join("rejection_buffer.json"),
    ).context("Failed to initialise rejection buffer")?;

    tracing::info!(
        golden = golden_buffer.count(),
        rejection = rejection_buffer.count(),
        "Training buffers initialised"
    );

    let scheduler = ern_os::scheduler::store::JobStore::load(&data_dir)
        .context("Failed to initialise scheduler")?;
    tracing::info!(jobs = scheduler.jobs.len(), "Scheduler initialised");

    let agents = ern_os::agents::AgentRegistry::new(&data_dir)
        .context("Failed to initialise agent registry")?;

    let teams = ern_os::agents::teams::TeamRegistry::new(&data_dir)
        .context("Failed to initialise team registry")?;

    Ok(ern_os::web::state::AppState {
        config: Arc::new(config.clone()),
        model_spec: Arc::new(model_spec),
        memory: Arc::new(RwLock::new(memory)),
        sessions: Arc::new(RwLock::new(
            ern_os::session::SessionManager::new(&data_dir.join("sessions"))
                .context("Failed to initialise session manager")?,
        )),
        provider,
        golden_buffer: Arc::new(RwLock::new(golden_buffer)),
        rejection_buffer: Arc::new(RwLock::new(rejection_buffer)),
        scheduler: Arc::new(RwLock::new(scheduler)),
        agents: Arc::new(RwLock::new(agents)),
        teams: Arc::new(RwLock::new(teams)),
        browser: Arc::new(RwLock::new(ern_os::tools::browser_tool::BrowserState::new())),
        platforms: Arc::new(RwLock::new(ern_os::platform::registry::PlatformRegistry::new())),
        mutable_config: Arc::new(RwLock::new(config.clone())),
    })
}

/// Start the WebUI hub and optionally open browser.
async fn launch_webui(
    state: ern_os::web::state::AppState,
    config: &ern_os::config::AppConfig,
) -> Result<()> {
    let addr = format!("0.0.0.0:{}", config.web.port);
    tracing::info!(addr = %addr, "Starting WebUI hub");

    if config.web.open_browser {
        let url = format!("http://localhost:{}", config.web.port);
        tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
            let _ = open_browser(&url);
        });
    }

    ern_os::web::server::run(state, &addr).await
        .context("WebUI server failed")
}

/// Auto-start Kokoro TTS server if not already running.
async fn maybe_start_kokoro(config: &ern_os::config::AppConfig) {
    let port = config.general.kokoro_port.unwrap_or(8880);
    let url = format!("http://127.0.0.1:{}/v1/models", port);

    // Check if already running
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap_or_default();
    if client.get(&url).send().await.map_or(false, |r| r.status().is_success()) {
        tracing::info!(port, "Kokoro TTS already running");
        return;
    }

    // Find the startup script
    let script = find_kokoro_script();
    let script_path = match script {
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
async fn maybe_start_flux(config: &ern_os::config::AppConfig) {
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

    let script = find_flux_script();
    let script_path = match script {
        Some(p) => p,
        None => {
            tracing::debug!("Flux server script not found — image generation disabled");
            return;
        }
    };

    tracing::info!(script = %script_path.display(), port, "Starting Flux image server");

    let python = find_flux_python();
    match tokio::process::Command::new(&python)
        .arg(&script_path)
        .env("FLUX_PORT", port.to_string())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(child) => {
            tracing::info!(pid = child.id().unwrap_or(0), "Flux image server spawned");
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to start Flux — image generation disabled");
        }
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

/// Find Python binary with Flux/torch dependencies — checks dedicated Flux venv first,
/// then falls back to generic python.
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

/// Find Python 3.10+ binary — prefers the kokoro venv (has all deps), then standalone, then homebrew.
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
async fn maybe_start_code_server(config: &ern_os::config::AppConfig) {
    if !config.codes.enabled { return; }
    let port = config.codes.port;
    let url = format!("http://127.0.0.1:{}/healthz", port);

    // Check if already running
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap_or_default();
    if client.get(&url).send().await.is_ok() {
        tracing::info!(port, "code-server already running");
        return;
    }

    let binary = find_code_server_binary();
    let binary_path = match binary {
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

/// Find code-server binary in known locations.
fn find_code_server_binary() -> Option<String> {
    let home = std::env::var("HOME").unwrap_or_default();
    let candidates = [
        format!("{home}/.ernos/code-server-4.116.0-macos-arm64/bin/code-server"),
        "code-server".to_string(), // PATH lookup
    ];
    for c in &candidates {
        if c.contains('/') {
            if std::path::Path::new(c).exists() { return Some(c.clone()); }
        } else {
            // Check PATH
            if std::process::Command::new(c).arg("--version")
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status().is_ok()
            { return Some(c.clone()); }
        }
    }
    None
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
fn open_browser(url: &str) -> Result<()> {
    #[cfg(target_os = "macos")]
    { std::process::Command::new("open").arg(url).spawn()?; }
    #[cfg(target_os = "linux")]
    { std::process::Command::new("xdg-open").arg(url).spawn()?; }
    #[cfg(target_os = "windows")]
    { std::process::Command::new("cmd").args(["/C", "start", url]).spawn()?; }
    Ok(())
}

/// Check for post-recompile resume state.
fn check_recompile_resume(config: &ern_os::config::AppConfig) {
    let resume_path = config.general.data_dir.join("resume.json");
    if resume_path.exists() {
        match std::fs::read_to_string(&resume_path) {
            Ok(content) => {
                if let Ok(resume) = serde_json::from_str::<serde_json::Value>(&content) {
                    let msg = resume["message"].as_str().unwrap_or("Recompile complete");
                    let at = resume["compiled_at"].as_str().unwrap_or("unknown");
                    tracing::info!(
                        compiled_at = %at,
                        "POST-RECOMPILE RESUME: {}",
                        msg
                    );
                }
            }
            Err(e) => tracing::warn!(error = %e, "Failed to read resume state"),
        }
        let _ = std::fs::remove_file(&resume_path);
        tracing::info!("Resume state consumed and deleted");
    }
}
