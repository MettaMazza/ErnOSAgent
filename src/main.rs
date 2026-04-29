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

    start_optional_services(&config, &state).await;
    auto_connect_platforms(&config, &state).await;
    start_platform_router(&state, config.web.port);

    launch_webui(state, &config).await
}

/// Start optional services (Kokoro TTS, Flux image gen, code server, embedding server, SAE sidecar).
async fn start_optional_services(config: &ern_os::config::AppConfig, state: &ern_os::web::state::AppState) {
    ern_os::startup::maybe_start_kokoro(config).await;
    ern_os::startup::maybe_start_flux(config).await;
    ern_os::startup::maybe_start_code_server(config).await;
    ern_os::startup::maybe_start_embedding_server(config).await;

    // Start SAE embedding sidecar if SAE weights are loaded
    if state.sae.read().await.is_some() {
        start_sae_sidecar(config).await;
    }

    let resume_msg = ern_os::startup::check_recompile_resume(config);
    if resume_msg.is_some() {
        *state.resume_message.write().await = resume_msg;
    }
}

/// Auto-connect platforms that have tokens configured.
async fn auto_connect_platforms(config: &ern_os::config::AppConfig, state: &ern_os::web::state::AppState) {
    let mut reg = state.platforms.write().await;
    if config.discord.resolve_token().is_some() {
        match reg.connect_by_name("Discord").await {
            Ok(_) => tracing::info!("Discord auto-connected at startup"),
            Err(e) => tracing::error!(error = %e, "Discord startup connect failed"),
        }
    }
    if config.telegram.resolve_token().is_some() {
        match reg.connect_by_name("Telegram").await {
            Ok(_) => tracing::info!("Telegram auto-connected at startup"),
            Err(e) => tracing::error!(error = %e, "Telegram startup connect failed"),
        }
    }
}

/// Start the platform router in a background task.
fn start_platform_router(state: &ern_os::web::state::AppState, hub_port: u16) {
    let registry = state.platforms.clone();
    tokio::spawn(async move {
        ern_os::platform::router::start_platform_router(registry, hub_port).await;
    });
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

    // Kill any stale llama-server processes from previous runs
    // that may still be holding the port.
    let port_str = llama_config.port.to_string();
    let kill_result = tokio::process::Command::new("pkill")
        .args(["-f", &format!("llama-server.*--port {}", port_str)])
        .output()
        .await;
    if let Ok(output) = &kill_result {
        if output.status.success() {
            tracing::info!(port = llama_config.port, "Killed stale llama-server process");
            // Give the OS a moment to free the port
            tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;
        }
    }

    let child = tokio::process::Command::new(&llama_config.server_binary)
        .args(&args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .context(format!(
            "Failed to start llama-server at '{}'. Is it installed?",
            llama_config.server_binary
        ))?;

    tracing::info!(pid = child.id().unwrap_or(0), "llama-server started");
    Ok(Some(child))
}

/// Start a lightweight SAE embedding sidecar — same model in --embeddings mode.
/// This provides the residual stream activations the SAE was trained on.
async fn start_sae_sidecar(config: &ern_os::config::AppConfig) {
    let llama_config = &config.llamacpp;
    let port = llama_config.sae_embed_port;

    // Check if something is already running on the port
    if reqwest::Client::new()
        .get(format!("http://localhost:{}/health", port))
        .send().await
        .is_ok()
    {
        tracing::info!(port, "SAE embed sidecar already running");
        return;
    }

    tracing::info!(
        port, model = %llama_config.model_path,
        "Starting SAE embedding sidecar"
    );

    match tokio::process::Command::new(&llama_config.server_binary)
        .args([
            "--model", &llama_config.model_path,
            "--port", &port.to_string(),
            "--n-gpu-layers", &llama_config.n_gpu_layers.to_string(),
            "--embeddings",
            "--pooling", "mean",
            "--batch-size", "2048",
            "--ubatch-size", "2048",
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(child) => {
            tracing::info!(pid = child.id().unwrap_or(0), port, "SAE embed sidecar started");
        }
        Err(e) => {
            tracing::warn!(error = %e, "Failed to start SAE embed sidecar — live interpretability disabled");
        }
    }
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

    let curriculum = ern_os::learning::curriculum::CurriculumStore::open(
        &data_dir.join("curriculum"),
    ).context("Failed to initialise curriculum store")?;
    tracing::info!(courses = curriculum.course_count(), "Curriculum store initialised");

    let quarantine = ern_os::learning::verification::QuarantineBuffer::open(
        &data_dir.join("quarantine.json"),
    ).context("Failed to initialise quarantine buffer")?;
    tracing::info!(quarantine = quarantine.count(), "Quarantine buffer initialised");

    let review_deck = ern_os::learning::review::ReviewDeck::open(
        &data_dir.join("review_deck.json"),
    ).context("Failed to initialise review deck")?;
    tracing::info!(cards = review_deck.count(), "Review deck initialised");

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
        browser: Arc::new(RwLock::new(ern_os::tools::browser_tool::BrowserState::with_config(config.browser.clone()))),
        platforms: {
            let mut registry = ern_os::platform::registry::PlatformRegistry::new();
            registry.register(Box::new(
                ern_os::platform::discord::DiscordAdapter::new(config.discord.clone(), config.web.port),
            ));
            registry.register(Box::new(
                ern_os::platform::telegram::TelegramAdapter::new(config.telegram.clone()),
            ));
            Arc::new(RwLock::new(registry))
        },
        mutable_config: Arc::new(RwLock::new(config.clone())),
        resume_message: Arc::new(RwLock::new(None)),
        sae: Arc::new(RwLock::new(load_sae_weights())),
        live_monitor: Arc::new(RwLock::new(
            ern_os::interpretability::live::LiveMonitor::new(50),
        )),
        snapshot_store: Arc::new(RwLock::new(
            ern_os::interpretability::snapshot::SnapshotStore::new(
                &data_dir.join("snapshots"),
            ).unwrap_or_else(|e| {
                tracing::warn!(error = %e, "Failed to init snapshot store — creating empty");
                ern_os::interpretability::snapshot::SnapshotStore::new(
                    &std::path::Path::new("/tmp/ern-os-snapshots"),
                ).expect("fallback snapshot store")
            }),
        )),
        cancel_flag: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        curriculum: Arc::new(RwLock::new(curriculum)),
        quarantine: Arc::new(RwLock::new(quarantine)),
        review_deck: Arc::new(RwLock::new(review_deck)),
    })
}

/// Load SAE weights from models/sae/ directory at startup.
fn load_sae_weights() -> Option<ern_os::interpretability::sae::SparseAutoencoder> {
    let sae_dir = std::path::Path::new("models/sae");
    if !sae_dir.exists() {
        tracing::debug!("No models/sae/ directory — SAE interpretability disabled");
        return None;
    }

    // Find the first .safetensors file
    let entry = std::fs::read_dir(sae_dir).ok()?.find_map(|e| {
        let e = e.ok()?;
        if e.path().extension().map_or(false, |ext| ext == "safetensors") {
            Some(e.path())
        } else {
            None
        }
    });

    match entry {
        Some(path) => {
            tracing::info!(path = %path.display(), "Loading SAE weights...");
            match ern_os::interpretability::sae::SparseAutoencoder::load_safetensors(&path) {
                Ok(sae) => {
                    tracing::info!(
                        features = sae.num_features,
                        model_dim = sae.model_dim,
                        "SAE loaded — interpretability encode active"
                    );
                    Some(sae)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Failed to load SAE weights");
                    None
                }
            }
        }
        None => {
            tracing::debug!("No .safetensors files in models/sae/ — SAE disabled");
            None
        }
    }
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
            let _ = ern_os::startup::open_browser(&url);
        });
    }

    ern_os::web::server::run(state, &addr).await
        .context("WebUI server failed")
}



