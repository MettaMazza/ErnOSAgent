//! Android JNI bridge — entry point called from Kotlin EngineService.
//!
//! This module is only compiled when the `android` feature is enabled.
//! It provides a C-exported function that Kotlin calls via JNI to start
//! the Ern-OS engine (Axum server) inside the Android foreground service.

#[cfg(feature = "android")]
use jni::JNIEnv;
#[cfg(feature = "android")]
use jni::objects::{JClass, JString};

/// JNI entry point — starts the Ern-OS engine inside an Android service.
///
/// Called from Kotlin: `external fun startEngine(dataDir: String, providerUrl: String)`
///
/// # Arguments
/// - `data_dir`: App-private data directory (e.g., `/data/data/com.ernos.app/files`)
/// - `provider_url`: llama-server URL (e.g., `http://127.0.0.1:8080` for local mode)
/// - `compute_mode`: One of "local", "hybrid", "host"
#[cfg(feature = "android")]
#[no_mangle]
pub extern "C" fn Java_com_ernos_app_EngineService_startEngine(
    mut env: JNIEnv,
    _class: JClass,
    data_dir: JString,
    provider_url: JString,
    compute_mode: JString,
) {
    let data_dir: String = env.get_string(&data_dir)
        .map(|s| s.into())
        .unwrap_or_else(|_| "/data/data/com.ernos.app/files".to_string());

    let provider_url: String = env.get_string(&provider_url)
        .map(|s| s.into())
        .unwrap_or_else(|_| "http://127.0.0.1:8080".to_string());

    let mode: String = env.get_string(&compute_mode)
        .map(|s| s.into())
        .unwrap_or_else(|_| "local".to_string());

    let data_path = std::path::PathBuf::from(&data_dir);
    ensure_data_dirs(&data_path);

    let rt = match tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("Ern-OS: Failed to create Tokio runtime: {e}");
            return;
        }
    };

    rt.block_on(async move {
        init_logging();

        tracing::info!(
            data_dir = %data_dir,
            provider_url = %provider_url,
            compute_mode = %mode,
            "Ern-OS Android engine starting"
        );

        let config = build_config(&data_path, &provider_url);

        // Create provider (instant — no blocking health check yet)
        let provider = match create_provider_no_wait(&config) {
            Some(p) => p,
            None => return,
        };

        // Wait for provider health before querying model spec (auto-derived, per governance §2.1/§5)
        tracing::info!("Waiting for provider health before querying model spec...");
        let mut retries = 0;
        while !provider.health().await {
            retries += 1;
            if retries > 120 {
                tracing::error!("Provider not healthy after 120s — cannot start without inference");
                return;
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }
        tracing::info!(retries, "Provider healthy — querying model spec");

        let model_spec = match provider.get_model_spec().await {
            Ok(spec) => {
                tracing::info!(
                    name = %spec.name,
                    context_length = spec.context_length,
                    supports_thinking = spec.supports_thinking,
                    "Model spec derived from provider"
                );
                spec
            }
            Err(e) => {
                tracing::error!(error = %e, "Failed to get model spec from provider");
                return;
            }
        };

        let state = match build_app_state(&config, provider.clone(), model_spec) {
            Some(s) => s,
            None => return,
        };

        // Start web server IMMEDIATELY — UI is responsive while provider connects
        let addr = "0.0.0.0:3000";
        tracing::info!(addr, "Ern-OS Android WebUI starting");

        if let Err(e) = crate::web::server::run(state, addr).await {
            tracing::error!(error = %e, "Ern-OS Android server failed");
        }
    });
}

/// Ensure all required data directories exist.
#[cfg(feature = "android")]
fn ensure_data_dirs(data_path: &std::path::Path) {
    let dirs = ["sessions", "timeline", "steering", "logs", "prompts", "snapshots", "checkpoints"];
    for dir in &dirs {
        let _ = std::fs::create_dir_all(data_path.join(dir));
    }
}

/// Initialize tracing subscriber for Android logcat.
#[cfg(feature = "android")]
fn init_logging() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();
}

/// Build the application config from Android-provided paths.
#[cfg(feature = "android")]
fn build_config(
    data_path: &std::path::Path,
    provider_url: &str,
) -> crate::config::AppConfig {
    let mut config = crate::config::AppConfig::default();
    config.general.data_dir = data_path.to_path_buf();
    config.general.active_provider = "llamacpp".to_string();
    // Parse port from provider URL (e.g. "http://127.0.0.1:8080")
    let port: u16 = provider_url
        .rsplit(':')
        .next()
        .and_then(|p| p.trim_end_matches('/').parse().ok())
        .unwrap_or(8080);
    config.llamacpp.port = port;
    config.llamacpp.server_binary = data_path.join("bin/llama-server").to_string_lossy().to_string();
    config.web.port = 3000;
    config.web.open_browser = false;
    config
}

/// Create the inference provider without blocking on health check.
#[cfg(feature = "android")]
fn create_provider_no_wait(
    config: &crate::config::AppConfig,
) -> Option<std::sync::Arc<dyn crate::provider::Provider>> {
    match crate::provider::create_provider(config) {
        Ok(p) => {
            tracing::info!("Provider created (health check deferred to background)");
            Some(std::sync::Arc::from(p))
        }
        Err(e) => {
            tracing::error!(error = %e, "Failed to create provider on Android");
            None
        }
    }
}

/// Build AppState with clean error handling — logs and returns None on failure.
#[cfg(feature = "android")]
fn build_app_state(
    config: &crate::config::AppConfig,
    provider: std::sync::Arc<dyn crate::provider::Provider>,
    model_spec: crate::model::ModelSpec,
) -> Option<crate::web::state::AppState> {
    let data_dir = &config.general.data_dir;

    let memory = match crate::memory::MemoryManager::new(data_dir) {
        Ok(m) => m,
        Err(e) => { tracing::error!(error = %e, "Failed to init memory"); return None; }
    };
    let sessions = match crate::session::SessionManager::new(&data_dir.join("sessions")) {
        Ok(s) => s,
        Err(e) => { tracing::error!(error = %e, "Failed to init sessions"); return None; }
    };
    let scheduler = match crate::scheduler::store::JobStore::load(data_dir) {
        Ok(s) => s,
        Err(e) => { tracing::error!(error = %e, "Failed to init scheduler"); return None; }
    };
    let agents = match crate::agents::AgentRegistry::new(data_dir) {
        Ok(a) => a,
        Err(e) => { tracing::error!(error = %e, "Failed to init agents"); return None; }
    };
    let teams = match crate::agents::teams::TeamRegistry::new(data_dir) {
        Ok(t) => t,
        Err(e) => { tracing::error!(error = %e, "Failed to init teams"); return None; }
    };

    Some(crate::web::state::AppState {
        config: std::sync::Arc::new(config.clone()),
        model_spec: std::sync::Arc::new(model_spec),
        memory: std::sync::Arc::new(tokio::sync::RwLock::new(memory)),
        sessions: std::sync::Arc::new(tokio::sync::RwLock::new(sessions)),
        provider,
        golden_buffer: std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::learning::buffers::GoldenBuffer::new(500)
        )),
        rejection_buffer: std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::learning::buffers_rejection::RejectionBuffer::new()
        )),
        scheduler: std::sync::Arc::new(tokio::sync::RwLock::new(scheduler)),
        agents: std::sync::Arc::new(tokio::sync::RwLock::new(agents)),
        teams: std::sync::Arc::new(tokio::sync::RwLock::new(teams)),
        browser: std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::tools::browser_tool::BrowserState::new()
        )),
        platforms: std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::platform::registry::PlatformRegistry::new()
        )),
        mutable_config: std::sync::Arc::new(tokio::sync::RwLock::new(config.clone())),
        resume_message: std::sync::Arc::new(tokio::sync::RwLock::new(None)),
        sae: std::sync::Arc::new(tokio::sync::RwLock::new(None)),
        live_monitor: std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::interpretability::live::LiveMonitor::new(50),
        )),
        snapshot_store: std::sync::Arc::new(tokio::sync::RwLock::new(
            crate::interpretability::snapshot::SnapshotStore::new(
                &data_dir.join("snapshots"),
            ).unwrap_or_else(|_| {
                crate::interpretability::snapshot::SnapshotStore::new(
                    &std::path::Path::new("/tmp/ern-os-snapshots"),
                ).expect("fallback snapshot store")
            }),
        )),
        cancel_flag: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
    })
}
