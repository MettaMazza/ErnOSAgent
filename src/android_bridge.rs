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

    // Build a minimal config for Android
    let data_path = std::path::PathBuf::from(&data_dir);

    // Ensure data directories exist
    let _ = std::fs::create_dir_all(data_path.join("sessions"));
    let _ = std::fs::create_dir_all(data_path.join("timeline"));
    let _ = std::fs::create_dir_all(data_path.join("steering"));
    let _ = std::fs::create_dir_all(data_path.join("logs"));
    let _ = std::fs::create_dir_all(data_path.join("prompts"));
    let _ = std::fs::create_dir_all(data_path.join("snapshots"));
    let _ = std::fs::create_dir_all(data_path.join("checkpoints"));

    // Start the tokio runtime and launch the engine
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to create Tokio runtime on Android");

    rt.block_on(async move {
        // Initialize logging to Android logcat
        tracing_subscriber::fmt()
            .with_max_level(tracing::Level::INFO)
            .init();

        tracing::info!(
            data_dir = %data_dir,
            provider_url = %provider_url,
            compute_mode = %mode,
            "Ern-OS Android engine starting"
        );

        // Load or create config
        let mut config = crate::config::AppConfig::default();
        config.general.data_dir = data_path;
        config.general.active_provider = "llamacpp".to_string();
        config.llamacpp.api_url = provider_url;
        config.web.port = 3000;
        config.web.open_browser = false;

        // Create provider
        let provider: std::sync::Arc<dyn crate::provider::Provider> =
            std::sync::Arc::from(
                crate::provider::create_provider(&config)
                    .expect("Failed to create provider on Android")
            );

        // Wait for provider health
        tracing::info!("Waiting for provider health...");
        let mut retries = 0;
        while !provider.health().await {
            retries += 1;
            if retries > 120 {
                tracing::error!("Provider not healthy after 120s — engine starting without provider");
                break;
            }
            tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        }

        let model_spec = provider.get_model_spec().await
            .unwrap_or_default();

        let state = crate::web::state::AppState {
            config: std::sync::Arc::new(config.clone()),
            model_spec: std::sync::Arc::new(model_spec),
            memory: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::memory::MemoryManager::new(&config.general.data_dir)
                    .expect("Failed to init memory")
            )),
            sessions: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::session::SessionManager::new(&config.general.data_dir.join("sessions"))
                    .expect("Failed to init sessions")
            )),
            provider,
            golden_buffer: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::learning::buffers::GoldenBuffer::new(500)
            )),
            rejection_buffer: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::learning::buffers_rejection::RejectionBuffer::new()
            )),
            scheduler: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::scheduler::store::JobStore::load(&config.general.data_dir)
                    .expect("Failed to init scheduler")
            )),
            agents: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::agents::AgentRegistry::new(&config.general.data_dir)
                    .expect("Failed to init agents")
            )),
            teams: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::agents::teams::TeamRegistry::new(&config.general.data_dir)
                    .expect("Failed to init teams")
            )),
            browser: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::tools::browser_tool::BrowserState::new()
            )),
            platforms: std::sync::Arc::new(tokio::sync::RwLock::new(
                crate::platform::registry::PlatformRegistry::new()
            )),
            mutable_config: std::sync::Arc::new(tokio::sync::RwLock::new(config.clone())),
            resume_message: std::sync::Arc::new(tokio::sync::RwLock::new(None)),
            sae: std::sync::Arc::new(tokio::sync::RwLock::new(None)),
        };

        let addr = "0.0.0.0:3000";
        tracing::info!(addr, "Ern-OS Android WebUI starting");

        if let Err(e) = crate::web::server::run(state, addr).await {
            tracing::error!(error = %e, "Ern-OS Android server failed");
        }
    });
}
