//! ErnOSAgent — Local Multi-Modal Agentic Reasoning HUD
//!
//! Central command center for multi-modal agentic reasoning and tool calling.
//! Built for the Gemma 4 model family with llama.cpp as the primary inference backend.

pub mod config;
pub mod logging;
pub mod model;
pub mod provider;
pub mod inference;
pub mod steering;
pub mod prompt;
pub mod session;
pub mod react;
pub mod observer;
pub mod memory;
pub mod tools;
pub mod audio;
pub mod platform;
pub mod ui;
pub mod app;

use anyhow::{Context, Result};

#[tokio::main]
async fn main() -> Result<()> {
    let config = config::AppConfig::load()
        .context("Failed to load configuration")?;

    // Ensure all data directories exist
    let dirs = [
        config.sessions_dir(),
        config.logs_dir(),
        config.vectors_dir(),
        config.timeline_dir(),
    ];
    for dir in &dirs {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("Failed to create directory: {}", dir.display()))?;
    }

    // Initialise per-session logging
    let initial_session_id = uuid::Uuid::new_v4().to_string();
    let _logging = logging::init_logging(&config.logs_dir(), &initial_session_id)
        .context("Failed to initialise logging system")?;

    tracing::info!(
        provider = %config.general.active_provider,
        model = %config.general.active_model,
        data_dir = %config.general.data_dir.display(),
        "ErnOSAgent starting"
    );

    // Launch the TUI application
    app::run(config).await
}
