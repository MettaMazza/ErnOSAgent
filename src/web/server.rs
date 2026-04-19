// Ern-OS — High-performance, model-neutral Rust AI agent engine
// Created by @mettamazza (github.com/mettamazza)
// License: MIT
//! Axum web server — thin router orchestrator. Handlers live in `handlers/`.

use crate::web::state::AppState;
use crate::web::handlers::{system, sessions, memory, scheduler, onboarding, api_keys, agents, content, tts, codes, platforms, voice, video, upload, version, checkpoint};
use anyhow::Result;
use axum::{Router, routing::{get, post, put, delete}};
use tower_http::cors::CorsLayer;
use std::net::SocketAddr;

/// Start the WebUI hub server — the only public interface.
pub async fn run(state: AppState, addr: &str) -> Result<()> {
    // Load persisted API keys into process env so web_search picks them up
    api_keys::load_into_env(&state);

    let app = Router::new()
        // Static files
        .route("/", get(content::index))
        .route("/app.css", get(content::css))
        .route("/app.js", get(content::js))
        .route("/api/images/{filename}", get(content::serve_image))
        // REST API — Sessions
        .route("/api/sessions", get(sessions::list_sessions))
        .route("/api/sessions", post(sessions::create_session))
        .route("/api/sessions/search", get(sessions::search_sessions))
        .route("/api/sessions/{id}", get(sessions::get_session))
        .route("/api/sessions/{id}", put(sessions::rename_session))
        .route("/api/sessions/{id}", delete(sessions::delete_session))
        .route("/api/sessions/{id}/pin", put(sessions::toggle_pin))
        .route("/api/sessions/{id}/archive", put(sessions::toggle_archive))
        .route("/api/sessions/{id}/export", get(sessions::export_session))
        .route("/api/sessions/{id}/fork/{idx}", post(sessions::fork_session))
        .route("/api/sessions/{id}/messages/{idx}", delete(sessions::delete_message))
        .route("/api/sessions/{id}/messages/{idx}/react", post(sessions::react_message))
        // REST API — TTS
        .route("/api/tts", post(tts::synthesize))
        .route("/api/tts/status", get(tts::tts_status))
        // REST API — Codes IDE
        .route("/api/codes/status", get(codes::codes_status))
        // REST API — Platforms
        .route("/api/platforms", get(platforms::list_platforms))
        .route("/api/platforms/config", get(platforms::get_platform_config).put(platforms::update_platform_config))
        .route("/api/platforms/{name}/connect", post(platforms::connect_platform))
        .route("/api/platforms/{name}/disconnect", post(platforms::disconnect_platform))
        .route("/api/chat/platform", post(platforms::platform_ingest))
        // REST API — System
        .route("/api/health", get(system::health_check))
        .route("/api/status", get(system::system_status))
        .route("/api/models", get(system::list_models))
        .route("/api/factory-reset", post(system::factory_reset))
        // REST API — Memory (all 7 tiers)
        .route("/api/memory/stats", get(memory::stats))
        .route("/api/memory/timeline", get(memory::timeline))
        .route("/api/memory/lessons", get(memory::lessons))
        .route("/api/memory/procedures", get(memory::procedures))
        .route("/api/memory/scratchpad", get(memory::scratchpad))
        .route("/api/memory/synaptic", get(memory::synaptic))
        // REST API — Tools
        .route("/api/tools", get(system::tools_catalog))
        // REST API — Training
        .route("/api/training", get(system::training_buffers))
        // REST API — Interpretability
        .route("/api/interpretability/features", get(system::interp_features))
        .route("/api/interpretability/snapshots", get(system::interp_snapshots))
        .route("/api/interpretability/sae", get(system::interp_sae))
        // REST API — Steering
        .route("/api/steering/vectors", get(system::steering_vectors))
        // REST API — Learning pipeline
        .route("/api/learning/status", get(system::learning_status))
        .route("/api/learning/adapters", get(system::learning_adapters))
        .route("/api/learning/sleep-history", get(system::learning_sleep_history))
        // REST API — Observer / Logs / Scheduler
        .route("/api/observer/history", get(system::observer_history))
        .route("/api/logs", get(system::logs_recent))
        .route("/api/self-edits", get(system::self_edits))
        .route("/api/checkpoints", get(system::checkpoints))
        .route("/api/scheduler", get(scheduler::status))
        .route("/api/scheduler/jobs", post(scheduler::create_job))
        .route("/api/scheduler/jobs/{id}", delete(scheduler::delete_job))
        .route("/api/scheduler/jobs/{id}/toggle", put(scheduler::toggle_job))
        .route("/api/scheduler/history", get(scheduler::history))
        // REST API — Onboarding & Identity
        .route("/api/onboarding/status", get(onboarding::status))
        .route("/api/onboarding/profile", post(onboarding::save_profile))
        .route("/api/onboarding/complete", post(onboarding::complete))
        .route("/api/prompts/{name}", get(system::get_prompt))
        .route("/api/prompts/{name}", put(system::save_prompt))
        // REST API — Agents & Teams
        .route("/api/agents", get(agents::list_agents))
        .route("/api/agents", post(agents::create_agent))
        .route("/api/agents/{id}", get(agents::get_agent))
        .route("/api/agents/{id}", put(agents::update_agent))
        .route("/api/agents/{id}", delete(agents::delete_agent))
        .route("/api/teams", get(agents::list_teams))
        .route("/api/teams", post(agents::create_team))
        .route("/api/teams/{id}", delete(agents::delete_team))
        // REST API — API Keys
        .route("/api/api-keys", get(api_keys::get_keys))
        .route("/api/api-keys", put(api_keys::save_keys))
        // REST API — File Upload
        .route("/api/upload", post(upload::upload_file))
        // REST API — Version Management
        .route("/api/version", get(version::get_version))
        .route("/api/version/check", get(version::check_updates))
        .route("/api/version/update", post(version::update_version))
        .route("/api/version/rollback", post(version::rollback_version))
        .route("/api/version/history", get(version::version_history))
        // REST API — State Checkpoints (atomic snapshots)
        .route("/api/state-checkpoint", get(checkpoint::list_checkpoints))
        .route("/api/state-checkpoint", post(checkpoint::create_checkpoint))
        .route("/api/state-checkpoint/restore", post(checkpoint::restore_checkpoint))
        .route("/api/state-checkpoint/{id}", delete(checkpoint::delete_checkpoint))
        // WebSocket
        .route("/ws", get(crate::web::ws::ws_handler))
        .route("/ws/voice", get(voice::ws_voice_handler))
        .route("/ws/video", get(video::ws_video_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

    let addr: SocketAddr = addr.parse()?;
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "WebUI hub listening");

    axum::serve(listener, app).await?;
    Ok(())
}
