//! Browser tool stub for non-desktop platforms (Android).
//! Provides the same public API surface as browser_tool.rs but returns
//! "not available on mobile" for all operations.

use std::sync::Arc;
use tokio::sync::RwLock;

/// Stub browser state — no actual browser.
pub struct BrowserState;

impl BrowserState {
    pub fn new() -> Self { Self }
    pub fn with_config(_config: crate::config::BrowserConfig) -> Self { Self }
}

const MOBILE_MSG: &str = "This tool requires the desktop engine (headless Chrome). \
Switch to Host mode to access browser tools, or use the desktop WebUI.";

pub async fn browse_url(
    _browser: &Arc<RwLock<BrowserState>>,
    _url: &str,
) -> anyhow::Result<String> {
    Ok(MOBILE_MSG.to_string())
}

pub async fn screenshot_url(
    _browser: &Arc<RwLock<BrowserState>>,
    _url: &str,
) -> anyhow::Result<String> {
    Ok(MOBILE_MSG.to_string())
}

pub async fn execute_action(
    _browser: &Arc<RwLock<BrowserState>>,
    _args: &serde_json::Value,
) -> anyhow::Result<String> {
    Ok(MOBILE_MSG.to_string())
}
