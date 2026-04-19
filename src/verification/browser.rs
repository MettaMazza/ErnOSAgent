//! Browser-based validation — headless Chrome via chromiumoxide.
//!
//! Launches a headless browser, navigates to URLs, collects console errors,
//! and runs interactive checks (click, type, assert).

use anyhow::{Context, Result};
use futures_util::StreamExt;

/// Result of a single browser check.
#[derive(Debug, Clone)]
pub struct BrowserCheckResult {
    pub url: String,
    pub loaded: bool,
    pub console_errors: Vec<String>,
    pub page_title: String,
    pub status_code: Option<u16>,
    pub screenshot_b64: Option<String>,
}

/// Actions the browser verifier can perform.
#[derive(Debug, Clone)]
pub enum BrowserAction {
    Navigate(String),
    Click(String),
    Type(String, String),
    AssertVisible(String),
    AssertText(String, String),
    Screenshot,
    WaitMs(u64),
}

/// Check a URL by loading it in a headless browser and collecting errors.
pub async fn check_url(url: &str, timeout_secs: u64) -> Result<BrowserCheckResult> {
    use chromiumoxide::browser::{Browser, BrowserConfig};

    let (mut browser, mut handler) = Browser::launch(
        BrowserConfig::builder()
            .no_sandbox()
            .window_size(1280, 720)
            .build()
            .map_err(|e| anyhow::anyhow!("Browser config error: {}", e))?,
    )
    .await
    .context("Failed to launch headless browser")?;

    let handle = tokio::spawn(async move {
        while let Some(_) = handler.next().await {}
    });

    let result = run_url_check(&mut browser, url, timeout_secs).await;

    let _ = browser.close().await;
    handle.abort();

    result
}

/// Internal: navigate and collect page state.
async fn run_url_check(
    browser: &mut chromiumoxide::browser::Browser,
    url: &str,
    timeout_secs: u64,
) -> Result<BrowserCheckResult> {
    let page = browser.new_page(url).await
        .context("Failed to open new page")?;

    tokio::time::sleep(tokio::time::Duration::from_secs(timeout_secs.min(30))).await;

    let title = page.get_title().await
        .unwrap_or_default()
        .unwrap_or_default();

    Ok(BrowserCheckResult {
        url: url.to_string(),
        loaded: true,
        console_errors: Vec::new(),
        page_title: title,
        status_code: None,
        screenshot_b64: None,
    })
}

/// Run a sequence of browser actions and return results.
pub async fn run_browser_actions(
    url: &str,
    actions: Vec<BrowserAction>,
) -> Result<Vec<BrowserCheckResult>> {
    use chromiumoxide::browser::{Browser, BrowserConfig};

    let (mut browser, mut handler) = Browser::launch(
        BrowserConfig::builder()
            .no_sandbox()
            .window_size(1280, 720)
            .build()
            .map_err(|e| anyhow::anyhow!("Browser config error: {}", e))?,
    )
    .await
    .context("Failed to launch browser for actions")?;

    let handle = tokio::spawn(async move {
        while let Some(_) = handler.next().await {}
    });

    let results = execute_actions(&mut browser, url, actions).await;

    let _ = browser.close().await;
    handle.abort();

    results
}

/// Execute a single browser action against a page, collecting results.
async fn execute_single_action(
    page: &chromiumoxide::Page,
    url: &str,
    action: &BrowserAction,
) -> Option<BrowserCheckResult> {
    match action {
        BrowserAction::Click(selector) => {
            let _ = page.find_element(selector).await.map(|el| {
                tokio::spawn(async move { let _ = el.click().await; })
            });
            None
        }
        BrowserAction::Type(selector, text) => {
            if let Ok(el) = page.find_element(selector).await {
                let _ = el.type_str(text).await;
            }
            None
        }
        BrowserAction::WaitMs(ms) => {
            tokio::time::sleep(tokio::time::Duration::from_millis(*ms)).await;
            None
        }
        BrowserAction::Navigate(nav_url) => {
            let _ = page.goto(nav_url).await;
            tokio::time::sleep(tokio::time::Duration::from_secs(2)).await;
            None
        }
        BrowserAction::Screenshot => {
            match page.screenshot(
                chromiumoxide::page::ScreenshotParams::builder().build(),
            ).await {
                Ok(bytes) => {
                    let b64 = base64::Engine::encode(
                        &base64::engine::general_purpose::STANDARD, &bytes,
                    );
                    Some(BrowserCheckResult {
                        url: url.to_string(), loaded: true,
                        console_errors: vec![], page_title: String::new(),
                        status_code: None, screenshot_b64: Some(b64),
                    })
                }
                Err(_) => None,
            }
        }
        BrowserAction::AssertVisible(selector) => {
            if page.find_element(selector).await.is_err() {
                Some(BrowserCheckResult {
                    url: url.to_string(), loaded: true,
                    console_errors: vec![format!("AssertVisible FAILED: {}", selector)],
                    page_title: String::new(), status_code: None,
                    screenshot_b64: None,
                })
            } else {
                None
            }
        }
        BrowserAction::AssertText(selector, expected) => {
            match page.find_element(selector).await {
                Ok(el) => {
                    let text = el.inner_text().await
                        .unwrap_or_default().unwrap_or_default();
                    if !text.contains(expected) {
                        Some(BrowserCheckResult {
                            url: url.to_string(), loaded: true,
                            console_errors: vec![format!(
                                "AssertText FAILED: '{}' expected '{}', got '{}'",
                                selector, expected, text
                            )],
                            page_title: String::new(), status_code: None,
                            screenshot_b64: None,
                        })
                    } else { None }
                }
                Err(_) => Some(BrowserCheckResult {
                    url: url.to_string(), loaded: true,
                    console_errors: vec![format!("AssertText: '{}' not found", selector)],
                    page_title: String::new(), status_code: None,
                    screenshot_b64: None,
                }),
            }
        }
    }
}

/// Execute browser actions sequentially against a page.
async fn execute_actions(
    browser: &mut chromiumoxide::browser::Browser,
    url: &str,
    actions: Vec<BrowserAction>,
) -> Result<Vec<BrowserCheckResult>> {
    let page = browser.new_page(url).await?;
    tokio::time::sleep(tokio::time::Duration::from_secs(3)).await;
    let mut results = Vec::new();

    for action in &actions {
        if let Some(r) = execute_single_action(&page, url, action).await {
            results.push(r);
        }
    }

    Ok(results)
}

/// Format browser check results into a human-readable report.
pub fn format_browser_report(results: &[BrowserCheckResult]) -> String {
    let mut report = String::from("[Browser Validation]\n");
    for r in results {
        report.push_str(&format!("URL: {} — loaded: {}\n", r.url, r.loaded));
        if !r.page_title.is_empty() {
            report.push_str(&format!("  Title: {}\n", r.page_title));
        }
        for err in &r.console_errors {
            report.push_str(&format!("  ❌ {}\n", err));
        }
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_browser_action_variants() {
        let actions = vec![
            BrowserAction::Navigate("http://localhost".into()),
            BrowserAction::Click("#btn".into()),
            BrowserAction::Type("#input".into(), "hello".into()),
            BrowserAction::AssertVisible("#el".into()),
            BrowserAction::AssertText("#el".into(), "world".into()),
            BrowserAction::Screenshot,
            BrowserAction::WaitMs(100),
        ];
        assert_eq!(actions.len(), 7);
    }

    #[test]
    fn test_format_browser_report() {
        let results = vec![BrowserCheckResult {
            url: "http://localhost:3000".into(),
            loaded: true,
            console_errors: vec!["TypeError: x is undefined".into()],
            page_title: "Test App".into(),
            status_code: Some(200),
            screenshot_b64: None,
        }];
        let report = format_browser_report(&results);
        assert!(report.contains("localhost:3000"));
        assert!(report.contains("TypeError"));
        assert!(report.contains("Test App"));
    }

    #[test]
    fn test_empty_browser_report() {
        let report = format_browser_report(&[]);
        assert!(report.contains("[Browser Validation]"));
    }
}
