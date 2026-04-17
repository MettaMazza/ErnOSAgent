// ErnOSAgent — Local-first AI agent with recursive self-improvement
// Created by @mettamazza (github.com/mettamazza)
//! Headless Chrome DOM tool suite

use std::sync::{Arc, Mutex, OnceLock};
use headless_chrome::{Browser, LaunchOptions};
use serde_json::Value;

static BROWSER: OnceLock<Arc<Mutex<Option<Arc<Browser>>>>> = OnceLock::new();

fn get_browser_lock() -> Arc<Mutex<Option<Arc<Browser>>>> {
    BROWSER.get_or_init(|| Arc::new(Mutex::new(None))).clone()
}

fn get_browser() -> Result<Arc<Browser>, String> {
    let browser_mux = get_browser_lock();
    let mut browser_lock = browser_mux.lock().map_err(|e| e.to_string())?;
    if let Some(b) = &*browser_lock {
        return Ok(b.clone());
    }

    // Launch non-headless so the user can literally see ErnOS pulling up Chrome
    let b = Browser::new(LaunchOptions {
        headless: false,
        window_size: Some((1280, 800)),
        ..Default::default()
    }).map_err(|e| format!("Failed to launch Chrome process: {}", e))?;
    
    let arc_b = Arc::new(b);
    *browser_lock = Some(arc_b.clone());
    Ok(arc_b)
}

pub fn register_tools(executor: &mut crate::tools::executor::ToolExecutor) {
    executor.register("browser_navigate", Box::new(|call| {
        let result = execute_browser_navigate(&call.arguments);
        crate::tools::schema::ToolResult {
            tool_call_id: call.id.clone(),
            name: call.name.clone(),
            output: match &result { Ok(s) => s.clone(), Err(_) => String::new() },
            success: result.is_ok(),
            error: result.err(),
        }
    }));
    executor.register("browser_click", Box::new(|call| {
        let result = execute_browser_click(&call.arguments);
        crate::tools::schema::ToolResult {
            tool_call_id: call.id.clone(),
            name: call.name.clone(),
            output: match &result { Ok(s) => s.clone(), Err(_) => String::new() },
            success: result.is_ok(),
            error: result.err(),
        }
    }));
    executor.register("browser_type", Box::new(|call| {
        let result = execute_browser_type(&call.arguments);
        crate::tools::schema::ToolResult {
            tool_call_id: call.id.clone(),
            name: call.name.clone(),
            output: match &result { Ok(s) => s.clone(), Err(_) => String::new() },
            success: result.is_ok(),
            error: result.err(),
        }
    }));
}

fn capture_and_save_screenshot(tab: &std::sync::Arc<headless_chrome::Tab>) -> Result<String, String> {
    let png_data = tab.capture_screenshot(
        headless_chrome::protocol::cdp::Page::CaptureScreenshotFormatOption::Png,
        None,
        None,
        true
    ).map_err(|e| format!("Screenshot failed: {}", e))?;
    
    let path = "/tmp/ernos_browser_state.png";
    std::fs::write(path, png_data).map_err(|e| format!("Save failed: {}", e))?;
    
    Ok(format!("\nMEDIA: {}\n", path))
}

pub fn execute_browser_navigate(args: &Value) -> Result<String, String> {
    let url = args.get("url").and_then(|v| v.as_str()).ok_or("Missing 'url'")?;
    
    let browser = get_browser()?;
    let tab = browser.new_tab().map_err(|e| format!("Failed to open tab: {}", e))?;
    
    tab.navigate_to(url).map_err(|e| format!("Navigation failed: {}", e))?;
    tab.wait_until_navigated().map_err(|e| format!("Wait failed: {}", e))?;
    
    let text = tab.evaluate("document.body.innerText", false)
        .map_err(|e| format!("DOM read failed: {}", e))?;
    
    let text_content = text.value.unwrap_or(serde_json::json!("")).as_str().unwrap_or("").to_string();
    let truncated: String = text_content.chars().take(8000).collect();
    
    let media_tag = capture_and_save_screenshot(&tab)?;
    
    Ok(format!("Navigated to {}.\n\nDOM PREVIEW:\n{}{}", url, truncated, media_tag))
}

pub fn execute_browser_click(args: &Value) -> Result<String, String> {
    let selector = args.get("selector").and_then(|v| v.as_str()).ok_or("Missing 'selector'")?;
    
    let browser = get_browser()?;
    let tabs = browser.get_tabs().lock().unwrap();
    let tab = tabs.last().ok_or("No active browser tabs found")?.clone();
    
    let element = tab.wait_for_element(selector).map_err(|e| format!("Timeout waiting for {}: {}", selector, e))?;
    element.click().map_err(|e| format!("Click failed: {}", e))?;
    
    // Wait for network/DOM response before snapping
    std::thread::sleep(std::time::Duration::from_millis(500));
    let media_tag = capture_and_save_screenshot(&tab)?;
    
    Ok(format!("Successfully clicked on element '{}'.{}", selector, media_tag))
}

pub fn execute_browser_type(args: &Value) -> Result<String, String> {
    let selector = args.get("selector").and_then(|v| v.as_str()).ok_or("Missing 'selector'")?;
    let text = args.get("text").and_then(|v| v.as_str()).ok_or("Missing 'text'")?;
    
    let browser = get_browser()?;
    let tabs = browser.get_tabs().lock().unwrap();
    let tab = tabs.last().ok_or("No active browser tabs found")?.clone();
    
    let element = tab.wait_for_element(selector).map_err(|e| format!("Timeout waiting for {}: {}", selector, e))?;
    element.click().map_err(|e| format!("Click before typing failed: {}", e))?;
    element.type_into(text).map_err(|e| format!("Type failed: {}", e))?;
    
    // Wait for DOM response before snapping
    std::thread::sleep(std::time::Duration::from_millis(500));
    let media_tag = capture_and_save_screenshot(&tab)?;
    
    Ok(format!("Successfully typed into element '{}'.{}", selector, media_tag))
}
