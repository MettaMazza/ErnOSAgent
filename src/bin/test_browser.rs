use anyhow::Result;
use futures_util::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    let chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
    
    println!("Testing chromiumoxide with Chrome at: {}", chrome_path);
    println!("Chrome exists: {}", std::path::Path::new(chrome_path).exists());
    
    let user_data_dir = std::env::temp_dir()
        .join(format!("ern-os-chrome-test-{}", std::process::id()));
    std::fs::create_dir_all(&user_data_dir).ok();
    println!("User data dir: {}", user_data_dir.display());
    
    let config = chromiumoxide::BrowserConfig::builder()
        .chrome_executable(chrome_path)
        .user_data_dir(&user_data_dir)
        .arg("--headless=new")
        .arg("--disable-gpu")
        .arg("--no-sandbox")
        .arg("--disable-dev-shm-usage")
        .arg("--no-first-run")
        .arg("--disable-extensions")
        .build()
        .map_err(|e| anyhow::anyhow!("Config build error: {}", e))?;
    
    println!("Config built successfully");
    
    match chromiumoxide::Browser::launch(config).await {
        Ok((browser, mut handler)) => {
            println!("Browser launched successfully!");
            let handle = tokio::spawn(async move {
                while handler.next().await.is_some() {}
            });
            
            match browser.new_page("https://httpbin.org/html").await {
                Ok(page) => {
                    println!("Page opened!");
                    let title = page.get_title().await.unwrap_or_default().unwrap_or_default();
                    println!("Title: {}", title);
                    page.close().await.ok();
                }
                Err(e) => println!("Page error: {}", e),
            }
            
            drop(browser);
            handle.abort();
        }
        Err(e) => {
            println!("Launch FAILED: {:#}", e);
        }
    }
    
    // Cleanup
    std::fs::remove_dir_all(&user_data_dir).ok();
    
    Ok(())
}
