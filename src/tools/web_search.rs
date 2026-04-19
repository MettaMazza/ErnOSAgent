// Ern-OS — 8-Tier Waterfall Web Search
//
// Ported from ErnOSAgent. Each tier is tried in order; the first to return
// results wins. API-key tiers are skipped when no key is set.
//
//! Tier 1: Brave Search API (BRAVE_API_KEY)
//! Tier 2: Serper.dev Google SERP (SERPER_API_KEY)
//! Tier 3: Tavily AI search (TAVILY_API_KEY)
//! Tier 4: SerpAPI multi-engine (SERPAPI_API_KEY)
//! Tier 5: DuckDuckGo HTML scrape (with CAPTCHA detection)
//! Tier 6: Google Web scrape (with CAPTCHA detection)
//! Tier 7: Wikipedia API (structured factual fallback)
//! Tier 8: Google News RSS (news fallback, no CAPTCHA)

use anyhow::Result;
use super::search_providers as p;

/// 8-tier waterfall web search. Cascades through providers until one returns results.
pub async fn search(query: &str) -> Result<String> {
    if query.is_empty() { anyhow::bail!("Empty search query"); }
    let client = build_client()?;

    // Tier 1: Brave
    let k = resolve_env_key("BRAVE_API_KEY", &["BRAVE_SEARCH_API_KEY"]);
    if let Some(r) = try_keyed(&client, query, 1, "brave", &k, p::brave_search(&client, query, &k)).await { return Ok(r); }

    // Tier 2: Serper
    let k = resolve_env_key("SERPER_API_KEY", &[]);
    if let Some(r) = try_keyed(&client, query, 2, "serper", &k, p::serper_search(&client, query, &k)).await { return Ok(r); }

    // Tier 3: Tavily
    let k = resolve_env_key("TAVILY_API_KEY", &[]);
    if let Some(r) = try_keyed(&client, query, 3, "tavily", &k, p::tavily_search(&client, query, &k)).await { return Ok(r); }

    // Tier 4: SerpAPI
    let k = resolve_env_key("SERPAPI_API_KEY", &[]);
    if let Some(r) = try_keyed(&client, query, 4, "serpapi", &k, p::serpapi_search(&client, query, &k)).await { return Ok(r); }

    // Tier 5: DuckDuckGo (free)
    if let Some(r) = try_free(query, 5, "duckduckgo", p::duckduckgo_search(&client, query)).await { return Ok(r); }

    // Tier 6: Google scrape (free)
    if let Some(r) = try_free(query, 6, "google_scrape", p::google_web_scrape(&client, query)).await { return Ok(r); }

    // Tier 7: Wikipedia (free)
    if let Some(r) = try_free(query, 7, "wikipedia", p::wikipedia_search(&client, query)).await { return Ok(r); }

    // Tier 8: Google News RSS (free)
    if let Some(r) = try_free(query, 8, "google_rss", p::google_news_rss(&client, query)).await { return Ok(r); }

    Ok(format!(
        "[TOOL FAILURE: web_search]\n\
         Query: \"{}\"\n\
         Status: All 8 search tiers exhausted.\n\
         Action: Respond with your own knowledge.",
        query
    ))
}

/// Try a keyed tier — skip if no key is set.
async fn try_keyed(
    _client: &reqwest::Client, _query: &str, tier: u8, provider: &str,
    key: &str, fut: impl std::future::Future<Output = Result<String, String>>,
) -> Option<String> {
    if key.is_empty() {
        tracing::debug!("web_search: Tier {} skipped — no key", tier);
        return None;
    }
    tracing::info!("web_search: Tier {} — trying {}…", tier, provider);
    match fut.await {
        Ok(r) => { tracing::info!(tier, provider, "web_search: ✅ success"); Some(r) }
        Err(e) => { tracing::warn!(tier, error = %e, "web_search: ⚠️ {} failed", provider); None }
    }
}

/// Try a free (no-key) tier.
async fn try_free(
    _query: &str, tier: u8, provider: &str,
    fut: impl std::future::Future<Output = Result<String, String>>,
) -> Option<String> {
    tracing::info!("web_search: Tier {} — trying {}…", tier, provider);
    match fut.await {
        Ok(r) => { tracing::info!(tier, provider, "web_search: ✅ success"); Some(r) }
        Err(e) => { tracing::warn!(tier, error = %e, "web_search: ⚠️ {} failed", provider); None }
    }
}

/// Visit a URL directly and extract text content.
pub async fn visit(url: &str) -> Result<String> {
    if url.is_empty() { anyhow::bail!("Missing URL"); }
    if !url.starts_with("http://") && !url.starts_with("https://") {
        anyhow::bail!("URL must start with http:// or https://");
    }
    let client = build_client()?;
    let resp = client.get(url).send().await?;
    if !resp.status().is_success() { anyhow::bail!("HTTP Error: {}", resp.status()); }

    let content_type = resp.headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok()).unwrap_or("").to_lowercase();
    if !content_type.starts_with("text/")
        && !content_type.starts_with("application/json")
        && !content_type.starts_with("application/xml")
        && !content_type.starts_with("application/xhtml")
    {
        anyhow::bail!("URL returns non-text content ({}). Cannot read binary files as text.", content_type);
    }

    let html = resp.text().await?;
    let text = p::strip_html(&html);
    let cleaned: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    Ok(format!("--- WEBPAGE CONTENT ({}) ---\n{}", url, cleaned))
}

// ── Environment Key Resolution ──

fn resolve_env_key(primary: &str, aliases: &[&str]) -> String {
    if let Ok(key) = std::env::var(primary) { if !key.is_empty() { return key; } }
    for alias in aliases {
        if let Ok(key) = std::env::var(alias) { if !key.is_empty() { return key; } }
    }
    for env_path in &[".env", "../.env"] {
        if let Ok(content) = std::fs::read_to_string(env_path) {
            for line in content.lines() {
                let check_keys: Vec<&str> = std::iter::once(primary).chain(aliases.iter().copied()).collect();
                for key_name in &check_keys {
                    let prefix = format!("{}=", key_name);
                    if line.starts_with(&prefix) {
                        let parts: Vec<&str> = line.splitn(2, '=').collect();
                        if parts.len() == 2 {
                            let key = parts[1].trim_matches('"').trim_matches('\'').to_string();
                            if !key.is_empty() { return key; }
                        }
                    }
                }
            }
        }
    }
    String::new()
}

fn build_client() -> Result<reqwest::Client> {
    Ok(reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(15))
        .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")
        .build()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use p::*;

    #[test]
    fn test_empty_query() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        assert!(rt.block_on(search("")).is_err());
    }

    #[test]
    fn test_strip_html() {
        assert_eq!(strip_html("<b>bold</b> text").trim(), "bold  text");
        assert_eq!(strip_html("no tags"), "no tags");
    }

    #[test]
    fn test_strip_html_script() {
        let html = "<p>Before</p><script>alert('x')</script><p>After</p>";
        let text = strip_html(html);
        assert!(text.contains("Before"));
        assert!(text.contains("After"));
        assert!(!text.contains("alert"));
    }

    #[test]
    fn test_strip_html_entities() {
        assert!(strip_html("&amp; &lt; &gt; &quot;").contains("&"));
    }

    #[test]
    fn test_resolve_env_key_empty() {
        assert!(resolve_env_key("NONEXISTENT_KEY_12345", &[]).is_empty());
    }

    #[test]
    fn test_xml_tag_extraction() {
        let xml = "<title>Hello World</title><link>https://example.com</link>";
        assert_eq!(xml_tag_content(xml, "title"), "Hello World");
        assert_eq!(xml_tag_content(xml, "link"), "https://example.com");
        assert_eq!(xml_tag_content(xml, "missing"), "");
    }

    #[test]
    fn test_xml_tag_cdata() {
        assert_eq!(xml_tag_content("<title><![CDATA[Test Title]]></title>", "title"), "Test Title");
    }

    #[test]
    fn test_visit_empty() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        assert!(rt.block_on(visit("")).is_err());
    }

    #[test]
    fn test_visit_invalid_url() {
        let rt = tokio::runtime::Runtime::new().unwrap();
        assert!(rt.block_on(visit("not-a-url")).unwrap_err().to_string().contains("http"));
    }
}
