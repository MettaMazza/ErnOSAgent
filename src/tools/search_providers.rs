//! Search provider implementations — one function per tier.
//! All providers share the same signature: `(client, query, [api_key]) -> Result<String, String>`.

/// Strip HTML tags from content. Fully char-based to handle multi-byte UTF-8 safely.
pub fn strip_html(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;

    let chars: Vec<char> = html.chars().collect();
    let lower_chars: Vec<char> = html.to_lowercase().chars().collect();

    let script_open: Vec<char> = "<script".chars().collect();
    let script_close: Vec<char> = "</script>".chars().collect();
    let style_open: Vec<char> = "<style".chars().collect();
    let style_close: Vec<char> = "</style>".chars().collect();

    fn starts_with_at(hay: &[char], needle: &[char], pos: usize) -> bool {
        if pos + needle.len() > hay.len() { return false; }
        hay[pos..pos + needle.len()] == *needle
    }

    let mut i = 0;
    while i < chars.len() {
        if in_script {
            if starts_with_at(&lower_chars, &script_close, i) { in_script = false; i += script_close.len(); continue; }
            i += 1; continue;
        }
        if in_style {
            if starts_with_at(&lower_chars, &style_close, i) { in_style = false; i += style_close.len(); continue; }
            i += 1; continue;
        }
        if chars[i] == '<' {
            if starts_with_at(&lower_chars, &script_open, i) { in_script = true; in_tag = true; }
            else if starts_with_at(&lower_chars, &style_open, i) { in_style = true; in_tag = true; }
            else { in_tag = true; }
        } else if chars[i] == '>' {
            in_tag = false;
            result.push(' ');
        } else if !in_tag {
            result.push(chars[i]);
        }
        i += 1;
    }

    result
        .replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        .replace("&quot;", "\"").replace("&#39;", "'").replace("&nbsp;", " ")
}

/// Extract content between XML tags (lightweight, no xml crate).
pub fn xml_tag_content(text: &str, tag: &str) -> String {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    if let Some(start) = text.find(&open) {
        let after = start + open.len();
        if let Some(end) = text[after..].find(&close) {
            let content = &text[after..after + end];
            return content.trim()
                .strip_prefix("<![CDATA[").and_then(|s| s.strip_suffix("]]>"))
                .unwrap_or(content).to_string();
        }
    }
    String::new()
}

// ── Tier 1: Brave Search API ──

pub async fn brave_search(client: &reqwest::Client, query: &str, api_key: &str) -> Result<String, String> {
    let url = format!(
        "https://api.search.brave.com/res/v1/web/search?q={}&count=10&text_decorations=false",
        urlencoding::encode(query)
    );
    let resp = client.get(&url)
        .header("Accept", "application/json").header("Accept-Encoding", "gzip")
        .header("X-Subscription-Token", api_key)
        .send().await.map_err(|e| format!("Brave API request failed: {}", e))?;

    if !resp.status().is_success() { return Err(format!("Brave API status: {}", resp.status())); }

    let body: serde_json::Value = resp.json().await
        .map_err(|e| format!("Failed to parse Brave JSON: {}", e))?;

    let mut results = Vec::new();
    if let Some(web) = body.get("web").and_then(|w| w.get("results")) {
        if let Some(items) = web.as_array() {
            for item in items.iter().take(8) {
                let title = item.get("title").and_then(|t| t.as_str()).unwrap_or("Untitled");
                let desc = item.get("description").and_then(|d| d.as_str()).unwrap_or("");
                let url = item.get("url").and_then(|u| u.as_str()).unwrap_or("");
                results.push(format!("• {}\n  {}\n  {}", title, desc, url));
            }
        }
    }
    if results.is_empty() { return Err("Brave returned no results".to_string()); }
    Ok(format!("--- BRAVE SEARCH RESULTS for '{}' ---\n{}", query, results.join("\n\n")))
}

// ── Tier 2: Serper.dev Google SERP ──

pub async fn serper_search(client: &reqwest::Client, query: &str, api_key: &str) -> Result<String, String> {
    let body = serde_json::json!({ "q": query, "num": 10 });
    let resp = client.post("https://google.serper.dev/search")
        .header("X-API-KEY", api_key).header("Content-Type", "application/json")
        .json(&body).send().await.map_err(|e| format!("Serper request failed: {}", e))?;

    if !resp.status().is_success() { return Err(format!("Serper API status: {}", resp.status())); }

    let data: serde_json::Value = resp.json().await
        .map_err(|e| format!("Serper JSON parse failed: {}", e))?;
    let mut results = Vec::new();

    if let Some(kg) = data.get("knowledgeGraph") {
        let title = kg.get("title").and_then(|t| t.as_str()).unwrap_or("");
        let desc = kg.get("description").and_then(|d| d.as_str()).unwrap_or("");
        if !title.is_empty() { results.push(format!("📋 Knowledge Graph: {} — {}", title, desc)); }
    }
    if let Some(ab) = data.get("answerBox") {
        let answer = ab.get("answer").or(ab.get("snippet")).and_then(|a| a.as_str()).unwrap_or("");
        if !answer.is_empty() { results.push(format!("💡 Answer Box: {}", answer)); }
    }
    if let Some(organic) = data.get("organic").and_then(|o| o.as_array()) {
        for item in organic.iter().take(8) {
            let title = item.get("title").and_then(|t| t.as_str()).unwrap_or("Untitled");
            let snippet = item.get("snippet").and_then(|s| s.as_str()).unwrap_or("");
            let link = item.get("link").and_then(|l| l.as_str()).unwrap_or("");
            results.push(format!("• {}\n  {}\n  {}", title, snippet, link));
        }
    }
    if results.is_empty() { return Err("Serper returned no results".to_string()); }
    Ok(format!("--- SERPER SEARCH RESULTS for '{}' ---\n{}", query, results.join("\n\n")))
}

// ── Tier 3: Tavily AI Search ──

pub async fn tavily_search(client: &reqwest::Client, query: &str, api_key: &str) -> Result<String, String> {
    let body = serde_json::json!({
        "api_key": api_key, "query": query, "search_depth": "basic",
        "max_results": 8, "include_answer": true,
    });
    let resp = client.post("https://api.tavily.com/search")
        .header("Content-Type", "application/json").json(&body)
        .send().await.map_err(|e| format!("Tavily request failed: {}", e))?;

    if !resp.status().is_success() { return Err(format!("Tavily API status: {}", resp.status())); }

    let data: serde_json::Value = resp.json().await
        .map_err(|e| format!("Tavily JSON parse failed: {}", e))?;
    let mut results = Vec::new();

    if let Some(answer) = data.get("answer").and_then(|a| a.as_str()) {
        if !answer.is_empty() { results.push(format!("💡 Tavily Answer: {}", answer)); }
    }
    if let Some(items) = data.get("results").and_then(|r| r.as_array()) {
        for item in items.iter().take(8) {
            let title = item.get("title").and_then(|t| t.as_str()).unwrap_or("Untitled");
            let content = item.get("content").and_then(|c| c.as_str()).unwrap_or("");
            let url = item.get("url").and_then(|u| u.as_str()).unwrap_or("");
            results.push(format!("• {}\n  {}\n  {}", title, content, url));
        }
    }
    if results.is_empty() { return Err("Tavily returned no results".to_string()); }
    Ok(format!("--- TAVILY SEARCH RESULTS for '{}' ---\n{}", query, results.join("\n\n")))
}

// ── Tier 4: SerpAPI Multi-Engine ──

pub async fn serpapi_search(client: &reqwest::Client, query: &str, api_key: &str) -> Result<String, String> {
    let url = format!(
        "https://serpapi.com/search.json?q={}&api_key={}&engine=google&num=10",
        urlencoding::encode(query), urlencoding::encode(api_key)
    );
    let resp = client.get(&url).send().await.map_err(|e| format!("SerpAPI request failed: {}", e))?;
    if !resp.status().is_success() { return Err(format!("SerpAPI status: {}", resp.status())); }

    let data: serde_json::Value = resp.json().await
        .map_err(|e| format!("SerpAPI JSON parse failed: {}", e))?;
    let mut results = Vec::new();

    if let Some(ab) = data.get("answer_box") {
        let answer = ab.get("answer").or(ab.get("snippet")).and_then(|a| a.as_str()).unwrap_or("");
        if !answer.is_empty() { results.push(format!("💡 Answer: {}", answer)); }
    }
    if let Some(kg) = data.get("knowledge_graph") {
        let title = kg.get("title").and_then(|t| t.as_str()).unwrap_or("");
        let desc = kg.get("description").and_then(|d| d.as_str()).unwrap_or("");
        if !title.is_empty() { results.push(format!("📋 {}: {}", title, desc)); }
    }
    if let Some(sports) = data.get("sports_results") {
        if let Some(games) = sports.get("games").and_then(|g| g.as_array()) {
            for game in games.iter().take(3) {
                let teams_str = game.get("teams").and_then(|t| t.as_array()).map(|t| {
                    t.iter().filter_map(|team| {
                        let name = team.get("name").and_then(|n| n.as_str()).unwrap_or("");
                        let score = team.get("score").and_then(|s| s.as_str()).unwrap_or("");
                        if !name.is_empty() { Some(format!("{} {}", name, score)) } else { None }
                    }).collect::<Vec<_>>().join(" vs ")
                }).unwrap_or_default();
                let status = game.get("status").and_then(|s| s.as_str()).unwrap_or("");
                let date = game.get("date").and_then(|d| d.as_str()).unwrap_or("");
                if !teams_str.is_empty() { results.push(format!("  {} | {} | {}", teams_str, date, status)); }
            }
        }
    }
    if let Some(organic) = data.get("organic_results").and_then(|o| o.as_array()) {
        for item in organic.iter().take(8) {
            let title = item.get("title").and_then(|t| t.as_str()).unwrap_or("Untitled");
            let snippet = item.get("snippet").and_then(|s| s.as_str()).unwrap_or("");
            let link = item.get("link").and_then(|l| l.as_str()).unwrap_or("");
            results.push(format!("• {}\n  {}\n  {}", title, snippet, link));
        }
    }
    if results.is_empty() { return Err("SerpAPI returned no results".to_string()); }
    Ok(format!("--- SERPAPI SEARCH RESULTS for '{}' ---\n{}", query, results.join("\n\n")))
}

// ── Tier 5: DuckDuckGo HTML ──

pub async fn duckduckgo_search(client: &reqwest::Client, query: &str) -> Result<String, String> {
    let url = format!("https://html.duckduckgo.com/html/?q={}", urlencoding::encode(query));
    let resp = client.get(&url).send().await.map_err(|e| format!("DDG request failed: {}", e))?;
    let status = resp.status();
    if status.as_u16() == 202 || status.as_u16() == 403 {
        return Err(format!("DDG bot-detection status: {}", status));
    }
    let html = resp.text().await.map_err(|e| format!("Failed to read DDG response: {}", e))?;
    if html.contains("anomaly.js") || html.contains("challenge-form") {
        return Err("DDG returned CAPTCHA/bot-detection page".to_string());
    }
    let text = strip_html(&html);
    let word_count = text.split_whitespace().count();
    if word_count < 50 { return Err(format!("DDG too little content ({} words — likely captcha)", word_count)); }
    let cleaned: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    Ok(format!("--- DDG SEARCH RESULTS for '{}' ---\n{}", query, cleaned))
}

// ── Tier 6: Google Web Scrape ──

pub async fn google_web_scrape(client: &reqwest::Client, query: &str) -> Result<String, String> {
    let url = format!("https://www.google.com/search?q={}&hl=en&num=10", urlencoding::encode(query));
    let resp = client.get(&url).send().await.map_err(|e| format!("Google scrape request failed: {}", e))?;
    let status = resp.status();
    if status.as_u16() == 429 || status.as_u16() == 403 || status.as_u16() == 503 {
        return Err(format!("Google bot-detection status: {}", status));
    }
    let html = resp.text().await.map_err(|e| format!("Failed to read Google response: {}", e))?;
    if html.contains("unusual traffic") || html.contains("captcha") || html.contains("recaptcha") {
        return Err("Google returned CAPTCHA page".to_string());
    }
    let text = strip_html(&html);
    let word_count = text.split_whitespace().count();
    if word_count < 80 { return Err(format!("Google too little content ({} words — likely blocked)", word_count)); }
    let cleaned: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    Ok(format!("--- GOOGLE SEARCH RESULTS for '{}' ---\n{}", query, cleaned))
}

// ── Tier 7: Wikipedia API ──

pub async fn wikipedia_search(client: &reqwest::Client, query: &str) -> Result<String, String> {
    let search_url = format!(
        "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={}&srlimit=5&format=json",
        urlencoding::encode(query)
    );
    let resp = client.get(&search_url).send().await.map_err(|e| format!("Wikipedia search failed: {}", e))?;
    if !resp.status().is_success() { return Err(format!("Wikipedia API status: {}", resp.status())); }

    let data: serde_json::Value = resp.json().await
        .map_err(|e| format!("Wikipedia JSON failed: {}", e))?;
    let mut results = Vec::new();

    if let Some(items) = data.pointer("/query/search").and_then(|s| s.as_array()) {
        for item in items.iter().take(5) {
            let title = item.get("title").and_then(|t| t.as_str()).unwrap_or("");
            let snippet = item.get("snippet").and_then(|s| s.as_str()).unwrap_or("");
            let clean_snippet = strip_html(snippet);
            let url = format!("https://en.wikipedia.org/wiki/{}", urlencoding::encode(title));
            results.push(format!("• {}\n  {}\n  {}", title, clean_snippet.trim(), url));
        }
    }
    if results.is_empty() { return Err("Wikipedia returned no results".to_string()); }

    // Get first article extract
    if let Some(first_title) = data.pointer("/query/search/0/title").and_then(|t| t.as_str()) {
        let extract_url = format!(
            "https://en.wikipedia.org/w/api.php?action=query&titles={}&prop=extracts&exintro=true&explaintext=true&format=json",
            urlencoding::encode(first_title)
        );
        if let Ok(resp) = client.get(&extract_url).send().await {
            if let Ok(extract_data) = resp.json::<serde_json::Value>().await {
                if let Some(pages) = extract_data.pointer("/query/pages").and_then(|p| p.as_object()) {
                    for (_id, page) in pages {
                        if let Some(extract) = page.get("extract").and_then(|e| e.as_str()) {
                            if !extract.is_empty() {
                                results.insert(0, format!("📋 Wikipedia Summary: {}\n{}", first_title, extract));
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(format!("--- WIKIPEDIA RESULTS for '{}' ---\n{}", query, results.join("\n\n")))
}

// ── Tier 8: Google News RSS ──

pub async fn google_news_rss(client: &reqwest::Client, query: &str) -> Result<String, String> {
    let url = format!(
        "https://news.google.com/rss/search?q={}&hl=en-GB&gl=GB&ceid=GB:en",
        urlencoding::encode(query)
    );
    let resp = client.get(&url).send().await.map_err(|e| format!("Google RSS request failed: {}", e))?;
    if !resp.status().is_success() { return Err(format!("Google RSS status: {}", resp.status())); }

    let xml = resp.text().await.map_err(|e| format!("Failed to read RSS: {}", e))?;
    let mut items: Vec<String> = Vec::new();
    for chunk in xml.split("<item>").skip(1) {
        let title = xml_tag_content(chunk, "title");
        let description = xml_tag_content(chunk, "description");
        let link = xml_tag_content(chunk, "link");
        let pubdate = xml_tag_content(chunk, "pubDate");
        if !title.is_empty() {
            items.push(format!("• {}\n  {}\n  {} | {}", title, description, link, pubdate));
        }
        if items.len() >= 8 { break; }
    }
    if items.is_empty() { return Err("Google News RSS returned no items".to_string()); }
    Ok(format!("--- GOOGLE NEWS RSS for '{}' ---\n{}", query, items.join("\n\n")))
}
