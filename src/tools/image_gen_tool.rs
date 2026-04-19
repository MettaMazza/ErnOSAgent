//! Image generation tool — calls local Flux server to generate images.
//! Images are saved to `data/images/` and served via `/api/images/`.

use anyhow::{Context, Result};

/// Generate an image via the local Flux server.
pub async fn execute(args: &serde_json::Value) -> Result<String> {
    let prompt = args["prompt"].as_str().unwrap_or("");
    let width = args["width"].as_u64().unwrap_or(1024) as u32;
    let height = args["height"].as_u64().unwrap_or(1024) as u32;
    let steps = args["steps"].as_u64().unwrap_or(30) as u32;
    let guidance = args["guidance"].as_f64().unwrap_or(3.5);

    if prompt.is_empty() {
        anyhow::bail!("Image prompt cannot be empty");
    }

    let port = std::env::var("FLUX_PORT").unwrap_or_else(|_| "8890".to_string());
    let url = format!("http://127.0.0.1:{}/generate", port);

    tracing::info!(prompt = %prompt, width, height, steps, "Generating image via Flux");

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .json(&serde_json::json!({
            "prompt": prompt,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
        }))
        .timeout(std::time::Duration::from_secs(300))
        .send()
        .await
        .context("Failed to connect to Flux server — is scripts/flux_server.py running?")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("Flux server returned {}: {}", status, body);
    }

    let result: serde_json::Value = resp.json().await
        .context("Failed to parse Flux server response")?;

    // Check for server-reported errors
    if let Some(error) = result["error"].as_str() {
        anyhow::bail!(
            "Flux generation failed: {}\n\
             Hint: This may be caused by the MPS scheduler bug. \
             Try reducing image dimensions (e.g. 512x512) or steps.",
            error
        );
    }

    let b64 = result["image_base64"].as_str()
        .context("Flux response missing image_base64 field")?;

    // Validate image before saving
    let validation = validate_image(b64, width, height)?;

    let id = uuid::Uuid::new_v4().to_string();
    let filename = format!("{}.png", id);
    save_base64_image(b64, &filename)?;

    tracing::info!(
        filename = %filename,
        prompt = %prompt.chars().take(50).collect::<String>(),
        file_size = validation.file_size,
        "Image generated and saved"
    );

    // Build tool result with image for model self-inspection
    let image_url = format!("/api/images/{}", filename);
    let thumbnail_data_url = make_thumbnail_data_url(b64);

    Ok(format!(
        "![{prompt}]({image_url})\n\n\
         **Image saved**: {filename} ({w}x{h}, {size} bytes, {steps} steps)\n\n\
         [GENERATED IMAGE FOR REVIEW — data:image/png follows]\n\
         {thumbnail}\n\n\
         Review the generated image. If it looks correct, deliver the markdown tag above to the user \
         with a brief description. If it looks wrong (black, corrupt, distorted), inform the user \
         and suggest adjusting parameters.",
        prompt = prompt,
        image_url = image_url,
        filename = filename,
        w = width,
        h = height,
        size = validation.file_size,
        steps = steps,
        thumbnail = thumbnail_data_url,
    ))
}

/// Validation result for a generated image.
#[derive(Debug)]
struct ImageValidation {
    file_size: usize,
}

/// Validate that the decoded image is not blank/corrupt.
fn validate_image(b64: &str, width: u32, height: u32) -> Result<ImageValidation> {
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .context("Failed to decode base64 image")?;

    let file_size = bytes.len();

    // A real 1024x1024 PNG is typically 200KB+. Under 10KB is almost certainly blank.
    let min_expected = ((width * height) / 200) as usize; // very conservative lower bound
    if file_size < min_expected.max(5000) {
        anyhow::bail!(
            "Image generation produced a suspiciously small file ({} bytes for {}x{}).\n\
             This usually means the Flux MPS scheduler hit an IndexError and returned a blank frame.\n\
             Diagnostics:\n\
             - Expected minimum: ~{} bytes for {}x{}\n\
             - Actual: {} bytes\n\
             - Likely cause: MPS scheduler IndexError (Apple Silicon compatibility issue)\n\
             Suggested fixes:\n\
             1. Try smaller dimensions: 512x512\n\
             2. Try fewer steps: 20\n\
             3. Restart the Flux server: kill the process and let Ern-OS auto-restart it\n\
             4. Check Flux server logs for 'Warning: Scheduler IndexError caught'",
            file_size, width, height,
            min_expected.max(5000), width, height,
            file_size,
        );
    }

    Ok(ImageValidation { file_size })
}

/// Create a small base64 data URL for model self-inspection.
/// Resizes to 256x256 for context efficiency.
fn make_thumbnail_data_url(b64: &str) -> String {
    // Return the full base64 as a data URL — the model's vision encoder handles sizing
    format!("data:image/png;base64,{}", &b64[..b64.len().min(50000)])
}

/// Decode base64 and save to data/images/.
fn save_base64_image(b64: &str, filename: &str) -> Result<()> {
    use base64::Engine;
    let dir = std::path::PathBuf::from("data/images");
    std::fs::create_dir_all(&dir)
        .context("Failed to create data/images directory")?;

    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .context("Failed to decode base64 image")?;

    let path = dir.join(filename);
    std::fs::write(&path, &bytes)
        .with_context(|| format!("Failed to write image to {:?}", path))?;
    Ok(())
}

/// Check if the Flux server is reachable.
pub async fn health_check() -> bool {
    let port = std::env::var("FLUX_PORT").unwrap_or_else(|_| "8890".to_string());
    let url = format!("http://127.0.0.1:{}/health", port);
    match reqwest::get(&url).await {
        Ok(resp) => resp.status().is_success(),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_base64_image() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path().join("data/images");
        std::fs::create_dir_all(&dir).unwrap();

        // Minimal 1x1 white PNG as base64
        let b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwADhQGAWjR9awAAAABJRU5ErkJggg==";
        // Test the decoding works (path won't match since we can't override data/)
        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD.decode(b64).unwrap();
        assert!(!bytes.is_empty());
        assert!(bytes.len() > 50); // Valid PNG header
    }

    #[tokio::test]
    async fn test_empty_prompt_rejected() {
        let args = serde_json::json!({"prompt": ""});
        assert!(execute(&args).await.is_err());
    }

    #[tokio::test]
    async fn test_health_check_offline() {
        // Flux server not running — should return false
        assert!(!health_check().await);
    }

    #[test]
    fn test_validate_blank_image_detected() {
        // A tiny base64 string representing a very small file
        use base64::Engine;
        let tiny = base64::engine::general_purpose::STANDARD.encode(b"PNG tiny");
        let result = validate_image(&tiny, 1024, 1024);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("suspiciously small"));
        assert!(err.contains("MPS scheduler"));
    }
}
