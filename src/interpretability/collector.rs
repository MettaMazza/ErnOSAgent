// Ern-OS — Activation collector — extract Gemma 4 residual stream activations for SAE training.
// Ported from ErnOSAgent with full parity.
//
// Uses the llama-server's /v1/embeddings endpoint pointed at the actual
// Gemma 4 GGUF model (NOT the separate embedding model).

use anyhow::{bail, Context, Result};
use std::path::Path;
use std::time::Instant;

/// Manages activation collection from the Gemma 4 model.
pub struct ActivationCollector {
    client: reqwest::Client,
    /// Base URL of the Gemma 4 embedding server
    embed_url: String,
    /// Auto-detected activation dimension
    pub activation_dim: Option<usize>,
}

impl ActivationCollector {
    /// Create a new collector pointing at a Gemma 4 embedding instance.
    pub fn new(embed_url: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            embed_url: embed_url.trim_end_matches('/').to_string(),
            activation_dim: None,
        }
    }

    /// Wait for the embedding server to become healthy.
    pub async fn wait_for_health(&self, timeout_secs: u64) -> Result<()> {
        let url = format!("{}/health", self.embed_url);
        let start = Instant::now();
        loop {
            if start.elapsed().as_secs() > timeout_secs {
                bail!(
                    "Embedding server at {} not healthy after {}s",
                    self.embed_url,
                    timeout_secs
                );
            }
            match self.client.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => return Ok(()),
                _ => tokio::time::sleep(tokio::time::Duration::from_millis(500)).await,
            }
        }
    }

    /// Extract activation vector for a single text.
    pub async fn extract_one(&mut self, text: &str) -> Result<Vec<f32>> {
        let url = format!("{}/v1/embeddings", self.embed_url);
        let body = serde_json::json!({
            "input": text,
            "encoding_format": "float",
        });

        let resp = self.client.post(&url)
            .json(&body)
            .send()
            .await
            .context("Failed to call embedding endpoint")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error_body = resp.text().await.unwrap_or_default();
            bail!("Embedding endpoint error {}: {}", status, error_body);
        }

        let parsed: serde_json::Value = resp.json().await
            .context("Failed to parse embedding response")?;

        let embedding = parsed
            .get("data")
            .and_then(|d| d.as_array())
            .and_then(|arr| arr.first())
            .and_then(|item| item.get("embedding"))
            .and_then(|e| e.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect::<Vec<f32>>()
            })
            .context("Missing embedding data in response")?;

        if embedding.is_empty() {
            bail!("Embedding server returned empty vector");
        }

        // Auto-detect dimension from first result
        if self.activation_dim.is_none() {
            self.activation_dim = Some(embedding.len());
            tracing::info!(
                dim = embedding.len(),
                "Auto-detected activation dimension from Gemma 4"
            );
        }

        Ok(embedding)
    }

    /// Collect activations for a batch of prompts with progress tracking.
    pub async fn collect_batch(
        &mut self,
        prompts: &[String],
        log_every: usize,
    ) -> Result<Vec<Vec<f32>>> {
        let total = prompts.len();
        let mut activations = Vec::with_capacity(total);
        let start = Instant::now();
        let mut errors = 0u64;

        tracing::info!(
            total_prompts = total,
            "Starting activation collection from Gemma 4"
        );

        for (i, prompt) in prompts.iter().enumerate() {
            match self.extract_one(prompt).await {
                Ok(act) => activations.push(act),
                Err(e) => {
                    errors += 1;
                    if errors <= 5 {
                        tracing::warn!(
                            error = %e,
                            prompt_idx = i,
                            "Failed to extract activation, skipping"
                        );
                    }
                    continue;
                }
            }

            if (i + 1) % log_every == 0 || i + 1 == total {
                let elapsed = start.elapsed();
                let rate = (i + 1) as f64 / elapsed.as_secs_f64();
                let remaining = (total - i - 1) as f64 / rate;
                let eta = std::time::Duration::from_secs_f64(remaining);

                tracing::info!(
                    collected = activations.len(),
                    total = total,
                    errors = errors,
                    rate_per_sec = format!("{:.1}", rate),
                    elapsed_secs = elapsed.as_secs(),
                    eta_secs = eta.as_secs(),
                    eta_human = format_eta(eta),
                    progress_pct = format!("{:.1}%", (i + 1) as f64 / total as f64 * 100.0),
                    "Activation collection progress"
                );
            }
        }

        tracing::info!(
            collected = activations.len(),
            errors = errors,
            elapsed = format_eta(start.elapsed()),
            dim = self.activation_dim.unwrap_or(0),
            "Activation collection complete"
        );

        Ok(activations)
    }

    /// Save collected activations to disk as raw f32 binary (memory-mappable).
    pub fn save_activations(
        activations: &[Vec<f32>],
        path: &Path,
        dim: usize,
    ) -> Result<()> {
        std::fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))?;

        // Header: [num_samples: u64, dim: u64]
        let num = activations.len() as u64;
        let dim_u64 = dim as u64;
        let mut data: Vec<u8> = Vec::new();
        data.extend_from_slice(&num.to_le_bytes());
        data.extend_from_slice(&dim_u64.to_le_bytes());

        // Body: flat f32 array
        for act in activations {
            for &val in act {
                data.extend_from_slice(&val.to_le_bytes());
            }
        }

        std::fs::write(path, &data)?;

        tracing::info!(
            num_samples = activations.len(),
            dim = dim,
            size_mb = format!("{:.1}", data.len() as f64 / 1_000_000.0),
            path = %path.display(),
            "Activations saved to disk"
        );
        Ok(())
    }

    /// Load activations from a previously saved binary file.
    pub fn load_activations(path: &Path) -> Result<(Vec<Vec<f32>>, usize)> {
        let data = std::fs::read(path)
            .with_context(|| format!("Failed to read activations: {}", path.display()))?;

        if data.len() < 16 {
            bail!("Activation file too small");
        }

        let num = u64::from_le_bytes(data[0..8].try_into().expect("header byte alignment")) as usize;
        let dim = u64::from_le_bytes(data[8..16].try_into().expect("header byte alignment")) as usize;

        let expected_size = 16 + num * dim * 4;
        if data.len() < expected_size {
            bail!(
                "Activation file truncated: expected {} bytes, got {}",
                expected_size,
                data.len()
            );
        }

        let mut activations = Vec::with_capacity(num);
        let float_data = &data[16..];
        for i in 0..num {
            let start = i * dim * 4;
            let act: Vec<f32> = (0..dim)
                .map(|j| {
                    let offset = start + j * 4;
                    f32::from_le_bytes([
                        float_data[offset],
                        float_data[offset + 1],
                        float_data[offset + 2],
                        float_data[offset + 3],
                    ])
                })
                .collect();
            activations.push(act);
        }

        tracing::info!(
            num_samples = num,
            dim = dim,
            size_mb = format!("{:.1}", data.len() as f64 / 1_000_000.0),
            path = %path.display(),
            "Activations loaded from disk"
        );
        Ok((activations, dim))
    }
}

// Corpus building (build_corpus, built_in_diversity_prompts, format_eta)
// moved to crate::interpretability::corpus.
pub use super::corpus::{build_corpus, format_eta};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_load_activations_roundtrip() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("test_activations.bin");

        let activations = vec![
            vec![1.0f32, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        ActivationCollector::save_activations(&activations, &path, 3).unwrap();
        let (loaded, dim) = ActivationCollector::load_activations(&path).unwrap();

        assert_eq!(dim, 3);
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(loaded[2], vec![7.0, 8.0, 9.0]);
    }
}

