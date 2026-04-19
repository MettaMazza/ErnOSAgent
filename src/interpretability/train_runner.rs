// Ern-OS — SAE Training Runner — orchestrates the full collect → train → export pipeline.
// Ported from ErnOSAgent with full parity.
//
// Manages the complete lifecycle:
// 1. Start Gemma 4 in embedding mode (or connect to existing)
// 2. Collect activation vectors from diverse corpus
// 3. Train JumpReLU SAE on Metal with ETA tracking
// 4. Save checkpoints and final weights as safetensors
// 5. Can resume from checkpoint if interrupted

use crate::interpretability::collector::{self, ActivationCollector, format_eta};
use crate::interpretability::trainer::{SaeTrainer, TrainConfig};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;
use tokio::process::Command;
use std::process::Stdio;

/// Training run configuration.
#[derive(Debug, Clone)]
pub struct TrainingRunConfig {
    /// Path to the Gemma 4 GGUF model file
    pub model_path: String,
    /// Path to llama-server binary
    pub server_binary: String,
    /// Port for the embedding server
    pub embed_port: u16,
    /// GPU layers (-1 = all)
    pub n_gpu_layers: i32,
    /// Data directory for saving activations, checkpoints, and final weights
    pub data_dir: PathBuf,
    /// SAE training config
    pub train_config: TrainConfig,
    /// Minimum activation samples before training starts
    pub min_samples: usize,
    /// Whether to skip collection if activations are already saved
    pub resume_collection: bool,
}

impl Default for TrainingRunConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            server_binary: "llama-server".to_string(),
            embed_port: 8082,
            n_gpu_layers: -1,
            data_dir: PathBuf::new(),
            train_config: TrainConfig::default(),
            min_samples: 10_000,
            resume_collection: true,
        }
    }
}

/// Progress update for the training run.
#[derive(Debug, Clone, serde::Serialize)]
pub struct TrainingProgress {
    pub phase: String,
    pub step: usize,
    pub total_steps: usize,
    pub progress_pct: f64,
    pub elapsed_secs: u64,
    pub eta_secs: u64,
    pub eta_human: String,
    pub metrics: serde_json::Value,
}

/// Run the complete SAE training pipeline.
///
/// This is the main entry point. Call with the model config from AppConfig.
pub async fn run_sae_training(config: TrainingRunConfig) -> Result<PathBuf> {
    let total_start = Instant::now();

    tracing::info!(
        model = %config.model_path,
        features = config.train_config.num_features,
        steps = config.train_config.num_steps,
        "Starting SAE training pipeline"
    );

    // Setup directories
    let sae_dir = config.data_dir.join("sae_training");
    let activations_path = sae_dir.join("activations.bin");
    let checkpoint_dir = sae_dir.join("checkpoints");
    let output_path = sae_dir.join("gemma4_sae_131k.safetensors");
    let progress_path = sae_dir.join("progress.jsonl");
    std::fs::create_dir_all(&sae_dir)?;
    std::fs::create_dir_all(&checkpoint_dir)?;

    // Phase 1: Collect activations (or load from disk)
    let (activations, model_dim) = if config.resume_collection && activations_path.exists() {
        tracing::info!(
            path = %activations_path.display(),
            "Loading previously collected activations"
        );
        ActivationCollector::load_activations(&activations_path)?
    } else {
        collect_activations(&config, &activations_path, &progress_path).await?
    };

    if activations.len() < config.min_samples {
        bail!(
            "Insufficient activations: got {}, need at least {}",
            activations.len(),
            config.min_samples
        );
    }

    // Phase 2: Train SAE
    let mut train_config = config.train_config.clone();
    train_config.model_dim = model_dim;
    train_config.checkpoint_dir = checkpoint_dir.clone();

    train_sae(&activations, &train_config, &output_path, &progress_path).await?;

    let elapsed = total_start.elapsed();
    tracing::info!(
        total_elapsed = format_eta(elapsed),
        output = %output_path.display(),
        features = train_config.num_features,
        model_dim = model_dim,
        samples = activations.len(),
        "SAE training pipeline complete"
    );

    Ok(output_path)
}

/// Phase 1: Collect activations from Gemma 4.
async fn collect_activations(
    config: &TrainingRunConfig,
    activations_path: &Path,
    progress_path: &Path,
) -> Result<(Vec<Vec<f32>>, usize)> {
    tracing::info!("Phase 1: Collecting activations from Gemma 4");

    // Start embedding server
    let embed_url = format!("http://localhost:{}", config.embed_port);
    let _child = start_gemma_embedding_server(config).await?;

    // Wait for health
    let mut collector = ActivationCollector::new(&embed_url);
    collector.wait_for_health(60).await?;

    // Build corpus
    let corpus = collector::build_corpus(&config.data_dir);
    if corpus.is_empty() {
        bail!("Empty corpus — no training data available");
    }

    // Log progress start
    log_progress(progress_path, &TrainingProgress {
        phase: "collection".to_string(),
        step: 0,
        total_steps: corpus.len(),
        progress_pct: 0.0,
        elapsed_secs: 0,
        eta_secs: 0,
        eta_human: "calculating...".to_string(),
        metrics: serde_json::json!({"corpus_size": corpus.len()}),
    });

    // Collect
    let activations = collector.collect_batch(&corpus, 100).await?;
    let dim = collector.activation_dim.unwrap_or(0);

    if dim == 0 {
        bail!("Failed to determine activation dimension");
    }

    // Save to disk
    ActivationCollector::save_activations(&activations, activations_path, dim)?;

    log_progress(progress_path, &TrainingProgress {
        phase: "collection_complete".to_string(),
        step: activations.len(),
        total_steps: corpus.len(),
        progress_pct: 100.0,
        elapsed_secs: 0,
        eta_secs: 0,
        eta_human: "done".to_string(),
        metrics: serde_json::json!({
            "samples_collected": activations.len(),
            "activation_dim": dim,
        }),
    });

    Ok((activations, dim))
}

/// Phase 2: Train the SAE on collected activations.
async fn train_sae(
    activations: &[Vec<f32>],
    config: &TrainConfig,
    output_path: &Path,
    progress_path: &Path,
) -> Result<()> {
    tracing::info!(
        samples = activations.len(),
        features = config.num_features,
        model_dim = config.model_dim,
        steps = config.num_steps,
        "Phase 2: Training SAE on Metal"
    );

    let mut trainer = SaeTrainer::new(config.clone())?;

    // Check for existing checkpoint to resume
    let latest_ckpt = find_latest_checkpoint(&config.checkpoint_dir);
    if let Some(ckpt_path) = latest_ckpt {
        tracing::info!(
            path = %ckpt_path.display(),
            "Resuming from checkpoint"
        );
        trainer.load_checkpoint(&ckpt_path)?;
    }

    let total_start = Instant::now();
    let remaining_steps = config.num_steps - trainer.current_step;
    let batch_size = config.batch_size.min(activations.len());

    tracing::info!(
        remaining_steps = remaining_steps,
        batch_size = batch_size,
        starting_step = trainer.current_step,
        "Training loop starting"
    );

    // Simple deterministic batch cycling
    let mut batch_offset = 0usize;

    for step_idx in 0..remaining_steps {
        // Build batch by cycling through activations
        let batch: Vec<Vec<f32>> = (0..batch_size)
            .map(|i| {
                let idx = (batch_offset + i) % activations.len();
                activations[idx].clone()
            })
            .collect();
        batch_offset = (batch_offset + batch_size) % activations.len();

        let stats = trainer.train_step(&batch)?;

        // Log progress
        if (step_idx + 1) % config.log_interval == 0 || step_idx + 1 == remaining_steps {
            let elapsed = total_start.elapsed();
            let rate = (step_idx + 1) as f64 / elapsed.as_secs_f64();
            let remaining = (remaining_steps - step_idx - 1) as f64 / rate;
            let eta = std::time::Duration::from_secs_f64(remaining);

            tracing::info!(
                step = stats.step,
                total = config.num_steps,
                recon_loss = format!("{:.6}", stats.reconstruction_loss),
                l1_loss = format!("{:.6}", stats.l1_loss),
                total_loss = format!("{:.6}", stats.total_loss),
                active_features = stats.active_features,
                dead_features = stats.dead_features,
                density = format!("{:.4}", stats.feature_density),
                rate_steps_sec = format!("{:.1}", rate),
                elapsed = format_eta(elapsed),
                eta = format_eta(eta),
                progress_pct = format!("{:.1}%", stats.step as f64 / config.num_steps as f64 * 100.0),
                "SAE training progress"
            );

            log_progress(progress_path, &TrainingProgress {
                phase: "training".to_string(),
                step: stats.step,
                total_steps: config.num_steps,
                progress_pct: stats.step as f64 / config.num_steps as f64 * 100.0,
                elapsed_secs: elapsed.as_secs(),
                eta_secs: eta.as_secs(),
                eta_human: format_eta(eta),
                metrics: serde_json::json!({
                    "reconstruction_loss": stats.reconstruction_loss,
                    "l1_loss": stats.l1_loss,
                    "total_loss": stats.total_loss,
                    "active_features": stats.active_features,
                    "dead_features": stats.dead_features,
                    "feature_density": stats.feature_density,
                    "steps_per_sec": rate,
                }),
            });
        }

        // Checkpoint
        if stats.step % config.checkpoint_interval == 0 {
            trainer.checkpoint()?;
        }

        // Dead feature resampling
        if config.dead_feature_resample_interval > 0
            && stats.step % config.dead_feature_resample_interval == 0
            && stats.step > 0
        {
            let resampled = trainer.resample_dead_features(activations)?;
            if resampled > 0 {
                tracing::info!(
                    resampled = resampled,
                    step = stats.step,
                    "Dead features resampled"
                );
            }
        }
    }

    // Save final weights
    trainer.save_safetensors(output_path)?;

    // Final checkpoint
    trainer.checkpoint()?;

    log_progress(progress_path, &TrainingProgress {
        phase: "training_complete".to_string(),
        step: config.num_steps,
        total_steps: config.num_steps,
        progress_pct: 100.0,
        elapsed_secs: total_start.elapsed().as_secs(),
        eta_secs: 0,
        eta_human: "done".to_string(),
        metrics: serde_json::json!({
            "output_path": output_path.display().to_string(),
            "total_elapsed": format_eta(total_start.elapsed()),
        }),
    });

    Ok(())
}

/// Start a llama-server instance in embedding mode with the Gemma 4 model.
async fn start_gemma_embedding_server(
    config: &TrainingRunConfig,
) -> Result<tokio::process::Child> {
    if config.model_path.is_empty() {
        bail!("No model path configured (LLAMACPP_MODEL_PATH)");
    }

    tracing::info!(
        binary = %config.server_binary,
        model = %config.model_path,
        port = config.embed_port,
        "Starting Gemma 4 embedding server for SAE activation collection"
    );

    let child = Command::new(&config.server_binary)
        .args([
            "--model", &config.model_path,
            "--port", &config.embed_port.to_string(),
            "--n-gpu-layers", &config.n_gpu_layers.to_string(),
            "--embeddings",
            "--pooling", "mean",
            "--batch-size", "2048",
            "--ubatch-size", "2048",
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!(
            "Failed to start embedding server with model '{}'",
            config.model_path
        ))?;

    Ok(child)
}

/// Find the latest checkpoint in a directory.
fn find_latest_checkpoint(dir: &Path) -> Option<PathBuf> {
    if !dir.exists() {
        return None;
    }

    let mut checkpoints: Vec<PathBuf> = std::fs::read_dir(dir)
        .ok()?
        .flatten()
        .filter(|e| {
            e.path()
                .extension()
                .map(|ext| ext == "safetensors")
                .unwrap_or(false)
        })
        .map(|e| e.path())
        .collect();

    checkpoints.sort();
    checkpoints.pop()
}

/// Log a progress entry to the JSONL file.
fn log_progress(path: &Path, progress: &TrainingProgress) {
    if let Ok(json) = serde_json::to_string(progress) {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
        {
            let _ = writeln!(f, "{}", json);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_latest_checkpoint() {
        let tmp = tempfile::TempDir::new().unwrap();
        let dir = tmp.path();

        assert!(find_latest_checkpoint(dir).is_none());

        std::fs::write(dir.join("sae_step_001000.safetensors"), "").unwrap();
        std::fs::write(dir.join("sae_step_005000.safetensors"), "").unwrap();
        std::fs::write(dir.join("sae_step_003000.safetensors"), "").unwrap();

        let latest = find_latest_checkpoint(dir).unwrap();
        assert!(latest.to_str().unwrap().contains("005000"));
    }

    #[test]
    fn test_log_progress() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("progress.jsonl");

        log_progress(&path, &TrainingProgress {
            phase: "test".to_string(),
            step: 1,
            total_steps: 100,
            progress_pct: 1.0,
            elapsed_secs: 5,
            eta_secs: 495,
            eta_human: "8m 15s".to_string(),
            metrics: serde_json::json!({"loss": 0.5}),
        });

        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("\"phase\":\"test\""));
        assert!(content.contains("\"step\":1"));
    }
}
