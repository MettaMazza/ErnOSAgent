// Ern-OS — SAE Training CLI
// Standalone binary to run the full SAE training pipeline.
// Usage: cargo run --release --bin train_sae

use anyhow::{Context, Result};

#[tokio::main]
async fn main() -> Result<()> {
    // Init logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let config = ern_os::config::AppConfig::load()
        .context("Failed to load ern-os.toml")?;

    let model_path = config.llamacpp.model_path.clone();
    let server_binary = config.llamacpp.server_binary.clone();
    let data_dir = config.general.data_dir.clone();

    tracing::info!(
        model = %model_path,
        "Starting SAE training pipeline for Gemma 4 31B"
    );

    let train_config = ern_os::interpretability::trainer::TrainConfig {
        num_features: 131_072,  // 128K features — Gemma Scope standard
        model_dim: 0,          // auto-detected from activations (5376 for 31B dense)
        l1_coefficient: 5e-3,
        learning_rate: 3e-4,
        weight_decay: 0.0,
        num_steps: 100_000,
        batch_size: 4096,
        log_interval: 10,
        checkpoint_interval: 5000,
        dead_feature_resample_interval: 25_000,
        jump_threshold: 0.001,
        checkpoint_dir: data_dir.join("sae_training/checkpoints"),
    };

    let run_config = ern_os::interpretability::train_runner::TrainingRunConfig {
        model_path,
        server_binary,
        embed_port: 8082,
        n_gpu_layers: -1,  // All layers on GPU (if available)
        data_dir,
        train_config,
        min_samples: 50,  // 95+ diversity prompts should exceed this easily
        resume_collection: true,
    };

    // Estimate training time
    let eta = ern_os::interpretability::trainer_persist::estimate_training_time(&run_config.train_config);
    tracing::info!(
        features = run_config.train_config.num_features,
        steps = run_config.train_config.num_steps,
        batch_size = run_config.train_config.batch_size,
        estimated_time = format!("{:.0}m", eta.as_secs_f64() / 60.0),
        "Training plan"
    );

    let output = ern_os::interpretability::train_runner::run_sae_training(run_config).await?;

    tracing::info!(
        output = %output.display(),
        "SAE training complete — weights saved"
    );

    Ok(())
}
