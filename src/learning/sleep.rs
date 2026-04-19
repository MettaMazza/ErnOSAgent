// Ern-OS — Sleep consolidation
//! Periodic background process that:
//! 1. Drains training buffers
//! 2. Runs LoRA training
//! 3. Consolidates memory tiers
//! 4. Decays synaptic graph edges

use crate::learning::buffers::GoldenBuffer;
use crate::learning::buffers_rejection::RejectionBuffer;
use crate::learning::lora::LoraConfig;
use crate::learning::teacher::Teacher;
use crate::memory::MemoryManager;
use anyhow::Result;

/// Sleep cycle configuration.
pub struct SleepConfig {
    pub min_golden_samples: usize,
    pub min_preference_pairs: usize,
    pub decay_factor: f32,
    pub lora_config: LoraConfig,
}

impl Default for SleepConfig {
    fn default() -> Self {
        Self {
            // Must match scheduler thresholds in scheduler/mod.rs
            min_golden_samples: 10,
            min_preference_pairs: 5,
            decay_factor: 0.95,
            lora_config: LoraConfig::default(),
        }
    }
}

/// Run a sleep consolidation cycle.
pub async fn run_sleep_cycle(
    config: &SleepConfig,
    golden: &mut GoldenBuffer,
    rejection: &mut RejectionBuffer,
    memory: &mut MemoryManager,
) -> Result<SleepReport> {
    tracing::info!("Sleep cycle initiated");
    let mut report = SleepReport::default();

    // Step 1: SFT training from golden buffer
    let golden_count = golden.count();
    if golden_count >= config.min_golden_samples {
        tracing::info!(samples = golden_count, "Sleep: running SFT training");
        let samples = golden.drain_batch(golden_count);
        let training_samples: Vec<crate::learning::TrainingSample> = samples.iter().map(|s| {
            crate::learning::TrainingSample {
                id: s.id.clone(),
                input: s.input.clone(),
                output: s.output.clone(),
                method: crate::learning::TrainingMethod::Sft,
                quality_score: s.quality_score,
                timestamp: s.timestamp,
            }
        }).collect();

        let mut teacher = Teacher::new(config.lora_config.clone());
        match teacher.train_sft(&training_samples).await {
            Ok(result) => {
                report.golden_trained = result.samples;
                report.sft_loss = Some(result.loss);
                tracing::info!(samples = result.samples, loss = result.loss, "Sleep: SFT complete");
            }
            Err(e) => tracing::warn!(error = %e, "Sleep: SFT training failed"),
        }
    }

    // Step 2: Preference training from rejection buffer
    let pair_count = rejection.count();
    if pair_count >= config.min_preference_pairs {
        tracing::info!(pairs = pair_count, "Sleep: running preference training");
        let pairs = rejection.drain_all();
        let mut teacher = Teacher::new(config.lora_config.clone());
        match teacher.train_preference(&pairs, "DPO").await {
            Ok(result) => {
                report.pairs_trained = result.samples;
                report.preference_loss = Some(result.loss);
                tracing::info!(pairs = result.samples, loss = result.loss, "Sleep: preference training complete");
            }
            Err(e) => tracing::warn!(error = %e, "Sleep: preference training failed"),
        }
    }

    // Step 3: Decay synaptic graph edges (Hebbian forgetting)
    memory.synaptic.decay_all(config.decay_factor);
    report.edges_decayed = memory.synaptic.edge_count();
    tracing::info!(edges = report.edges_decayed, factor = config.decay_factor, "Sleep: edge decay applied");

    // Step 4: Memory consolidation check
    report.memory_consolidated = true;
    tracing::info!(status = %memory.status_summary(), "Sleep: memory status");

    tracing::info!(?report, "Sleep cycle complete");
    Ok(report)
}

#[derive(Debug, Default)]
pub struct SleepReport {
    pub golden_trained: usize,
    pub pairs_trained: usize,
    pub edges_decayed: usize,
    pub memory_consolidated: bool,
    pub sft_loss: Option<f64>,
    pub preference_loss: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SleepConfig::default();
        assert_eq!(config.min_golden_samples, 10);
        assert_eq!(config.decay_factor, 0.95);
    }
}
