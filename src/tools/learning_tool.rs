// Ern-OS — Learning tool — live access to training pipeline

use crate::web::state::AppState;
use anyhow::Result;

pub async fn execute(args: &serde_json::Value, state: &AppState) -> Result<String> {
    tracing::info!(tool = "learning", "tool START");
    let action = args["action"].as_str().unwrap_or("");
    match action {
        "status" => get_status(state).await,
        "buffer_stats" => get_buffer_stats(state).await,
        "trigger_training" => trigger_training(args, state).await,
        "list_adapters" => list_adapters().await,
        "sleep" => trigger_sleep(state).await,
        other => Ok(format!("Unknown learning action: {}", other)),
    }
}

async fn get_status(state: &AppState) -> Result<String> {
    let golden = state.golden_buffer.read().await;
    let rejection = state.rejection_buffer.read().await;
    Ok(format!(
        "Training pipeline:\n• Golden buffer: {} samples\n• Rejection buffer: {} pairs\n• Status: {}",
        golden.count(), rejection.count(),
        if golden.count() >= 10 { "Ready for training" } else { "Accumulating samples" }
    ))
}

async fn get_buffer_stats(state: &AppState) -> Result<String> {
    let golden = state.golden_buffer.read().await;
    let rejection = state.rejection_buffer.read().await;
    Ok(format!(
        "Golden: {} / 500 capacity\nRejection: {} pairs\nThreshold: 10 golden OR 5 rejection",
        golden.count(), rejection.count()
    ))
}

async fn trigger_training(_args: &serde_json::Value, state: &AppState) -> Result<String> {
    let config = crate::learning::sleep::SleepConfig::default();
    let mut golden = state.golden_buffer.write().await;
    let mut rejection = state.rejection_buffer.write().await;
    let mut memory = state.memory.write().await;

    let golden_count = golden.count();
    let rejection_count = rejection.count();

    if golden_count < 5 && rejection_count < 3 {
        return Ok(format!(
            "Insufficient data for training. Need ≥5 golden (have {}) or ≥3 rejection pairs (have {}).",
            golden_count, rejection_count
        ));
    }

    match crate::learning::sleep::run_sleep_cycle(
        &config, &mut golden, &mut rejection, &mut memory
    ).await {
        Ok(report) => {
            let mut result = String::from("Training cycle complete:\n");
            if report.golden_trained > 0 {
                result.push_str(&format!(
                    "• SFT: {} samples trained (loss: {:.4})\n",
                    report.golden_trained,
                    report.sft_loss.unwrap_or(0.0)
                ));
            }
            if report.pairs_trained > 0 {
                result.push_str(&format!(
                    "• DPO: {} pairs trained (loss: {:.4})\n",
                    report.pairs_trained,
                    report.preference_loss.unwrap_or(0.0)
                ));
            }
            result.push_str(&format!("• Synaptic edges decayed: {}", report.edges_decayed));
            Ok(result)
        }
        Err(e) => Ok(format!("Training cycle failed: {}", e)),
    }
}

async fn trigger_sleep(state: &AppState) -> Result<String> {
    // Sleep cycle = training + memory consolidation + synaptic decay
    trigger_training(&serde_json::json!({}), state).await
}

async fn list_adapters() -> Result<String> {
    let adapters_dir = std::path::Path::new("data/adapters");
    if !adapters_dir.exists() {
        return Ok("No adapters directory found. Training will create adapters in data/adapters/".to_string());
    }
    let mut entries = Vec::new();
    for entry in std::fs::read_dir(adapters_dir)?.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let meta = entry.metadata()?;
        let size_kb = meta.len() / 1024;
        entries.push(format!("• {} ({}KB)", name, size_kb));
    }
    if entries.is_empty() {
        Ok("Adapters directory exists but is empty. Run training to generate adapters.".to_string())
    } else {
        Ok(format!("Trained adapters:\n{}", entries.join("\n")))
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_training_method_parse() {
        let method = "sft";
        assert_eq!(method.to_uppercase(), "SFT");
    }
}
