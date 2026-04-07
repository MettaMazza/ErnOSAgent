//! Prompt 2: Contextual HUD — regenerated before every inference call.

use crate::model::spec::ModelSpec;
use crate::steering::vectors::SteeringConfig;

/// Build the contextual HUD prompt from live system state.
pub fn build_context_prompt(
    model_spec: &ModelSpec,
    session_title: &str,
    message_count: usize,
    context_usage_pct: f32,
    available_tools: &[String],
    steering: &SteeringConfig,
    memory_summary: &str,
    platform_status: &str,
) -> String {
    let mut sections = Vec::new();

    // Model spec
    sections.push(format!(
        "## Active Model\n\
         - Name: {}\n\
         - Provider: {}\n\
         - Parameters: {} ({})\n\
         - Context: {} tokens\n\
         - Capabilities: {}\n\
         - Temperature: {}, Top-K: {}, Top-P: {}",
        model_spec.name,
        model_spec.provider,
        model_spec.parameter_size,
        model_spec.quantization_level,
        model_spec.context_length,
        model_spec.capabilities.modality_badges(),
        model_spec.default_temperature,
        model_spec.default_top_k,
        model_spec.default_top_p,
    ));

    // Session
    sections.push(format!(
        "## Session\n\
         - Title: {}\n\
         - Messages: {}\n\
         - Context usage: {:.0}%",
        session_title,
        message_count,
        context_usage_pct * 100.0,
    ));

    // Tools
    if !available_tools.is_empty() {
        sections.push(format!(
            "## Available Tools\n{}",
            available_tools.join(", ")
        ));
    }

    // Steering
    if steering.has_active_vectors() {
        sections.push(format!(
            "## Active Steering\n{}",
            steering.status_summary()
        ));
    }

    // Memory
    if !memory_summary.is_empty() {
        sections.push(format!("## Memory State\n{}", memory_summary));
    }

    // Platforms
    if !platform_status.is_empty() {
        sections.push(format!("## Platform Status\n{}", platform_status));
    }

    format!("# System State (Live)\n\n{}", sections.join("\n\n"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::spec::ModelCapabilities;

    #[test]
    fn test_build_context_prompt() {
        let spec = ModelSpec {
            name: "gemma4:26b".to_string(),
            provider: "llamacpp".to_string(),
            parameter_size: "26B".to_string(),
            quantization_level: "Q4_K_M".to_string(),
            context_length: 262144,
            default_temperature: 0.7,
            default_top_k: 40,
            default_top_p: 0.9,
            capabilities: ModelCapabilities {
                text: true, vision: true, video: true,
                tool_calling: true, thinking: true, ..Default::default()
            },
            ..Default::default()
        };

        let prompt = build_context_prompt(
            &spec, "Test session", 5, 0.15,
            &["web_search".to_string(), "file_read".to_string()],
            &SteeringConfig::default(), "", "",
        );

        assert!(prompt.contains("gemma4:26b"));
        assert!(prompt.contains("262144"));
        assert!(prompt.contains("15%"));
        assert!(prompt.contains("web_search"));
    }
}
