//! Modality router — routes content to the correct model/provider based on input type.
//!
//! The engine does the routing, not the agent. Audio goes to E2B/E4B capable models,
//! text/image/video goes to the primary model. If no capable provider exists for a
//! modality, the system displays a clear error — it does not silently degrade.

use crate::model::spec::{Modality, ModelSpec};
use anyhow::{bail, Result};

/// Routing decision for a given modality.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// The model to use.
    pub model_name: String,
    /// The provider to route through.
    pub provider_id: String,
    /// The modality being routed.
    pub modality: Modality,
}

/// Routes content to the appropriate model/provider combination.
pub struct ModalityRouter {
    /// The primary model spec (e.g. gemma4:26b).
    primary_spec: ModelSpec,
    /// The primary provider id.
    primary_provider: String,
    /// Optional audio-capable model spec (e.g. gemma4:e2b).
    audio_spec: Option<(ModelSpec, String)>,
}

impl ModalityRouter {
    /// Create a new router with the primary model.
    /// `audio_model` is an optional (ModelSpec, provider_id) for audio-capable models.
    pub fn new(
        primary_spec: ModelSpec,
        primary_provider: String,
        audio_model: Option<(ModelSpec, String)>,
    ) -> Self {
        Self {
            primary_spec,
            primary_provider,
            audio_spec: audio_model,
        }
    }

    /// Determine the routing for a given modality.
    pub fn route(&self, modality: Modality) -> Result<RoutingDecision> {
        match modality {
            Modality::Text | Modality::Image | Modality::Video => {
                if !self.primary_spec.supports(modality) {
                    bail!(
                        "Primary model '{}' does not support {} input. \
                         Available modalities: {}",
                        self.primary_spec.name,
                        modality,
                        self.primary_spec
                            .capabilities
                            .supported_modalities()
                            .iter()
                            .map(|m| m.to_string())
                            .collect::<Vec<_>>()
                            .join(", ")
                    );
                }

                Ok(RoutingDecision {
                    model_name: self.primary_spec.name.clone(),
                    provider_id: self.primary_provider.clone(),
                    modality,
                })
            }

            Modality::Audio => {
                // First check if primary supports audio (unlikely for 26B)
                if self.primary_spec.supports(Modality::Audio) {
                    return Ok(RoutingDecision {
                        model_name: self.primary_spec.name.clone(),
                        provider_id: self.primary_provider.clone(),
                        modality,
                    });
                }

                // Route to audio-capable model
                if let Some((ref audio_spec, ref audio_provider)) = self.audio_spec {
                    if audio_spec.supports(Modality::Audio) {
                        return Ok(RoutingDecision {
                            model_name: audio_spec.name.clone(),
                            provider_id: audio_provider.clone(),
                            modality,
                        });
                    }
                }

                bail!(
                    "No audio-capable model available. The primary model '{}' does not \
                     support audio input. Configure an audio-capable model (e.g. gemma4:e2b) \
                     or load a Whisper model for speech-to-text.",
                    self.primary_spec.name
                );
            }
        }
    }

    /// Update the primary model spec (e.g. after a /model swap).
    pub fn set_primary(&mut self, spec: ModelSpec, provider: String) {
        self.primary_spec = spec;
        self.primary_provider = provider;
    }

    /// Update the audio-capable model.
    pub fn set_audio_model(&mut self, spec: ModelSpec, provider: String) {
        self.audio_spec = Some((spec, provider));
    }

    /// Clear the audio-capable model.
    pub fn clear_audio_model(&mut self) {
        self.audio_spec = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::spec::ModelCapabilities;

    fn make_primary() -> ModelSpec {
        ModelSpec {
            name: "gemma4:26b".to_string(),
            capabilities: ModelCapabilities {
                text: true,
                vision: true,
                audio: false,
                video: true,
                tool_calling: true,
                thinking: true,
            },
            ..Default::default()
        }
    }

    fn make_audio() -> ModelSpec {
        ModelSpec {
            name: "gemma4:e2b".to_string(),
            capabilities: ModelCapabilities {
                text: true,
                vision: true,
                audio: true,
                video: true,
                tool_calling: false,
                thinking: false,
            },
            ..Default::default()
        }
    }

    #[test]
    fn test_route_text_to_primary() {
        let router = ModalityRouter::new(make_primary(), "llamacpp".to_string(), None);
        let decision = router.route(Modality::Text).unwrap();
        assert_eq!(decision.model_name, "gemma4:26b");
        assert_eq!(decision.provider_id, "llamacpp");
    }

    #[test]
    fn test_route_image_to_primary() {
        let router = ModalityRouter::new(make_primary(), "llamacpp".to_string(), None);
        let decision = router.route(Modality::Image).unwrap();
        assert_eq!(decision.model_name, "gemma4:26b");
    }

    #[test]
    fn test_route_video_to_primary() {
        let router = ModalityRouter::new(make_primary(), "llamacpp".to_string(), None);
        let decision = router.route(Modality::Video).unwrap();
        assert_eq!(decision.model_name, "gemma4:26b");
    }

    #[test]
    fn test_route_audio_to_e2b() {
        let router = ModalityRouter::new(
            make_primary(),
            "llamacpp".to_string(),
            Some((make_audio(), "llamacpp-audio".to_string())),
        );
        let decision = router.route(Modality::Audio).unwrap();
        assert_eq!(decision.model_name, "gemma4:e2b");
        assert_eq!(decision.provider_id, "llamacpp-audio");
    }

    #[test]
    fn test_route_audio_error_when_no_audio_model() {
        let router = ModalityRouter::new(make_primary(), "llamacpp".to_string(), None);
        let result = router.route(Modality::Audio);
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("No audio-capable model available"));
    }

    #[test]
    fn test_set_primary() {
        let mut router = ModalityRouter::new(make_primary(), "llamacpp".to_string(), None);
        let new_spec = ModelSpec {
            name: "llama3.1:70b".to_string(),
            capabilities: ModelCapabilities {
                text: true,
                ..Default::default()
            },
            ..Default::default()
        };
        router.set_primary(new_spec, "ollama".to_string());
        let decision = router.route(Modality::Text).unwrap();
        assert_eq!(decision.model_name, "llama3.1:70b");
        assert_eq!(decision.provider_id, "ollama");
    }

    #[test]
    fn test_set_audio_model() {
        let mut router = ModalityRouter::new(make_primary(), "llamacpp".to_string(), None);
        assert!(router.route(Modality::Audio).is_err());

        router.set_audio_model(make_audio(), "lmstudio".to_string());
        let decision = router.route(Modality::Audio).unwrap();
        assert_eq!(decision.model_name, "gemma4:e2b");
    }

    #[test]
    fn test_clear_audio_model() {
        let mut router = ModalityRouter::new(
            make_primary(),
            "llamacpp".to_string(),
            Some((make_audio(), "lmstudio".to_string())),
        );
        assert!(router.route(Modality::Audio).is_ok());

        router.clear_audio_model();
        assert!(router.route(Modality::Audio).is_err());
    }

    #[test]
    fn test_route_unsupported_modality_on_primary() {
        let spec = ModelSpec {
            name: "text-only".to_string(),
            capabilities: ModelCapabilities {
                text: true,
                ..Default::default()
            },
            ..Default::default()
        };
        let router = ModalityRouter::new(spec, "test".to_string(), None);
        let result = router.route(Modality::Image);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not support image"));
    }
}
