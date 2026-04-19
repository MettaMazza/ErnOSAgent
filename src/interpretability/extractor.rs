// Ern-OS — Activation extractor
//! Extracts hidden state activations from model inference for SAE analysis.
//! Uses the provider's embedding endpoint as a proxy for internal activations.

use crate::provider::Provider;

/// Activation extraction result.
pub struct ExtractionResult {
    pub layer_index: usize,
    pub activations: Vec<f32>,
    pub token_position: usize,
}

/// Extract activations using the provider's embedding endpoint.
/// The embedding endpoint returns the model's internal representation,
/// which serves as the activation vector for SAE analysis.
pub async fn extract_via_embedding(
    provider: &dyn Provider,
    text: &str,
) -> anyhow::Result<ExtractionResult> {
    let activations = provider.embed(text).await?;

    Ok(ExtractionResult {
        layer_index: 0, // Embedding layer
        activations,
        token_position: 0,
    })
}

/// Extract activations from a batch of texts.
pub async fn extract_batch(
    provider: &dyn Provider,
    texts: &[&str],
) -> Vec<ExtractionResult> {
    let mut results = Vec::with_capacity(texts.len());
    for (i, text) in texts.iter().enumerate() {
        match extract_via_embedding(provider, text).await {
            Ok(mut result) => {
                result.token_position = i;
                results.push(result);
            }
            Err(e) => {
                tracing::warn!(text_idx = i, error = %e, "Activation extraction failed");
            }
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extraction_result() {
        let r = ExtractionResult {
            layer_index: 12, activations: vec![0.1, 0.2],
            token_position: 5,
        };
        assert_eq!(r.layer_index, 12);
        assert_eq!(r.activations.len(), 2);
    }
}
