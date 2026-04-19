// Ern-OS — Observer-driven training buffer
//! Captures observer audit verdicts as training signals.

use super::buffers_rejection::PreferencePair;
use serde::{Deserialize, Serialize};

/// Observer audit result as a training signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObserverSignal {
    pub query: String,
    pub response: String,
    pub approved: bool,
    pub score: f32,
    pub reason: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Process observer signals into training data.
pub fn process_signals(
    signals: &[ObserverSignal],
) -> (Vec<super::TrainingSample>, Vec<PreferencePair>) {
        tracing::info!(module = "observer_buffer", fn_name = "process_signals", "observer_buffer::process_signals called");
    let mut golden = Vec::new();
    let mut pairs = Vec::new();

    // Group by query for preference pairs
    let mut by_query: std::collections::HashMap<String, Vec<&ObserverSignal>> =
        std::collections::HashMap::new();
    for signal in signals {
        by_query.entry(signal.query.clone()).or_default().push(signal);
    }

    for (query, sigs) in &by_query {
        let approved: Vec<_> = sigs.iter().filter(|s| s.approved).collect();
        let rejected: Vec<_> = sigs.iter().filter(|s| !s.approved).collect();

        // Golden samples from approved responses
        for sig in &approved {
            golden.push(super::TrainingSample {
                id: uuid::Uuid::new_v4().to_string(),
                input: sig.query.clone(),
                output: sig.response.clone(),
                method: super::TrainingMethod::Sft,
                quality_score: sig.score / 10.0,
                timestamp: sig.timestamp,
            });
        }

        // Preference pairs from approved + rejected combos
        for chosen in &approved {
            for rej in &rejected {
                pairs.push(PreferencePair {
                    id: uuid::Uuid::new_v4().to_string(),
                    input: query.clone(),
                    chosen: chosen.response.clone(),
                    rejected: rej.response.clone(),
                    rejection_reason: rej.reason.clone().unwrap_or_default(),
                    timestamp: chrono::Utc::now(),
                });
            }
        }
    }

    (golden, pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_process_mixed_signals() {
        let signals = vec![
            ObserverSignal {
                query: "What is Rust?".into(), response: "Great answer".into(),
                approved: true, score: 9.0, reason: None, timestamp: chrono::Utc::now(),
            },
            ObserverSignal {
                query: "What is Rust?".into(), response: "Bad answer".into(),
                approved: false, score: 3.0, reason: Some("Too vague".into()),
                timestamp: chrono::Utc::now(),
            },
        ];

        let (golden, pairs) = process_signals(&signals);
        assert_eq!(golden.len(), 1);
        assert_eq!(pairs.len(), 1);
        assert_eq!(pairs[0].chosen, "Great answer");
        assert_eq!(pairs[0].rejected, "Bad answer");
    }
}
