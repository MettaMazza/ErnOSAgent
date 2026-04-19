// Ern-OS — Steering server integration
//! Generates llama-server args for active steering vectors.

use super::SteeringVector;

/// Build command-line args for active steering vectors.
pub fn build_steering_args(vectors: &[SteeringVector]) -> Vec<String> {
        tracing::info!(module = "steering_server", fn_name = "build_steering_args", "steering_server::build_steering_args called");
    let mut args = Vec::new();
    for v in vectors.iter().filter(|v| v.active) {
        args.push("--control-vector".to_string());
        args.push(v.path.clone());
        args.push("--control-vector-scaled".to_string());
        args.push(format!("{:.2}", v.strength));
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_active() {
        let vectors = vec![SteeringVector {
            name: "x".into(), path: "x.gguf".into(),
            strength: 1.0, active: false, description: String::new(),
        }];
        assert!(build_steering_args(&vectors).is_empty());
    }

    #[test]
    fn test_active_vector() {
        let vectors = vec![SteeringVector {
            name: "focus".into(), path: "/vectors/focus.gguf".into(),
            strength: 0.75, active: true, description: String::new(),
        }];
        let args = build_steering_args(&vectors);
        assert_eq!(args.len(), 4);
        assert_eq!(args[0], "--control-vector");
        assert_eq!(args[1], "/vectors/focus.gguf");
        assert_eq!(args[3], "0.75");
    }
}
