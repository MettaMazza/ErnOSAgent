// Ern-OS — GRPO self-play module declarations
pub mod generation;
pub mod rewards;
pub mod training;

/// GRPO configuration.
pub struct GrpoConfig {
    pub group_size: usize,
    pub temperature: f32,
    pub beta: f32,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            group_size: 4,
            temperature: 0.7,
            beta: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default() {
        let config = GrpoConfig::default();
        assert_eq!(config.group_size, 4);
    }
}
