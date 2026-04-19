// Trainer tests — extracted for governance compliance.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trainer_init() {
        let config = TrainConfig {
            num_features: 256,
            model_dim: 64,
            num_steps: 10,
            batch_size: 4,
            checkpoint_dir: std::env::temp_dir().join("sae_test"),
            ..Default::default()
        };
        let trainer = SaeTrainer::new(config);
        assert!(trainer.is_ok(), "Trainer init failed: {:?}", trainer.err());
    }

    #[test]
    fn test_train_step_loss_decreases() {
        let config = TrainConfig {
            num_features: 128,
            model_dim: 32,
            num_steps: 10,
            batch_size: 8,
            l1_coefficient: 1e-4,
            learning_rate: 1e-3,
            checkpoint_dir: std::env::temp_dir().join("sae_test_loss"),
            ..Default::default()
        };
        let mut trainer = SaeTrainer::new(config).unwrap();

        // Create some random activations
        let activations: Vec<Vec<f32>> = (0..8)
            .map(|i| (0..32).map(|j| ((i * 32 + j) as f32 / 256.0) - 0.5).collect())
            .collect();

        let stats1 = trainer.train_step(&activations).unwrap();
        // Run a few more steps
        for _ in 0..9 {
            trainer.train_step(&activations).unwrap();
        }
        let stats10 = trainer.train_step(&activations).unwrap();

        // Loss should decrease (or at least not explode)
        assert!(
            stats10.total_loss < stats1.total_loss * 2.0,
            "Loss exploded: {} -> {}",
            stats1.total_loss,
            stats10.total_loss
        );
    }

    #[test]
    fn test_checkpoint_save_load() {
        let dir = std::env::temp_dir().join("sae_test_ckpt");
        let config = TrainConfig {
            num_features: 64,
            model_dim: 16,
            num_steps: 5,
            batch_size: 4,
            checkpoint_dir: dir.clone(),
            ..Default::default()
        };

        let mut trainer = SaeTrainer::new(config.clone()).unwrap();
        let activations: Vec<Vec<f32>> = (0..4)
            .map(|_| vec![0.1f32; 16])
            .collect();
        trainer.train_step(&activations).unwrap();

        let ckpt_path = trainer.checkpoint().unwrap();
        assert!(ckpt_path.exists());

        // Load into fresh trainer
        let mut trainer2 = SaeTrainer::new(config).unwrap();
        trainer2.load_checkpoint(&ckpt_path).unwrap();
        assert_eq!(trainer2.current_step, 1);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_export_sae() {
        let config = TrainConfig {
            num_features: 64,
            model_dim: 16,
            num_steps: 1,
            batch_size: 4,
            checkpoint_dir: std::env::temp_dir().join("sae_test_export"),
            ..Default::default()
        };
        let trainer = SaeTrainer::new(config).unwrap();
        let sae = trainer.export_sae().unwrap();
        assert_eq!(sae.num_features, 64);
        assert_eq!(sae.model_dim, 16);

        // Verify encode works
        let activations = vec![0.1f32; 16];
        let features = sae.encode(&activations, 10);
        assert!(features.len() <= 10);
    }
}
