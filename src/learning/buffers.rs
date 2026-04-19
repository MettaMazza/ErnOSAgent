// Ern-OS — Training data buffers
//! Golden buffer for approved high-quality samples.

use super::TrainingSample;
use anyhow::Result;
use std::path::{Path, PathBuf};

/// Golden buffer — stores approved, high-quality training samples.
pub struct GoldenBuffer {
    samples: Vec<TrainingSample>,
    file_path: Option<PathBuf>,
    max_samples: usize,
}

impl GoldenBuffer {
    pub fn new(max_samples: usize) -> Self {
        Self { samples: Vec::new(), file_path: None, max_samples }
    }

    pub fn open(path: &Path, max_samples: usize) -> Result<Self> {
        tracing::info!(module = "golden_buffer", fn_name = "open", "golden_buffer::open called");
        let mut buf = Self { samples: Vec::new(), file_path: Some(path.to_path_buf()), max_samples };
        if path.exists() {
            let content = std::fs::read_to_string(path)?;
            buf.samples = serde_json::from_str(&content)?;
        }
        Ok(buf)
    }

    pub fn add(&mut self, sample: TrainingSample) -> Result<()> {
        tracing::info!(module = "golden_buffer", fn_name = "add", "golden_buffer::add called");
        if self.samples.len() >= self.max_samples {
            // Remove lowest quality to make room
            self.samples.sort_by(|a, b| a.quality_score.partial_cmp(&b.quality_score).unwrap_or(std::cmp::Ordering::Equal));
            self.samples.remove(0);
        }
        self.samples.push(sample);
        self.persist()
    }

    pub fn drain_batch(&mut self, batch_size: usize) -> Vec<TrainingSample> {
        tracing::info!(module = "golden_buffer", fn_name = "drain_batch", "golden_buffer::drain_batch called");
        let n = batch_size.min(self.samples.len());
        self.samples.drain(..n).collect()
    }

    pub fn count(&self) -> usize { self.samples.len() }

    fn persist(&self) -> Result<()> {
        if let Some(ref path) = self.file_path {
            if let Some(parent) = path.parent() { std::fs::create_dir_all(parent)?; }
            std::fs::write(path, serde_json::to_string(&self.samples)?)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::TrainingMethod;

    fn sample(score: f32) -> TrainingSample {
        TrainingSample {
            id: uuid::Uuid::new_v4().to_string(),
            input: "test".into(), output: "resp".into(),
            method: TrainingMethod::Sft, quality_score: score,
            timestamp: chrono::Utc::now(),
        }
    }

    #[test]
    fn test_add_and_count() {
        let mut buf = GoldenBuffer::new(100);
        buf.add(sample(0.9)).unwrap();
        assert_eq!(buf.count(), 1);
    }

    #[test]
    fn test_eviction() {
        let mut buf = GoldenBuffer::new(2);
        buf.add(sample(0.5)).unwrap();
        buf.add(sample(0.9)).unwrap();
        buf.add(sample(0.8)).unwrap();
        assert_eq!(buf.count(), 2);
        // Lowest quality (0.5) should have been evicted
    }

    #[test]
    fn test_drain_batch() {
        let mut buf = GoldenBuffer::new(100);
        buf.add(sample(0.9)).unwrap();
        buf.add(sample(0.8)).unwrap();
        let batch = buf.drain_batch(1);
        assert_eq!(batch.len(), 1);
        assert_eq!(buf.count(), 1);
    }
}
