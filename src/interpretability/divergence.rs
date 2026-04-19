// Ern-OS — Activation divergence tracking

/// Compute KL divergence between two activation distributions.
pub fn kl_divergence(baseline: &[f32], current: &[f32]) -> f32 {
    if baseline.len() != current.len() || baseline.is_empty() { return 0.0; }
    let eps = 1e-8;
    baseline.iter().zip(current).map(|(p, q)| {
        let p = p.max(eps);
        let q = q.max(eps);
        p * (p / q).ln()
    }).sum()
}

/// Compute cosine distance (1 - cosine_similarity).
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - crate::memory::embeddings::cosine_similarity(a, b)
}

/// Track divergence over time.
pub struct DivergenceTracker {
    baseline: Vec<f32>,
    history: Vec<f32>,
}

impl DivergenceTracker {
    pub fn new(baseline: Vec<f32>) -> Self {
        Self { baseline, history: Vec::new() }
    }

    pub fn record(&mut self, current: &[f32]) -> f32 {
        let div = kl_divergence(&self.baseline, current);
        self.history.push(div);
        div
    }

    pub fn trend(&self) -> f32 {
        if self.history.len() < 2 { return 0.0; }
        let last = self.history.last().copied().unwrap_or(0.0);
        let prev = self.history[self.history.len() - 2];
        last - prev
    }

    pub fn mean_divergence(&self) -> f32 {
        if self.history.is_empty() { return 0.0; }
        self.history.iter().sum::<f32>() / self.history.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kl_identical() {
        let a = vec![0.5, 0.5];
        assert!((kl_divergence(&a, &a)).abs() < 0.001);
    }

    #[test]
    fn test_tracker() {
        let mut tracker = DivergenceTracker::new(vec![0.5, 0.5]);
        let d1 = tracker.record(&[0.6, 0.4]);
        assert!(d1 > 0.0);
        let d2 = tracker.record(&[0.7, 0.3]);
        assert!(d2 > d1);
        assert!(tracker.trend() > 0.0);
    }
}
