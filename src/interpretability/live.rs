// Ern-OS — Live interpretability monitoring

use super::FeatureActivation;
use std::collections::VecDeque;

/// Rolling window of feature activations for live monitoring.
pub struct LiveMonitor {
    window: VecDeque<Vec<FeatureActivation>>,
    window_size: usize,
}

impl LiveMonitor {
    pub fn new(window_size: usize) -> Self {
        Self { window: VecDeque::new(), window_size }
    }

    pub fn push(&mut self, activations: Vec<FeatureActivation>) {
        if self.window.len() >= self.window_size {
            self.window.pop_front();
        }
        self.window.push_back(activations);
    }

    /// Get the average activation per feature across the window.
    pub fn averages(&self) -> Vec<(usize, f32)> {
        if self.window.is_empty() { return Vec::new(); }
        let mut sums: std::collections::HashMap<usize, (f32, usize)> = std::collections::HashMap::new();
        for activations in &self.window {
            for fa in activations {
                let entry = sums.entry(fa.feature_index).or_insert((0.0, 0));
                entry.0 += fa.activation;
                entry.1 += 1;
            }
        }
        let mut avgs: Vec<(usize, f32)> = sums.into_iter()
            .map(|(idx, (sum, count))| (idx, sum / count as f32))
            .collect();
        avgs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        avgs
    }

    pub fn window_len(&self) -> usize { self.window.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_and_average() {
        let mut monitor = LiveMonitor::new(10);
        monitor.push(vec![FeatureActivation {
            feature_index: 0, label: "test".into(),
            activation: 2.0, baseline: 1.0, delta: 1.0,
        }]);
        let avgs = monitor.averages();
        assert_eq!(avgs.len(), 1);
        assert!((avgs[0].1 - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_window_limit() {
        let mut monitor = LiveMonitor::new(3);
        for _ in 0..5 { monitor.push(Vec::new()); }
        assert_eq!(monitor.window_len(), 3);
    }
}
