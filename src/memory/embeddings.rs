//! Embedding service — provider-agnostic vector generation and cosine similarity.

pub struct EmbeddingService {
    dimensions: usize,
}

impl EmbeddingService {
    pub fn new() -> Self {
        Self { dimensions: 0 }
    }

    pub fn set_dimensions(&mut self, dims: usize) {
        self.dimensions = dims;
    }

    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Compute cosine similarity between two vectors.
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() || a.is_empty() { return 0.0; }
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 { return 0.0; }
        dot / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let similarity = EmbeddingService::cosine_similarity(&a, &a);
        assert!((similarity - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let similarity = EmbeddingService::cosine_similarity(&a, &b);
        assert!(similarity.abs() < f32::EPSILON);
    }

    #[test]
    fn test_cosine_opposite() {
        let a = vec![1.0, 0.0];
        let b = vec![-1.0, 0.0];
        let similarity = EmbeddingService::cosine_similarity(&a, &b);
        assert!((similarity + 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_cosine_different_lengths() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(EmbeddingService::cosine_similarity(&a, &b), 0.0);
    }
}
