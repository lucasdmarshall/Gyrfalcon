//! Rerankers. Rerank off is `NoopReranker`, so the pipeline always has the stage.

use crate::error::Result;

/// Default number of fused hits the reranker scores.
pub const RERANK_K: usize = 32;

pub trait Reranker: Send + Sync {
    /// Scores `(query, body)` pairs. Bodies never include a contextual prefix. Higher is better.
    fn rerank(&self, query: &str, bodies: &[String]) -> Result<Vec<f32>>;
}

/// Returns equal scores, which keeps the fused order.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoopReranker;

impl Reranker for NoopReranker {
    fn rerank(&self, _query: &str, bodies: &[String]) -> Result<Vec<f32>> {
        Ok(vec![0.0; bodies.len()])
    }
}
