//! Rerankers. Rerank off is `NoopReranker`, so the pipeline always has the stage.

use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::error::Result;
use crate::ort_session::OrtModel;

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

/// The default local reranker: ort, ONNX weights for `Xenova/ms-marco-MiniLM-L-6-v2`.
pub struct OrtReranker {
    model: OrtModel,
}

impl OrtReranker {
    pub fn from_dir(dir: &Path) -> Result<OrtReranker> {
        Ok(OrtReranker { model: OrtModel::load(dir)? })
    }
}

impl Reranker for OrtReranker {
    fn rerank(&self, query: &str, bodies: &[String]) -> Result<Vec<f32>> {
        self.model.score_pairs(query, bodies)
    }
}

/// Test double: scores a body by a scripted table and counts calls.
#[derive(Debug, Clone, Default)]
pub struct ScriptedReranker {
    /// `(substring, score)`: a body containing the substring gets that score, else 0.
    pub scores: Vec<(String, f32)>,
    calls: Arc<AtomicUsize>,
}

impl ScriptedReranker {
    pub fn new(scores: Vec<(String, f32)>) -> ScriptedReranker {
        ScriptedReranker { scores, calls: Arc::new(AtomicUsize::new(0)) }
    }

    pub fn rerank_calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}

impl Reranker for ScriptedReranker {
    fn rerank(&self, _query: &str, bodies: &[String]) -> Result<Vec<f32>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(bodies
            .iter()
            .map(|body| self.scores.iter().find(|(needle, _)| body.contains(needle.as_str())).map(|(_, s)| *s).unwrap_or(0.0))
            .collect())
    }
}
