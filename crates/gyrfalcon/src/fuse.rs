//! Reciprocal rank fusion of the lexical and dense lists.

pub const RRF_K: f32 = 60.0;
pub const LEXICAL_WEIGHT: f32 = 1.0;
pub const DENSE_WEIGHT: f32 = 1.0;

/// Sums `weight / (60 + rank)` over both legs, rank starting at 1. A hit missing from a leg
/// adds 0. Sorted by score descending, then chunk id ascending.
pub fn rrf(lexical: &[u64], dense: &[u64]) -> Vec<(u64, f32)> {
    rrf_weighted(lexical, dense, [LEXICAL_WEIGHT, DENSE_WEIGHT])
}

/// `rrf` with per-leg weights `[lexical, dense]`.
pub fn rrf_weighted(lexical: &[u64], dense: &[u64], weights: [f32; 2]) -> Vec<(u64, f32)> {
    let mut scores: Vec<(u64, f32)> = Vec::with_capacity(lexical.len() + dense.len());
    let mut add = |id: u64, score: f32| match scores.iter_mut().find(|(seen, _)| *seen == id) {
        Some(entry) => entry.1 += score,
        None => scores.push((id, score)),
    };
    for (rank, id) in lexical.iter().enumerate() {
        add(*id, weights[0] / (RRF_K + rank as f32 + 1.0));
    }
    for (rank, id) in dense.iter().enumerate() {
        add(*id, weights[1] / (RRF_K + rank as f32 + 1.0));
    }
    scores.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scores
}

/// How the two ranked legs become one list.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum Fusion {
    /// Reciprocal rank fusion with `Index::rrf_weights` (the spec default).
    #[default]
    Rrf,
    /// Min-max normalized BM25 and cosine similarity, mixed as
    /// `(1 - dense_weight) * bm25 + dense_weight * cosine`. A hit missing from a leg scores 0
    /// on that leg.
    Score { dense_weight: f32 },
    /// `Score`, but every candidate from either leg gets its exact cosine similarity from the
    /// stored vector, so a lexical-only hit is not penalized on the dense side.
    Rescore { dense_weight: f32 },
}

fn min_max(values: &mut [(u64, f32)]) {
    let (min, max) =
        values.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), (_, v)| (lo.min(*v), hi.max(*v)));
    for (_, value) in values.iter_mut() {
        *value = if max > min { (*value - min) / (max - min) } else { 1.0 };
    }
}

/// Mixes normalized lexical scores and dense similarities. Sorted by score descending, then
/// chunk id ascending.
pub fn score_fusion(lexical: &[(u64, f32)], dense: &[(u64, f32)], dense_weight: f32) -> Vec<(u64, f32)> {
    let mut lexical = lexical.to_vec();
    let mut dense = dense.to_vec();
    min_max(&mut lexical);
    min_max(&mut dense);
    let mut scores: std::collections::HashMap<u64, f32> = std::collections::HashMap::new();
    for (id, value) in lexical {
        *scores.entry(id).or_default() += (1.0 - dense_weight) * value;
    }
    for (id, value) in dense {
        *scores.entry(id).or_default() += dense_weight * value;
    }
    let mut out: Vec<(u64, f32)> = scores.into_iter().collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    out
}
