//! Reciprocal rank fusion of the lexical and dense lists.

pub const RRF_K: f32 = 60.0;
pub const LEXICAL_WEIGHT: f32 = 1.0;
pub const DENSE_WEIGHT: f32 = 1.0;

/// Sums `weight / (60 + rank)` over both legs, rank starting at 1. A hit missing from a leg
/// adds 0. Sorted by score descending, then chunk id ascending.
pub fn rrf(lexical: &[u64], dense: &[u64]) -> Vec<(u64, f32)> {
    let mut scores: Vec<(u64, f32)> = Vec::with_capacity(lexical.len() + dense.len());
    let mut add = |id: u64, score: f32| match scores.iter_mut().find(|(seen, _)| *seen == id) {
        Some(entry) => entry.1 += score,
        None => scores.push((id, score)),
    };
    for (rank, id) in lexical.iter().enumerate() {
        add(*id, LEXICAL_WEIGHT / (RRF_K + rank as f32 + 1.0));
    }
    for (rank, id) in dense.iter().enumerate() {
        add(*id, DENSE_WEIGHT / (RRF_K + rank as f32 + 1.0));
    }
    scores.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    scores
}
