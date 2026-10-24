//! Retrieve: prepare once, fan out over every segment's two engines, merge, fuse, rerank.

use std::path::PathBuf;

use crate::error::{Error, Result};
use crate::fuse::{rrf_weighted, score_fusion, Fusion};
use crate::index::{Index, Snapshot};
use crate::lexical::{CollectionStats, ScoredId};
use crate::prepare::{prepare, Token};
use crate::rerank::{NoopReranker, Reranker, RERANK_K};
use crate::vector::{DenseQuery, FallbackStep};

/// Per-leg list length before fusion.
pub const LEG_LIMIT: usize = 50;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetrieveOptions {
    pub rerank: bool,
    pub limit: usize,
}

impl Default for RetrieveOptions {
    fn default() -> Self {
        RetrieveOptions { rerank: true, limit: 10 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub chunk_id: u64,
    pub doc_id: String,
    pub body: String,
    pub prefix: Option<String>,
    pub byte_start: u64,
    pub byte_end: u64,
    /// RRF score, or the reranker score when rerank ran.
    pub score: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Hits {
    pub chunk_ids: Vec<u64>,
    pub hits: Vec<Hit>,
    /// `embed` calls this retrieve made. Always 1 for a non-empty query.
    pub embed_calls: usize,
    /// Deepest HNSW step any segment reached.
    pub fallback: FallbackStep,
    pub epoch: u64,
    pub warning: Option<String>,
}

/// Merged dense results across segments.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseHits {
    pub ids: Vec<u64>,
    pub distances: Vec<f32>,
    pub deepest: FallbackStep,
    /// The query vector as embedded, for exact rescoring.
    pub query: Option<Vec<f32>>,
}

pub(crate) fn lexical_leg(snapshot: &Snapshot, tokens: &[Token], limit: usize) -> Result<Vec<ScoredId>> {
    let stats = CollectionStats {
        num_docs: snapshot.manifest.bm25_num_docs,
        num_tokens: snapshot.manifest.bm25_num_tokens,
        terms: &snapshot.bm25,
    };
    let mut merged = Vec::new();
    for segment in &snapshot.segments {
        merged.extend(segment.search_lexical(tokens, limit, &stats, &snapshot.tombstones)?);
    }
    merged.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.chunk_id.cmp(&b.chunk_id)));
    merged.truncate(limit);
    Ok(merged)
}

pub(crate) fn dense_leg(snapshot: &Snapshot, query: &DenseQuery, limit: usize) -> Result<DenseHits> {
    let dead = snapshot.tombstones.clone();
    let live = move |id: u64| !dead.contains(id);
    let mut merged = Vec::new();
    let mut deepest = FallbackStep::E64;
    for (segment, live_count) in snapshot.segments.iter().zip(&snapshot.live) {
        let found = segment.vector.search(query, limit, &live, &segment.ids, *live_count as usize)?;
        deepest = deepest.max(found.step);
        merged.extend(found.hits);
    }
    merged.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    merged.truncate(limit);
    Ok(DenseHits {
        ids: merged.iter().map(|(id, _)| *id).collect(),
        distances: merged.iter().map(|(_, d)| *d).collect(),
        deepest,
        query: Some(query.raw.clone()),
    })
}

pub(crate) fn hit_for(snapshot: &Snapshot, chunk_id: u64, score: f32) -> Result<Option<Hit>> {
    let Some(segment) = snapshot.segment_of(chunk_id) else { return Ok(None) };
    Ok(segment.stored(chunk_id)?.map(|stored| Hit {
        chunk_id,
        doc_id: stored.doc_key,
        body: stored.body,
        prefix: stored.prefix,
        byte_start: stored.byte_start,
        byte_end: stored.byte_end,
        score,
    }))
}

impl Index {
    /// Prepares the query once, embeds `embed_text` once, runs both legs, fuses with RRF, and
    /// optionally reranks the top `rerank_k`.
    pub fn retrieve(&self, query: &str, opts: RetrieveOptions) -> Result<Hits> {
        let prepared = prepare(query);
        if prepared.embed_text.is_empty() {
            return Err(Error::EmptyQuery);
        }
        let guard = self.acquire();
        let snapshot = guard.snapshot();
        let leg_limit = LEG_LIMIT.max(opts.limit);

        let (lexical, dense) = std::thread::scope(|scope| {
            let lexical = scope.spawn(|| lexical_leg(snapshot, &prepared.tokens, leg_limit));
            let dense = (|| -> Result<DenseHits> {
                let vectors = self.embedder.embed(std::slice::from_ref(&prepared.embed_text))?;
                let vector =
                    vectors.into_iter().next().ok_or_else(|| Error::Embed("embedder returned nothing".into()))?;
                self.check_width(vector.len())?;
                let mut dense_query = DenseQuery::new(snapshot.manifest.quantization, &vector);
                dense_query.starve_hnsw = self.testing_starve_hnsw;
                dense_leg(snapshot, &dense_query, leg_limit)
            })();
            (lexical.join().unwrap_or_else(|_| Err(Error::Lexical("lexical leg panicked".into()))), dense)
        });
        let lexical = lexical?;
        let dense = dense?;
        let fused = match self.fusion {
            Fusion::Rrf => {
                let lexical_ids: Vec<u64> = lexical.iter().map(|hit| hit.chunk_id).collect();
                rrf_weighted(&lexical_ids, &dense.ids, self.rrf_weights)
            }
            Fusion::Score { dense_weight } => {
                let lexical: Vec<(u64, f32)> = lexical.iter().map(|hit| (hit.chunk_id, hit.score)).collect();
                let dense: Vec<(u64, f32)> =
                    dense.ids.iter().zip(&dense.distances).map(|(id, d)| (*id, 1.0 - d)).collect();
                score_fusion(&lexical, &dense, dense_weight)
            }
            Fusion::Rescore { dense_weight } => {
                let lexical: Vec<(u64, f32)> = lexical.iter().map(|hit| (hit.chunk_id, hit.score)).collect();
                let mut candidates: Vec<u64> =
                    lexical.iter().map(|(id, _)| *id).chain(dense.ids.iter().copied()).collect();
                candidates.sort_unstable();
                candidates.dedup();
                let query = dense.query.as_deref().unwrap_or_default();
                let sims: Vec<(u64, f32)> = candidates
                    .into_iter()
                    .filter_map(|id| {
                        let segment = snapshot.segment_of(id)?;
                        let stored = segment.vector.stored_vector(id, &segment.ids)?;
                        Some((id, 1.0 - crate::vector::cosine_distance(query, &stored)))
                    })
                    .collect();
                score_fusion(&lexical, &sims, dense_weight)
            }
        };

        let reranker: &dyn Reranker = if opts.rerank { self.reranker.as_ref() } else { &NoopReranker };
        let mut hits = Vec::new();
        if opts.rerank {
            let head = fused.len().min(RERANK_K.max(opts.limit));
            for (id, score) in &fused[..head] {
                if let Some(hit) = hit_for(snapshot, *id, *score)? {
                    hits.push(hit);
                }
            }
            let bodies: Vec<String> = hits.iter().map(|hit| hit.body.clone()).collect();
            let scores = reranker.rerank(query, &bodies)?;
            for (hit, score) in hits.iter_mut().zip(scores) {
                hit.score = score;
            }
            // Stable: equal scores keep the fused order.
            hits.sort_by(|a, b| b.score.total_cmp(&a.score));
            hits.truncate(opts.limit);
        } else {
            for (id, score) in fused.iter().take(opts.limit) {
                if let Some(hit) = hit_for(snapshot, *id, *score)? {
                    hits.push(hit);
                }
            }
        }
        Ok(Hits {
            chunk_ids: hits.iter().map(|hit| hit.chunk_id).collect(),
            hits,
            embed_calls: 1,
            fallback: dense.deepest,
            epoch: snapshot.manifest.epoch,
            warning: snapshot.warning(),
        })
    }

    /// The lexical leg alone, over every published segment.
    #[doc(hidden)]
    pub fn lexical_search(&self, tokens: &[Token], limit: usize) -> Result<Vec<ScoredId>> {
        lexical_leg(&self.shared.snapshot(), tokens, limit)
    }

    /// The dense leg alone for a raw query vector.
    #[doc(hidden)]
    pub fn vector_search(&self, query: &[f32], limit: usize) -> Result<DenseHits> {
        let snapshot = self.shared.snapshot();
        let mut dense_query = DenseQuery::new(snapshot.manifest.quantization, query);
        dense_query.starve_hnsw = self.testing_starve_hnsw;
        dense_leg(&snapshot, &dense_query, limit)
    }

    /// Files the dense leg keeps mapped for search.
    #[doc(hidden)]
    pub fn vector_search_mapped_paths(&self) -> Vec<PathBuf> {
        let snapshot = self.shared.snapshot();
        snapshot.segments.iter().flat_map(|segment| segment.vector.mapped_paths(&segment.dir)).collect()
    }

    /// Collection `doc_freq` of `term` from the published BM25 map, dead chunks included.
    pub fn collection_doc_freq(&self, term: &str) -> u64 {
        self.shared.snapshot().bm25.get(term)
    }
}
