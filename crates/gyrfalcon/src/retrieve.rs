//! Retrieve: prepare once, fan out over every segment's two engines, merge, fuse, rerank.

use std::path::PathBuf;

use crate::error::{Error, Result};
use crate::filter::Filter;
use crate::fuse::{max_fusion, rrf_weighted, score_fusion, Fusion};
use crate::index::{Index, Snapshot};
use crate::lexical::{CollectionStats, ScoredId};
use crate::prepare::{prepare, PreparedText, Token};
use crate::rerank::{NoopReranker, Reranker, RERANK_K};
use crate::store::{Excluded, Filtered, TermMap};
use crate::vector::{Budget, DenseQuery, FallbackStep};

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
    /// The document's metadata.
    pub metadata: Vec<(String, String)>,
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
    /// Deepest fallback step any segment reached; always `E64` for the graph-free cascade.
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
    lexical_leg_excluding(snapshot, tokens, limit, snapshot.tombstones.as_ref())
}

/// The lexical leg over every chunk `excluded` does not hold.
pub(crate) fn lexical_leg_excluding(
    snapshot: &Snapshot,
    tokens: &[Token],
    limit: usize,
    excluded: &impl Excluded,
) -> Result<Vec<ScoredId>> {
    // Collection doc freqs of the query terms only, summed over the segment dictionaries
    // and the write buffer.
    let terms = TermMap(tokens.iter().map(|t| (t.text.clone(), snapshot.doc_freq(&t.text))).collect());
    let (num_docs, num_tokens) = snapshot.bm25_totals();
    let stats = CollectionStats { num_docs, num_tokens, terms: &terms };
    let mut merged = Vec::new();
    for segment in &snapshot.segments {
        merged.extend(segment.search_lexical(tokens, limit, &stats, excluded)?);
    }
    for batch in &snapshot.pending.batches {
        merged.extend(batch.search_lexical(tokens, &stats, excluded));
    }
    merged.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.chunk_id.cmp(&b.chunk_id)));
    merged.truncate(limit);
    Ok(merged)
}

pub(crate) fn dense_leg(snapshot: &Snapshot, query: &DenseQuery, limit: usize) -> Result<DenseHits> {
    dense_leg_excluding(snapshot, query, limit, snapshot.tombstones.as_ref())
}

/// The dense leg over every chunk `excluded` does not hold.
pub(crate) fn dense_leg_excluding(
    snapshot: &Snapshot,
    query: &DenseQuery,
    limit: usize,
    excluded: &impl Excluded,
) -> Result<DenseHits> {
    // Stages 1 and 2 per segment, then one exact rescore of the best estimates overall, so the
    // exact work does not grow with the segment count.
    let live: Vec<usize> = snapshot.live.iter().map(|count| *count as usize).collect();
    let budgets = Budget::split(limit, &live);
    let mut pool: Vec<(f32, usize, u32)> = Vec::new();
    for (index, (segment, budget)) in snapshot.segments.iter().zip(&budgets).enumerate() {
        let found = segment.vector.candidates(query, *budget, excluded, &segment.ids, live[index])?;
        pool.extend(found.into_iter().map(|(row, estimate)| (estimate, index, row)));
    }
    let keep = Budget::refine_total(limit, live.iter().sum());
    if pool.len() > keep {
        pool.select_nth_unstable_by(keep - 1, |a, b| b.0.total_cmp(&a.0));
        pool.truncate(keep);
    }
    pool.sort_unstable_by_key(|(_, index, row)| (*index, *row));
    let mut merged = Vec::with_capacity(pool.len());
    for group in pool.chunk_by(|a, b| a.1 == b.1) {
        let segment = &snapshot.segments[group[0].1];
        let mut rows: Vec<u32> = group.iter().map(|(_, _, row)| *row).collect();
        merged.extend(segment.vector.exact(query, &mut rows, &segment.ids));
    }
    // The write buffer is small and scanned exactly.
    for batch in &snapshot.pending.batches {
        merged.extend(batch.search_dense(query, excluded));
    }
    merged.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    merged.truncate(limit);
    Ok(DenseHits {
        ids: merged.iter().map(|(id, _)| *id).collect(),
        distances: merged.iter().map(|(_, d)| *d).collect(),
        deepest: FallbackStep::E64,
        query: Some(query.raw.clone()),
    })
}

pub(crate) fn hit_for(snapshot: &Snapshot, chunk_id: u64, score: f32) -> Result<Option<Hit>> {
    Ok(snapshot.stored(chunk_id)?.map(|stored| Hit {
        chunk_id,
        doc_id: stored.doc_key,
        body: stored.body,
        prefix: stored.prefix,
        metadata: stored.metadata,
        byte_start: stored.byte_start,
        byte_end: stored.byte_end,
        score,
    }))
}

impl Index {
    /// Prepares the query once, embeds `embed_text` once, runs both legs, fuses with RRF, and
    /// optionally reranks the top `rerank_k`.
    pub fn retrieve(&self, query: &str, opts: RetrieveOptions) -> Result<Hits> {
        self.retrieve_filtered(query, opts, None)
    }

    /// `retrieve` over the chunks whose documents match `filter` (every chunk for `None` or an
    /// empty filter).
    pub fn retrieve_filtered(&self, query: &str, opts: RetrieveOptions, filter: Option<&Filter>) -> Result<Hits> {
        let prepared = prepare(query);
        if prepared.embed_text.is_empty() {
            return Err(Error::EmptyQuery);
        }
        let guard = self.acquire();
        let snapshot = guard.snapshot();
        match filter.filter(|f| !f.is_empty()) {
            None => self.retrieve_in(snapshot, query, &prepared, opts, snapshot.tombstones.as_ref()),
            Some(filter) => {
                let allowed = snapshot.matching(filter)?;
                let excluded = Filtered { dead: &snapshot.tombstones, allowed: &allowed };
                self.retrieve_in(snapshot, query, &prepared, opts, &excluded)
            }
        }
    }

    fn retrieve_in(
        &self,
        snapshot: &Snapshot,
        query: &str,
        prepared: &PreparedText,
        opts: RetrieveOptions,
        excluded: &impl Excluded,
    ) -> Result<Hits> {
        let leg_limit = LEG_LIMIT.max(opts.limit);

        let (lexical, dense) = std::thread::scope(|scope| {
            let lexical = scope.spawn(|| lexical_leg_excluding(snapshot, &prepared.tokens, leg_limit, excluded));
            let dense = (|| -> Result<DenseHits> {
                let vectors = self.embedder.embed(std::slice::from_ref(&prepared.embed_text))?;
                let vector =
                    vectors.into_iter().next().ok_or_else(|| Error::Embed("embedder returned nothing".into()))?;
                self.check_width(vector.len())?;
                let query = DenseQuery::new(snapshot.manifest.quantization, &vector);
                dense_leg_excluding(snapshot, &query, leg_limit, excluded)
            })();
            (lexical.join().unwrap_or_else(|_| Err(Error::Lexical("lexical leg panicked".into()))), dense)
        });
        let lexical = lexical?;
        let dense = dense?;
        let lexical_scores: Vec<(u64, f32)> = lexical.iter().map(|hit| (hit.chunk_id, hit.score)).collect();
        // Exact similarity of every candidate from either leg to `query`.
        let sims_of = |query: &[f32], dense_ids: &[u64]| -> Vec<(u64, f32)> {
            let mut candidates: Vec<u64> =
                lexical_scores.iter().map(|(id, _)| *id).chain(dense_ids.iter().copied()).collect();
            candidates.sort_unstable();
            candidates.dedup();
            candidates
                .into_iter()
                .filter_map(|id| {
                    let stored = snapshot.stored_vector(id)?;
                    Some((id, 1.0 - crate::vector::cosine_distance(query, &stored)))
                })
                .collect()
        };
        let fuse = |query: &[f32], dense_ids: &[u64], dense_distances: &[f32]| match self.fusion {
            Fusion::Rrf => {
                let lexical_ids: Vec<u64> = lexical.iter().map(|hit| hit.chunk_id).collect();
                rrf_weighted(&lexical_ids, dense_ids, self.rrf_weights)
            }
            Fusion::Score { dense_weight } => {
                let dense: Vec<(u64, f32)> =
                    dense_ids.iter().zip(dense_distances).map(|(id, d)| (*id, 1.0 - d)).collect();
                score_fusion(&lexical_scores, &dense, dense_weight)
            }
            Fusion::Rescore { dense_weight } => score_fusion(&lexical_scores, &sims_of(query, dense_ids), dense_weight),
            Fusion::RescoreMax { dense_weight } => {
                max_fusion(&lexical_scores, &sims_of(query, dense_ids), dense_weight)
            }
        };
        let query_vector = dense.query.clone().unwrap_or_default();
        let fused = fuse(&query_vector, &dense.ids, &dense.distances);
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
        dense_leg(&snapshot, &DenseQuery::new(snapshot.manifest.quantization, query), limit)
    }

    /// Files the dense leg keeps mapped for search.
    #[doc(hidden)]
    pub fn vector_search_mapped_paths(&self) -> Vec<PathBuf> {
        let snapshot = self.shared.snapshot();
        snapshot.segments.iter().flat_map(|segment| segment.vector.mapped_paths()).collect()
    }

    /// Collection `doc_freq` of `word` as the analyzer indexes it (so `safety` counts the
    /// stem `safeti`), dead chunks included. A stopword has none.
    pub fn collection_doc_freq(&self, word: &str) -> u64 {
        match prepare(word).tokens.first() {
            Some(token) => self.shared.snapshot().doc_freq(&token.text),
            None => 0,
        }
    }
}
