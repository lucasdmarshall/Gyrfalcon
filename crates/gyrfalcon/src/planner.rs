//! Trace-and-replay for the adaptive retrieval planner.
//!
//! A trace runs every step any policy could use, once per query, and records the scores and
//! the wall time of each step. A policy is then evaluated by replaying traces: its ranking is
//! rebuilt from the stored scores and its latency is the sum of the steps it would run.
//! See `docs/superpowers/specs/2026-10-24-adaptive-planner.md`.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::bench::{mrr_at, ndcg_at, recall_at, Qrels};
use crate::error::{Error, Result};
use crate::fuse::score_fusion;
use crate::index::Index;
use crate::prepare::prepare;
use crate::rerank::Reranker;
use crate::retrieve::{dense_leg, hit_for, lexical_leg};
use crate::vector::{cosine_distance, DenseQuery};

/// Dense weights whose fused top `k` is reranked in a trace, so every replayed policy that
/// uses one of them finds its candidates scored.
pub const TRACE_RERANK_WEIGHTS: [f32; 3] = [0.2, 0.4, 0.6];

/// Everything one query needs for replay. Ids are document keys.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct QueryTrace {
    pub dataset: String,
    pub split: String,
    pub query_id: String,
    pub qrels: HashMap<String, i64>,
    pub features: QueryFeatures,
    /// BM25 top `depth` as `(doc, score)`, best first.
    pub lexical: Vec<(String, f32)>,
    /// Dense top `depth` as `(doc, cosine)`, best first.
    pub dense: Vec<(String, f32)>,
    /// Exact cosine of every document in either list.
    pub exact: HashMap<String, f32>,
    /// Cross-encoder score of documents in the reranked candidate set.
    pub rerank: HashMap<String, f32>,
    /// Chunk id of each document, so replay breaks score ties the way live fusion does.
    pub chunks: HashMap<String, u64>,
    pub timings: StepTimings,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct QueryFeatures {
    pub query_tokens: usize,
    /// Highest BM25 IDF among the query terms, from the collection statistics.
    pub idf_max: f32,
    /// A token holds a digit or an underscore, or the raw query has `::` or an inner capital.
    pub identifier: bool,
}

/// Wall time of each step in milliseconds, measured single-flight.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct StepTimings {
    pub prepare: f64,
    pub lexical: f64,
    pub embed: f64,
    pub dense: f64,
    /// Exact cosines for the union of both lists.
    pub exact: f64,
    /// Cross-encoder cost per scored pair, at the trace's token budget.
    pub rerank_per_pair: f64,
    /// Reading one returned hit's stored fields.
    #[serde(default)]
    pub fetch_per_hit: f64,
    /// Fixed cost of a retrieve call outside the steps above (guard, thread, fusion).
    #[serde(default)]
    pub overhead: f64,
}

fn ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1000.0
}

fn bm25_idf(num_docs: u64, doc_freq: u64) -> f32 {
    let (n, df) = (num_docs as f32, doc_freq as f32);
    (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
}

fn is_identifier(query: &str, tokens: &[crate::prepare::Token]) -> bool {
    tokens.iter().any(|t| t.text.chars().any(|c| c.is_ascii_digit() || c == '_'))
        || query.contains("::")
        || query.split_whitespace().any(|word| word.chars().skip(1).any(char::is_uppercase))
}

impl Index {
    /// Runs every step of a planned retrieve for `query` and records scores and timings.
    /// `rerank` scores the union of the fused top `k` under each of `TRACE_RERANK_WEIGHTS`.
    pub fn trace_query(&self, query: &str, depth: usize, rerank: Option<(&dyn Reranker, usize)>) -> Result<QueryTrace> {
        let guard = self.acquire();
        let snapshot = guard.snapshot();
        let mut timings = StepTimings::default();

        let started = Instant::now();
        let prepared = prepare(query);
        timings.prepare = ms(started);
        if prepared.embed_text.is_empty() {
            return Err(Error::EmptyQuery);
        }

        let started = Instant::now();
        let lexical = lexical_leg(snapshot, &prepared.tokens, depth)?;
        timings.lexical = ms(started);

        let started = Instant::now();
        let vector =
            self.embedder.embed(std::slice::from_ref(&prepared.embed_text))?.into_iter().next().unwrap_or_default();
        self.check_width(vector.len())?;
        timings.embed = ms(started);

        let started = Instant::now();
        let dense = dense_leg(snapshot, &DenseQuery::new(snapshot.manifest.quantization, &vector), depth)?;
        timings.dense = ms(started);

        let started = Instant::now();
        let mut candidates: Vec<u64> = lexical.iter().map(|h| h.chunk_id).chain(dense.ids.iter().copied()).collect();
        candidates.sort_unstable();
        candidates.dedup();
        let exact_by_chunk: HashMap<u64, f32> = candidates
            .iter()
            .filter_map(|id| {
                let segment = snapshot.segment_of(*id)?;
                let stored = segment.vector.stored_vector(*id, &segment.ids)?;
                Some((*id, 1.0 - cosine_distance(&vector, &stored)))
            })
            .collect();
        timings.exact = ms(started);

        // Map chunk ids to document keys; this is also the stored-field read a retrieve does
        // for each hit it returns.
        let started = Instant::now();
        let mut doc_of: HashMap<u64, (String, String)> = HashMap::new();
        for id in &candidates {
            if let Some(hit) = hit_for(snapshot, *id, 0.0)? {
                doc_of.insert(*id, (hit.doc_id, hit.body));
            }
        }
        timings.fetch_per_hit = ms(started) / candidates.len().max(1) as f64;
        let doc = |id: &u64| doc_of.get(id).map(|(doc, _)| doc.clone()).unwrap_or_default();

        let mut rerank_scores = HashMap::new();
        if let Some((reranker, k)) = rerank {
            let lexical_scores: Vec<(u64, f32)> = lexical.iter().map(|h| (h.chunk_id, h.score)).collect();
            let exact: Vec<(u64, f32)> = exact_by_chunk.iter().map(|(id, s)| (*id, *s)).collect();
            let mut pool: Vec<u64> = Vec::new();
            let mut seen = HashSet::new();
            for weight in TRACE_RERANK_WEIGHTS {
                for (id, _) in score_fusion(&lexical_scores, &exact, weight).into_iter().take(k) {
                    if seen.insert(id) {
                        pool.push(id);
                    }
                }
            }
            let bodies: Vec<String> =
                pool.iter().map(|id| doc_of.get(id).map(|(_, b)| b.clone()).unwrap_or_default()).collect();
            let started = Instant::now();
            let scores = reranker.rerank(query, &bodies)?;
            timings.rerank_per_pair = ms(started) / pool.len().max(1) as f64;
            for (id, score) in pool.iter().zip(scores) {
                rerank_scores.insert(doc(id), score);
            }
        }

        let idf_max = prepared
            .tokens
            .iter()
            .map(|t| bm25_idf(snapshot.manifest.bm25_num_docs, snapshot.bm25.get(&t.text)))
            .fold(0.0, f32::max);
        Ok(QueryTrace {
            features: QueryFeatures {
                query_tokens: prepared.tokens.len(),
                idf_max,
                identifier: is_identifier(query, &prepared.tokens),
            },
            lexical: lexical.iter().map(|h| (doc(&h.chunk_id), h.score)).collect(),
            dense: dense.ids.iter().zip(&dense.distances).map(|(id, d)| (doc(id), 1.0 - d)).collect(),
            exact: exact_by_chunk.iter().map(|(id, s)| (doc(id), *s)).collect(),
            rerank: rerank_scores,
            chunks: candidates.iter().map(|id| (doc(id), *id)).collect(),
            timings,
            ..QueryTrace::default()
        })
    }
}

/// A retrieval policy that replay can evaluate from a trace.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Policy {
    Lexical,
    Dense,
    Rrf,
    /// Rescore fusion with this dense weight, then optionally rerank the top `k`.
    Rescore {
        dense_weight: f32,
        rerank_k: usize,
    },
}

impl Policy {
    pub fn parse(spec: &str) -> Option<Policy> {
        let mut parts = spec.split(':');
        match parts.next()? {
            "lexical" => Some(Policy::Lexical),
            "dense" => Some(Policy::Dense),
            "rrf" => Some(Policy::Rrf),
            "rescore" => {
                let dense_weight = parts.next()?.parse().ok()?;
                let rerank_k = parts.next().map(|k| k.parse().ok()).unwrap_or(Some(0))?;
                Some(Policy::Rescore { dense_weight, rerank_k })
            }
            _ => None,
        }
    }
}

/// Ranked document keys a policy returns, and the milliseconds its steps cost.
pub fn replay(trace: &QueryTrace, policy: Policy, limit: usize) -> (Vec<String>, f64) {
    let t = &trace.timings;
    let (ranked, cost): (Vec<String>, f64) = match policy {
        Policy::Lexical => (trace.lexical.iter().map(|(d, _)| d.clone()).collect(), t.prepare + t.lexical),
        Policy::Dense => (trace.dense.iter().map(|(d, _)| d.clone()).collect(), t.prepare + t.embed + t.dense),
        Policy::Rrf => {
            let mut scores: HashMap<&str, f32> = HashMap::new();
            for (rank, (d, _)) in trace.lexical.iter().enumerate() {
                *scores.entry(d).or_default() += 1.0 / (61.0 + rank as f32);
            }
            for (rank, (d, _)) in trace.dense.iter().enumerate() {
                *scores.entry(d).or_default() += 1.0 / (61.0 + rank as f32);
            }
            let mut ranked: Vec<(&str, f32)> = scores.into_iter().collect();
            ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(b.0)));
            (ranked.into_iter().map(|(d, _)| d.to_string()).collect(), t.prepare + t.lexical + t.embed + t.dense)
        }
        Policy::Rescore { dense_weight, rerank_k } => {
            let mut ranked = rescore_order(trace, dense_weight);
            let mut cost = t.prepare + t.lexical + t.embed + t.dense + t.exact;
            if rerank_k > 0 {
                let k = rerank_k.min(ranked.len());
                let mut head: Vec<String> = ranked.drain(..k).collect();
                // Stable: unscored documents and ties keep the fused order.
                head.sort_by(|a, b| {
                    let score = |d: &String| trace.rerank.get(d).copied().unwrap_or(f32::NEG_INFINITY);
                    score(b).total_cmp(&score(a))
                });
                head.extend(ranked);
                ranked = head;
                cost += k as f64 * t.rerank_per_pair;
            }
            (ranked, cost)
        }
    };
    let mut ranked = crate::bench::dedupe_doc_keys(&ranked);
    let cost = cost + t.overhead + ranked.len().min(limit) as f64 * t.fetch_per_hit;
    // BEIR and MTEB ignore a retrieved document whose id is the query id (ArguAna).
    ranked.retain(|d| *d != trace.query_id);
    ranked.truncate(limit);
    (ranked, cost)
}

/// The live `Fusion::Rescore` order: document keys sorted by mixed normalized scores.
fn rescore_order(trace: &QueryTrace, dense_weight: f32) -> Vec<String> {
    let mut ids: Vec<&str> =
        trace.lexical.iter().map(|(d, _)| d.as_str()).chain(trace.dense.iter().map(|(d, _)| d.as_str())).collect();
    ids.sort_unstable();
    ids.dedup();
    let chunk = |d: &str| trace.chunks.get(d).copied().unwrap_or(u64::MAX);
    let doc_of: HashMap<u64, &str> = ids.iter().map(|d| (chunk(d), *d)).collect();
    let lexical: Vec<(u64, f32)> = trace.lexical.iter().map(|(d, s)| (chunk(d), *s)).collect();
    let exact: Vec<(u64, f32)> = ids.iter().filter_map(|d| trace.exact.get(*d).map(|s| (chunk(d), *s))).collect();
    score_fusion(&lexical, &exact, dense_weight)
        .into_iter()
        .filter_map(|(id, _)| doc_of.get(&id).map(|d| d.to_string()))
        .collect()
}

/// Per-query quality of one ranking.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct QueryScore {
    pub ndcg_at_10: f64,
    pub recall_at_10: f64,
    pub recall_at_100: f64,
    pub mrr_at_10: f64,
    pub cost_ms: f64,
}

pub fn score(trace: &QueryTrace, policy: Policy) -> QueryScore {
    let (ranked, cost_ms) = replay(trace, policy, 100);
    let qrels: &Qrels = &trace.qrels;
    QueryScore {
        ndcg_at_10: ndcg_at(&ranked, qrels, 10),
        recall_at_10: recall_at(&ranked, qrels, 10),
        recall_at_100: recall_at(&ranked, qrels, 100),
        mrr_at_10: mrr_at(&ranked, qrels, 10),
        cost_ms,
    }
}

/// Means over queries plus latency percentiles of the replayed cost.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PolicySummary {
    pub queries: usize,
    pub ndcg_at_10: f64,
    pub recall_at_10: f64,
    pub recall_at_100: f64,
    pub mrr_at_10: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
}

pub fn summarize(scores: &[QueryScore]) -> PolicySummary {
    let n = scores.len().max(1) as f64;
    let mut costs: Vec<f64> = scores.iter().map(|s| s.cost_ms).collect();
    costs.sort_by(|a, b| a.total_cmp(b));
    PolicySummary {
        queries: scores.len(),
        ndcg_at_10: scores.iter().map(|s| s.ndcg_at_10).sum::<f64>() / n,
        recall_at_10: scores.iter().map(|s| s.recall_at_10).sum::<f64>() / n,
        recall_at_100: scores.iter().map(|s| s.recall_at_100).sum::<f64>() / n,
        mrr_at_10: scores.iter().map(|s| s.mrr_at_10).sum::<f64>() / n,
        p50_ms: crate::bench::percentile(&costs, 50.0),
        p95_ms: crate::bench::percentile(&costs, 95.0),
        p99_ms: crate::bench::percentile(&costs, 99.0),
    }
}

/// Paired bootstrap of the mean difference `a - b`: `(mean, low, high)` of a 95% interval.
pub fn paired_bootstrap(a: &[f64], b: &[f64], resamples: usize, seed: u64) -> (f64, f64, f64) {
    let diffs: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    let n = diffs.len();
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let mean = diffs.iter().sum::<f64>() / n as f64;
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut means: Vec<f64> =
        (0..resamples).map(|_| (0..n).map(|_| diffs[(next() % n as u64) as usize]).sum::<f64>() / n as f64).collect();
    means.sort_by(|x, y| x.total_cmp(y));
    let at = |p: f64| means[((p * resamples as f64) as usize).min(resamples - 1)];
    (mean, at(0.025), at(0.975))
}

/// The protocol's hash split for datasets without an official validation split: the first
/// byte of `sha256(query_id)` even means validation.
pub fn hash_split(query_id: &str) -> &'static str {
    use sha2::Digest;
    if sha2::Sha256::digest(query_id.as_bytes())[0] % 2 == 0 {
        "validation"
    } else {
        "test"
    }
}
