//! The write buffer: chunks written by small calls, durable in the WAL and searchable at once,
//! sealed into a segment when the buffer fills.
//!
//! A snapshot holds the buffer as immutable batches. A batch scores BM25 with the same length
//! norms and arithmetic as a sealed segment, and dense distances are exact, so a chunk ranks the
//! same before and after its seal.

use std::sync::Arc;

use rustc_hash::FxHashMap;

use crate::lexical::{self, CollectionStats, ScoredId, StoredChunk};
use crate::manifest::Quantization;
use crate::prepare::Token;
use crate::segment::ChunkRecord;
use crate::store::Excluded;
use crate::vector::{self, DenseQuery};

/// Batches combine until they hold this many chunks.
const COMBINE_BELOW: usize = 64;

/// Chunks of one or more write calls, in chunk id order.
pub struct Batch {
    /// Shared with the batches this one was combined from, so combining copies no text.
    pub records: Vec<Arc<ChunkRecord>>,
    /// `term -> (row, tf)`, rows ascending.
    postings: FxHashMap<String, Vec<(u32, u32)>>,
    norms: Vec<u8>,
    /// `|row|` of each stored vector, as a sealed segment stores it.
    row_norms: Vec<f32>,
    pub num_tokens: u64,
    quantization: Quantization,
}

impl Batch {
    pub fn new(records: Vec<Arc<ChunkRecord>>, quantization: Quantization) -> Batch {
        let mut postings: FxHashMap<String, Vec<(u32, u32)>> = FxHashMap::default();
        let mut norms = Vec::with_capacity(records.len());
        let mut row_norms = Vec::with_capacity(records.len());
        let mut num_tokens = 0;
        for (row, record) in records.iter().enumerate() {
            let terms = &record.terms;
            num_tokens += u64::from(terms.num_tokens());
            norms.push(lexical::fieldnorm_to_id(terms.num_tokens()));
            row_norms.push(vector::row_norm(quantization, &record.vector));
            for (term, tf) in terms.iter() {
                // Allocate the key only for a term this batch has not seen.
                match postings.get_mut(term) {
                    Some(list) => list.push((row as u32, tf)),
                    None => {
                        postings.insert(term.to_string(), vec![(row as u32, tf)]);
                    }
                }
            }
        }
        Batch { records, postings, norms, row_norms, num_tokens, quantization }
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn bytes(&self) -> usize {
        self.records.iter().map(|r| r.body.len() + r.vector.len()).sum()
    }

    pub fn doc_freq(&self, term: &str) -> u64 {
        self.postings.get(term).map_or(0, |p| p.len() as u64)
    }

    /// The chunk ids holding `term`.
    pub fn chunks_with<'a>(&'a self, term: &str) -> impl Iterator<Item = u64> + 'a {
        let rows = self.postings.get(term).map(Vec::as_slice).unwrap_or_default();
        rows.iter().map(|&(row, _)| self.records[row as usize].chunk_id)
    }

    pub fn find(&self, chunk_id: u64) -> Option<&ChunkRecord> {
        self.records.binary_search_by_key(&chunk_id, |r| r.chunk_id).ok().map(|at| self.records[at].as_ref())
    }

    /// Every live chunk holding a query term, scored like `LexicalSegment::search`.
    pub fn search_lexical(&self, tokens: &[Token], stats: &CollectionStats, dead: &impl Excluded) -> Vec<ScoredId> {
        if tokens.is_empty() || stats.num_docs == 0 {
            return Vec::new();
        }
        let terms = lexical::query_terms(tokens);
        let factors = lexical::length_factors(stats.num_tokens as f32 / stats.num_docs as f32);
        let mut scores = vec![0f32; self.records.len()];
        for (text, repeats) in terms {
            let Some(list) = self.postings.get(text) else { continue };
            let weight = repeats * lexical::idf(stats.terms.get(text), stats.num_docs) * (1.0 + lexical::K1);
            for &(row, tf) in list {
                let tf = tf as f32;
                scores[row as usize] += weight * (tf / (tf + factors[self.norms[row as usize] as usize]));
            }
        }
        scores
            .into_iter()
            .enumerate()
            .filter(|(row, score)| *score > 0.0 && !dead.contains(self.records[*row].chunk_id))
            .map(|(row, score)| ScoredId { chunk_id: self.records[row].chunk_id, score })
            .collect()
    }

    /// Exact cosine distances of every live chunk.
    pub fn search_dense(&self, query: &DenseQuery, dead: &impl Excluded) -> Vec<(u64, f32)> {
        let query_norm = vector::query_norm(self.quantization, query);
        self.records
            .iter()
            .zip(&self.row_norms)
            .filter(|(record, _)| !dead.contains(record.chunk_id))
            .map(|(record, norm)| {
                let distance = vector::exact_distance(self.quantization, query, query_norm, &record.vector, *norm);
                (record.chunk_id, distance)
            })
            .collect()
    }
}

/// The buffered batches of one snapshot.
#[derive(Clone, Default)]
pub struct Pending {
    pub batches: Vec<Arc<Batch>>,
}

impl Pending {
    pub fn len(&self) -> usize {
        self.batches.iter().map(|b| b.len()).sum()
    }

    pub fn bytes(&self) -> usize {
        self.batches.iter().map(|b| b.bytes()).sum()
    }

    pub fn num_tokens(&self) -> u64 {
        self.batches.iter().map(|b| b.num_tokens).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }

    /// The buffer with `batch` appended. Small batches combine like carries in a binary counter
    /// (the newest with the one before it while the newest is at least as large), until a batch
    /// holds `COMBINE_BELOW` chunks. One-chunk writes then leave O(log n) small batches plus
    /// batches of at least `COMBINE_BELOW`, and no chunk is rebuilt more than a few times.
    pub fn with(&self, batch: Batch, quantization: Quantization) -> Pending {
        let mut batches = self.batches.clone();
        batches.push(Arc::new(batch));
        while batches.len() >= 2 {
            let (older, newer) = (&batches[batches.len() - 2], &batches[batches.len() - 1]);
            if older.len() >= COMBINE_BELOW || newer.len() < older.len() {
                break;
            }
            let newer = batches.pop().unwrap();
            let older = batches.pop().unwrap();
            let records = older.records.iter().chain(&newer.records).cloned().collect();
            batches.push(Arc::new(Batch::new(records, quantization)));
        }
        Pending { batches }
    }

    pub fn records(&self) -> impl Iterator<Item = &ChunkRecord> {
        self.batches.iter().flat_map(|b| b.records.iter().map(|r| r.as_ref()))
    }

    pub fn find(&self, chunk_id: u64) -> Option<&ChunkRecord> {
        self.batches.iter().find_map(|b| b.find(chunk_id))
    }

    pub fn doc_freq(&self, term: &str) -> u64 {
        self.batches.iter().map(|b| b.doc_freq(term)).sum()
    }
}

/// A buffered chunk's stored fields, as a segment would return them.
pub fn stored(record: &ChunkRecord) -> StoredChunk {
    StoredChunk {
        chunk_id: record.chunk_id,
        doc_key: record.doc_id.clone(),
        body: record.body.clone(),
        prefix: record.prefix.clone(),
        metadata: record.metadata.clone(),
        byte_start: record.byte_start,
        byte_end: record.byte_end,
    }
}
