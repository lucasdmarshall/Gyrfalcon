//! The lexical leg: Gyrfalcon's own BM25 index, one per sealed segment, scored with
//! collection-wide statistics.
//!
//! ```text
//! lexical/
//!   terms.dict     magic, num_terms u32, num_docs u32, blob_len u64,
//!                  term end offsets (u32 each), entries, term bytes (sorted)
//!                  entry: postings offset u64, doc_freq u32, max_tf u32, min_norm u8
//!   postings.bin   magic, then each term's list (see `postings`)
//!   norms.bin      magic, then one length-norm byte per doc
//!   docs.bin       stored chunk fields (see `docstore`)
//! ```
//!
//! Doc ordinals are positions in the segment's `ids.bin`. Scores are Tantivy's BM25 to the
//! bit: `k1 = 1.2`, `b = 0.75`, and the same one-byte length norm. A test checks rankings
//! against Tantivy itself, so the engine change does not move retrieval quality.

mod build;
mod docstore;
mod postings;
#[cfg(test)]
mod reference;

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::path::Path;

use memmap2::Mmap;

use crate::error::{Error, Result};
use crate::prepare::Token;
#[cfg(test)]
use crate::store::Tombstones;
use crate::store::{Excluded, TermMap};

pub use build::{write, Built};
use docstore::DocStore;
use postings::{PostingList, BLOCK};

const DICT_MAGIC: &[u8; 8] = b"GYRDICT1";
const POSTINGS_MAGIC: &[u8; 8] = b"GYRPOST1";
const NORMS_MAGIC: &[u8; 8] = b"GYRNORM1";
const DICT_HEADER: usize = 24;
const DICT_ENTRY: usize = 17;

pub(crate) const K1: f32 = 1.2;
const B: f32 = 0.75;

/// The query's distinct terms in order, each with how often the query repeats it. A term's
/// BM25 contribution is multiplied by its count, as summing a repeated clause does in Lucene
/// and Tantivy; long queries (an argument, a paragraph) rank better for it (ArguAna BM25
/// nDCG@10 0.403 to 0.482, spec "Lite fusion and query term counts").
pub(crate) fn query_terms(tokens: &[Token]) -> Vec<(&str, f32)> {
    let mut terms: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
    terms.sort_unstable();
    terms.chunk_by(|a, b| a == b).map(|run| (run[0], run.len() as f32)).collect()
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredId {
    pub chunk_id: u64,
    pub score: f32,
}

/// The fields a stored chunk keeps. `body` and `prefix` let a merge rebuild the chunk
/// without the original document.
#[derive(Debug, Clone, PartialEq)]
pub struct StoredChunk {
    pub chunk_id: u64,
    pub doc_key: String,
    pub body: String,
    pub prefix: Option<String>,
    pub metadata: Vec<(String, String)>,
    pub byte_start: u64,
    pub byte_end: u64,
}

/// Collection-wide BM25 statistics read from the published manifest and term map. Dead
/// chunks are included until a merge drops them.
pub struct CollectionStats<'a> {
    pub num_docs: u64,
    pub num_tokens: u64,
    pub terms: &'a TermMap,
}

/// Lucene's and Tantivy's one-byte length norm: exact up to 40 tokens, then 3 bits of
/// mantissa. `NORM_TABLE[id]` is the smallest length that encodes to `id`.
const NORM_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut id = 0;
    while id < 256 {
        table[id] = if id < 24 {
            id as u32
        } else {
            let b = (id - 24) as u32;
            let bits = b & 7;
            let shift = b >> 3;
            24 + if shift == 0 { bits } else { (bits | 8) << (shift - 1) }
        };
        id += 1;
    }
    table
};

pub fn fieldnorm_to_id(length: u32) -> u8 {
    NORM_TABLE.partition_point(|&smallest| smallest <= length).saturating_sub(1) as u8
}

/// Tantivy's `idf`, in `f32` like Tantivy computes it.
pub(crate) fn idf(doc_freq: u64, doc_count: u64) -> f32 {
    let doc_freq = doc_freq.min(doc_count);
    let x = ((doc_count - doc_freq) as f32 + 0.5) / (doc_freq as f32 + 0.5);
    (1.0 + x).ln()
}

/// `k1 * (1 - b + b * length / average)` for every norm byte.
pub(crate) fn length_factors(average: f32) -> [f32; 256] {
    let mut factors = [0f32; 256];
    for (id, factor) in factors.iter_mut().enumerate() {
        *factor = K1 * (1.0 - B + B * NORM_TABLE[id] as f32 / average);
    }
    factors
}

fn corrupt(what: &str) -> Error {
    Error::SegmentCorrupt(format!("lexical: {what}"))
}

/// Maps a file that must start with `magic`. A sealed segment file never changes, which is
/// what makes the mapping sound.
fn map_file(path: &Path, magic: &[u8; 8]) -> Result<Mmap> {
    let file = std::fs::File::open(path)?;
    // SAFETY: segment files are immutable after seal and are only removed once no reader
    // epoch can still use them.
    let map = unsafe { Mmap::map(&file)? };
    if map.len() < 8 || &map[..8] != magic {
        return Err(corrupt(&format!("{} has no {} header", path.display(), String::from_utf8_lossy(magic))));
    }
    Ok(map)
}

/// A term's dictionary entry.
#[derive(Debug, Clone, Copy)]
struct Entry {
    offset: usize,
    doc_freq: u32,
}

/// One segment's lexical index, memory-mapped.
pub struct LexicalSegment {
    dict: Mmap,
    num_terms: usize,
    num_docs: u32,
    entries_at: usize,
    blob_at: usize,
    postings: Mmap,
    norms: Mmap,
    docs: DocStore,
}

impl LexicalSegment {
    pub fn open(dir: &Path) -> Result<LexicalSegment> {
        let dict = map_file(&dir.join("terms.dict"), DICT_MAGIC)?;
        if dict.len() < DICT_HEADER {
            return Err(corrupt("terms.dict header is truncated"));
        }
        let num_terms = u32::from_le_bytes(dict[8..12].try_into().unwrap()) as usize;
        let num_docs = u32::from_le_bytes(dict[12..16].try_into().unwrap());
        let blob_len = u64::from_le_bytes(dict[16..24].try_into().unwrap());
        let entries_at = DICT_HEADER + 4 * num_terms;
        let blob_at = entries_at + DICT_ENTRY * num_terms;
        if dict.len() as u64 != blob_at as u64 + blob_len {
            return Err(corrupt("terms.dict size disagrees with its header"));
        }
        let postings = map_file(&dir.join("postings.bin"), POSTINGS_MAGIC)?;
        let norms = map_file(&dir.join("norms.bin"), NORMS_MAGIC)?;
        if norms.len() != 8 + num_docs as usize {
            return Err(corrupt("norms.bin length disagrees with the doc count"));
        }
        let docs = DocStore::open(&dir.join("docs.bin"))?;
        if docs.num_docs() != num_docs {
            return Err(corrupt("docs.bin doc count disagrees with terms.dict"));
        }
        let segment = LexicalSegment { dict, num_terms, num_docs, entries_at, blob_at, postings, norms, docs };
        segment.validate(blob_len as usize)?;
        Ok(segment)
    }

    /// Faults the dictionary, postings, norms, and stored fields in that order.
    pub(crate) fn warm(&self) {
        crate::warm::warm_map(&self.dict);
        crate::warm::warm_map(&self.postings);
        crate::warm::warm_map(&self.norms);
        self.docs.warm();
    }

    /// Checks the dictionary once so lookups can index it without further checks.
    fn validate(&self, blob_len: usize) -> Result<()> {
        let mut previous_end = 0;
        let mut previous_offset = 8;
        for term in 0..self.num_terms {
            let end = self.term_end(term);
            let entry = self.entry(term);
            if end < previous_end || end > blob_len {
                return Err(corrupt("term offsets are out of order"));
            }
            if entry.offset < previous_offset || entry.offset > self.postings.len() {
                return Err(corrupt("postings offsets are out of order"));
            }
            if entry.doc_freq == 0 || entry.doc_freq > self.num_docs {
                return Err(corrupt("a doc freq is out of range"));
            }
            if term > 0 && self.term(term - 1) >= self.term(term) {
                return Err(corrupt("terms are not sorted"));
            }
            previous_end = end;
            previous_offset = entry.offset;
        }
        Ok(())
    }

    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    /// Docs of this segment holding `text`, dead ones included.
    pub fn doc_freq(&self, text: &str) -> u32 {
        self.find(text).map_or(0, |term| self.entry(term).doc_freq)
    }

    /// Every term with its doc freq, for tests that need whole-segment statistics.
    #[cfg(test)]
    fn term_map(&self) -> TermMap {
        TermMap(
            (0..self.num_terms)
                .map(|t| (String::from_utf8(self.term(t).to_vec()).unwrap(), u64::from(self.entry(t).doc_freq)))
                .collect(),
        )
    }

    fn term_end(&self, term: usize) -> usize {
        let at = DICT_HEADER + 4 * term;
        u32::from_le_bytes(self.dict[at..at + 4].try_into().unwrap()) as usize
    }

    fn term(&self, term: usize) -> &[u8] {
        let start = if term == 0 { 0 } else { self.term_end(term - 1) };
        &self.dict[self.blob_at + start..self.blob_at + self.term_end(term)]
    }

    fn entry(&self, term: usize) -> Entry {
        let at = self.entries_at + DICT_ENTRY * term;
        let bytes = &self.dict[at..at + DICT_ENTRY];
        Entry {
            offset: u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize,
            doc_freq: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        }
    }

    fn find(&self, text: &str) -> Option<usize> {
        let (mut lo, mut hi) = (0, self.num_terms);
        while lo < hi {
            let mid = (lo + hi) / 2;
            match self.term(mid).cmp(text.as_bytes()) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    fn list(&self, term: usize) -> Result<PostingList<'_>> {
        let entry = self.entry(term);
        let end = if term + 1 < self.num_terms { self.entry(term + 1).offset } else { self.postings.len() };
        PostingList::new(&self.postings[entry.offset..end], entry.doc_freq)
    }

    /// The doc ordinals holding `term`, ascending, dead ones included.
    pub fn docs_with(&self, term: &str) -> Result<Vec<u32>> {
        let Some(term) = self.find(term) else { return Ok(Vec::new()) };
        let list = self.list(term)?;
        let (mut docs, mut tfs) = ([0u32; postings::BLOCK], [0u32; postings::BLOCK]);
        let mut out = Vec::with_capacity(self.entry(term).doc_freq as usize);
        for block in 0..list.blocks() {
            let len = list.decode(block, &mut docs, &mut tfs)?;
            out.extend_from_slice(&docs[..len]);
        }
        if out.iter().any(|&doc| doc >= self.num_docs) {
            return Err(corrupt("a posting names a doc past the segment"));
        }
        Ok(out)
    }

    /// `(doc ordinal, tf)` for every doc holding `text`.
    #[cfg(test)]
    fn postings(&self, text: &str) -> Vec<(u32, u32)> {
        let Some(term) = self.find(text) else { return Vec::new() };
        let list = self.list(term).unwrap();
        let (mut docs, mut tfs) = ([0; BLOCK], [0; BLOCK]);
        let mut out = Vec::new();
        for block in 0..list.blocks() {
            let len = list.decode(block, &mut docs, &mut tfs).unwrap();
            out.extend(docs[..len].iter().copied().zip(tfs[..len].iter().copied()));
        }
        out
    }

    /// The best `limit` live chunks by BM25 over the distinct query terms, sorted by score
    /// descending, then chunk id ascending. `ids` maps doc ordinals to chunk ids. A chunk
    /// `dead` holds never takes a slot.
    pub fn search(
        &self,
        tokens: &[Token],
        limit: usize,
        stats: &CollectionStats,
        dead: &impl Excluded,
        ids: &[u64],
    ) -> Result<Vec<ScoredId>> {
        if tokens.is_empty() || limit == 0 || stats.num_docs == 0 || self.num_docs == 0 {
            return Ok(Vec::new());
        }
        if ids.len() != self.num_docs as usize {
            return Err(corrupt("ids.bin and the lexical index disagree on the doc count"));
        }
        let terms = query_terms(tokens);
        let factors = length_factors(stats.num_tokens as f32 / stats.num_docs as f32);
        let norms = &self.norms[8..];

        // Term at a time into one accumulator per doc. Every term adds a positive score, so
        // a zero slot is an untouched doc.
        let mut scores = vec![0f32; self.num_docs as usize];
        let mut touched: Vec<u32> = Vec::new();
        let (mut docs, mut tfs) = ([0u32; BLOCK], [0u32; BLOCK]);
        for (text, repeats) in terms {
            let Some(term) = self.find(text) else { continue };
            let weight = repeats * idf(stats.terms.get(text), stats.num_docs) * (1.0 + K1);
            let list = self.list(term)?;
            for block in 0..list.blocks() {
                let len = list.decode(block, &mut docs, &mut tfs)?;
                for (&doc, &tf) in docs[..len].iter().zip(&tfs[..len]) {
                    let (Some(slot), Some(&norm)) = (scores.get_mut(doc as usize), norms.get(doc as usize)) else {
                        return Err(corrupt("a posting names a doc past the segment"));
                    };
                    if *slot == 0.0 {
                        touched.push(doc);
                    }
                    let tf = tf as f32;
                    *slot += weight * (tf / (tf + factors[norm as usize]));
                }
            }
        }

        let mut heap: BinaryHeap<Reverse<Ranked>> = BinaryHeap::with_capacity(limit + 1);
        for doc in touched {
            let candidate = Ranked(ScoredId { chunk_id: ids[doc as usize], score: scores[doc as usize] });
            let competitive = heap.len() < limit || heap.peek().is_some_and(|worst| candidate > worst.0);
            if competitive && !dead.contains(candidate.0.chunk_id) {
                if heap.len() == limit {
                    heap.pop();
                }
                heap.push(Reverse(candidate));
            }
        }
        Ok(heap.into_sorted_vec().into_iter().map(|Reverse(ranked)| ranked.0).collect())
    }

    /// The stored fields of doc ordinal `doc`.
    pub fn stored(&self, doc: u32) -> Result<StoredChunk> {
        self.docs.get(doc)
    }

    /// Every stored chunk, in doc order. A merge reads its inputs this way.
    pub fn all_stored(&self) -> Result<Vec<StoredChunk>> {
        self.docs.all()
    }
}

/// Orders hits so that greater is better: higher score, then lower chunk id.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Ranked(ScoredId);

impl Eq for Ranked {}

impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.score.total_cmp(&other.0.score).then(other.0.chunk_id.cmp(&self.0.chunk_id))
    }
}
