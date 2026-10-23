//! Sealed segment directories.
//!
//! ```text
//! segments/{segment_id}/
//!   segment.json   identity fields and counts
//!   ids.bin        stored chunk ids, ascending (dead ones included)
//!   vectors.bin    quantized rows in the same order
//!   terms.bin      (term, doc_freq) counted while the segment was written
//!   tantivy/       Tantivy index, one document per chunk
//!   usearch.index  HNSW graph keyed by chunk id
//! ```
//!
//! After seal none of these files change.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::lexical::{LexicalIndex, TantivyLexical};
use crate::manifest::{IndexConfig, Manifest};
use crate::prepare::PreparedText;
use crate::store::{self, TermMap};
use crate::vector::{self, VectorSegment};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentMeta {
    pub model_id: String,
    pub dimension: u32,
    pub quantization: crate::Quantization,
    pub analyzer_id: String,
    pub chunk_count: u64,
    /// Stored chunk records, dead ones included. Equals `chunk_count`.
    pub num_docs: u64,
    /// Token occurrences across stored chunks, repeats included.
    pub num_tokens: u64,
}

/// One chunk on its way into a sealed segment.
#[derive(Debug, Clone)]
pub struct ChunkRecord {
    pub chunk_id: u64,
    pub doc_id: String,
    pub byte_start: u64,
    pub byte_end: u64,
    pub body: String,
    pub prefix: Option<String>,
    pub prepared: PreparedText,
    /// The quantized row exactly as stored in `vectors.bin`.
    pub vector: Vec<u8>,
}

/// A sealed segment opened for reading.
pub struct Segment {
    pub id: u64,
    pub dir: PathBuf,
    pub meta: SegmentMeta,
    /// Stored chunk ids, ascending.
    pub ids: Vec<u64>,
    pub terms: TermMap,
    pub lexical: TantivyLexical,
    pub vector: VectorSegment,
}

impl Segment {
    pub fn name(&self) -> String {
        store::segment_name(self.id)
    }

    pub fn contains(&self, chunk_id: u64) -> bool {
        self.ids.binary_search(&chunk_id).is_ok()
    }

    pub fn open(root: &Path, id: u64, manifest: &Manifest) -> Result<Segment> {
        let dir = segment_dir(root, id);
        let meta: SegmentMeta = serde_json::from_slice(&std::fs::read(dir.join("segment.json"))?)?;
        let identity_matches = meta.model_id == manifest.model_id
            && meta.dimension == manifest.dimension
            && meta.quantization == manifest.quantization
            && meta.analyzer_id == manifest.analyzer_id;
        if !identity_matches {
            return Err(Error::SegmentCorrupt(format!(
                "segment {} identity disagrees with manifest.json",
                store::segment_name(id)
            )));
        }
        let ids = store::decode_ids(&std::fs::read(dir.join("ids.bin"))?)?;
        if ids.len() as u64 != meta.chunk_count {
            return Err(Error::SegmentCorrupt(format!(
                "segment {} stores {} ids, segment.json says {}",
                store::segment_name(id),
                ids.len(),
                meta.chunk_count
            )));
        }
        let terms = TermMap::read(&dir.join("terms.bin"))?;
        let lexical = TantivyLexical::open(&dir.join("tantivy"))?;
        let vector = VectorSegment::open(&dir, meta.quantization, meta.dimension as usize)?;
        Ok(Segment { id, dir, meta, ids, terms, lexical, vector })
    }
}

pub fn segment_dir(root: &Path, id: u64) -> PathBuf {
    root.join("segments").join(store::segment_name(id))
}

/// Counts `(term, doc_freq)` and token occurrences for a set of chunks.
pub fn count_terms<'a>(chunks: impl Iterator<Item = &'a PreparedText>) -> (TermMap, u64) {
    let mut terms = TermMap::default();
    let mut num_tokens = 0u64;
    for prepared in chunks {
        num_tokens += prepared.tokens.len() as u64;
        let mut unique: Vec<&str> = prepared.tokens.iter().map(|t| t.text.as_str()).collect();
        unique.sort_unstable();
        unique.dedup();
        for term in unique {
            *terms.0.entry(term.to_string()).or_insert(0) += 1;
        }
    }
    (terms, num_tokens)
}

/// Writes every file of a new segment directory and fsyncs it. `chunks` must be sorted by
/// `chunk_id`. The directory is not referenced by any manifest until the caller publishes.
pub fn write_segment(root: &Path, id: u64, config: &IndexConfig, chunks: &[ChunkRecord]) -> Result<PathBuf> {
    debug_assert!(chunks.windows(2).all(|w| w[0].chunk_id < w[1].chunk_id));
    let dir = segment_dir(root, id);
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    std::fs::create_dir_all(&dir)?;

    let ids: Vec<u64> = chunks.iter().map(|c| c.chunk_id).collect();
    store::write_synced(&dir.join("ids.bin"), &store::encode_ids(&ids))?;

    let row_len = config.dimension as usize * config.quantization.bytes_per_component();
    let mut vectors = Vec::with_capacity(chunks.len() * row_len);
    for chunk in chunks {
        if chunk.vector.len() != row_len {
            return Err(Error::SegmentCorrupt(format!(
                "chunk {} has a {}-byte row, expected {row_len}",
                chunk.chunk_id,
                chunk.vector.len()
            )));
        }
        vectors.extend_from_slice(&chunk.vector);
    }
    store::write_synced(&dir.join("vectors.bin"), &vectors)?;

    let (terms, num_tokens) = count_terms(chunks.iter().map(|c| &c.prepared));
    store::write_synced(&dir.join("terms.bin"), &terms.encode())?;

    let meta = SegmentMeta {
        model_id: config.model_id.clone(),
        dimension: config.dimension,
        quantization: config.quantization,
        analyzer_id: config.analyzer_id.clone(),
        chunk_count: chunks.len() as u64,
        num_docs: chunks.len() as u64,
        num_tokens,
    };
    let mut lexical = TantivyLexical::create(&dir.join("tantivy"))?;
    for chunk in chunks {
        lexical.add(chunk)?;
    }
    lexical.seal()?;
    drop(lexical);
    let rows: Vec<(u64, &[u8])> = chunks.iter().map(|c| (c.chunk_id, c.vector.as_slice())).collect();
    vector::write_usearch(&dir.join("usearch.index"), config.quantization, config.dimension as usize, &rows)?;
    sync_tree(&dir)?;

    store::write_synced(&dir.join("segment.json"), &serde_json::to_vec_pretty(&meta)?)?;
    store::sync_dir(&dir)?;
    store::sync_dir(&root.join("segments"))?;
    Ok(dir)
}

/// Fsyncs every file under `dir` and the directories themselves.
fn sync_tree(dir: &Path) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            sync_tree(&entry.path())?;
        } else {
            std::fs::File::open(entry.path())?.sync_all()?;
        }
    }
    store::sync_dir(dir)
}
