//! Sealed segment directories.
//!
//! ```text
//! segments/{segment_id}/
//!   segment.json   identity fields and counts
//!   ids.bin        stored chunk ids, ascending (dead ones included)
//!   keys.bin       chunk ids sorted by caller doc_id, for delete and update
//!   vectors.bin    quantized rows in the same order
//!   lexical/       BM25 index and stored chunk fields, doc ordinals in ids.bin order
//!   codes.bin      rotated sign bits per row for the dense cascade
//! ```
//!
//! After seal none of these files change.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::keys::DocKeys;
use crate::lexical::{self, CollectionStats, LexicalSegment, ScoredId, StoredChunk};
use crate::manifest::{IndexConfig, Manifest};
use crate::prepare::{PreparedText, Token};
use crate::store::{self, Tombstones};
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
    pub keys: DocKeys,
    pub lexical: LexicalSegment,
    pub vector: VectorSegment,
}

impl Segment {
    pub fn name(&self) -> String {
        store::segment_name(self.id)
    }

    pub fn contains(&self, chunk_id: u64) -> bool {
        self.ids.binary_search(&chunk_id).is_ok()
    }

    /// The best `limit` live chunks of this segment by collection BM25.
    pub fn search_lexical(
        &self,
        tokens: &[Token],
        limit: usize,
        stats: &CollectionStats,
        dead: &Tombstones,
    ) -> Result<Vec<ScoredId>> {
        self.lexical.search(tokens, limit, stats, dead, &self.ids)
    }

    /// The stored fields of `chunk_id`, or `None` when this segment does not hold it.
    pub fn stored(&self, chunk_id: u64) -> Result<Option<StoredChunk>> {
        let Ok(doc) = self.ids.binary_search(&chunk_id) else { return Ok(None) };
        let stored = self.lexical.stored(doc as u32)?;
        if stored.chunk_id != chunk_id {
            return Err(Error::SegmentCorrupt(format!(
                "segment {} stores chunk {} where ids.bin has {chunk_id}",
                self.name(),
                stored.chunk_id
            )));
        }
        Ok(Some(stored))
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
        let keys = DocKeys::open(&dir.join("keys.bin"))?;
        let lexical = LexicalSegment::open(&dir.join("lexical"))?;
        if lexical.num_docs() as usize != ids.len() {
            return Err(Error::SegmentCorrupt(format!(
                "segment {} lexical index holds {} docs, ids.bin {}",
                store::segment_name(id),
                lexical.num_docs(),
                ids.len()
            )));
        }
        let vector = VectorSegment::open(&dir, meta.quantization, meta.dimension as usize)?;
        Ok(Segment { id, dir, meta, ids, keys, lexical, vector })
    }
}

pub fn segment_dir(root: &Path, id: u64) -> PathBuf {
    root.join("segments").join(store::segment_name(id))
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

    let row_len = config.dimension as usize * config.quantization.bytes_per_component();
    if let Some(chunk) = chunks.iter().find(|c| c.vector.len() != row_len) {
        return Err(Error::SegmentCorrupt(format!(
            "chunk {} has a {}-byte row, expected {row_len}",
            chunk.chunk_id,
            chunk.vector.len()
        )));
    }

    // The lexical index and the dense codes build at the same time as the small files.
    let built = std::thread::scope(|scope| -> Result<lexical::Built> {
        let lexical = scope.spawn(|| lexical::write(&dir.join("lexical"), chunks));
        let codes = scope.spawn(|| {
            let rows: Vec<&[u8]> = chunks.iter().map(|c| c.vector.as_slice()).collect();
            let codes = vector::encode_codes(config.quantization, config.dimension as usize, &rows);
            std::fs::write(dir.join("codes.bin"), codes)
        });

        let ids: Vec<u64> = chunks.iter().map(|c| c.chunk_id).collect();
        store::write_synced(&dir.join("ids.bin"), &store::encode_ids(&ids))?;
        let pairs: Vec<(&str, u64)> = chunks.iter().map(|c| (c.doc_id.as_str(), c.chunk_id)).collect();
        store::write_synced(&dir.join("keys.bin"), &crate::keys::encode(&pairs))?;
        let mut vectors = Vec::with_capacity(chunks.len() * row_len);
        for chunk in chunks {
            vectors.extend_from_slice(&chunk.vector);
        }
        store::write_synced(&dir.join("vectors.bin"), &vectors)?;

        let built = lexical.join().map_err(|_| Error::Lexical("lexical build panicked".into()))??;
        codes.join().map_err(|_| Error::Vector("codes build panicked".into()))??;
        Ok(built)
    })?;
    let meta = SegmentMeta {
        model_id: config.model_id.clone(),
        dimension: config.dimension,
        quantization: config.quantization,
        analyzer_id: config.analyzer_id.clone(),
        chunk_count: chunks.len() as u64,
        num_docs: chunks.len() as u64,
        num_tokens: built.num_tokens,
    };
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
