//! Stored chunk fields, in LZ4 blocks of about 16 KiB.
//!
//! ```text
//! magic "GYRDOCS1", num_docs u32, num_blocks u32
//! index   per block: first_doc u32, offset u64, raw_len u32, packed_len u32
//! blocks  LZ4 block format, one after another
//! ```
//!
//! A record is `chunk_id u64, byte_start u64, byte_end u64`, the lengths of `doc_key`, `body`,
//! `prefix` (`u32::MAX` for none), and `metadata` as u32, then those bytes. Metadata is a pair
//! count u32, then each key and value as a u32 length and its bytes.

use std::borrow::Borrow;
use std::path::Path;

use memmap2::Mmap;

use super::StoredChunk;
use crate::error::{Error, Result};
use crate::segment::ChunkRecord;

const MAGIC: &[u8; 8] = b"GYRDOCS2";
const HEADER: usize = 16;
const INDEX_ENTRY: usize = 20;
const BLOCK_BYTES: usize = 16 * 1024;
const NO_PREFIX: u32 = u32::MAX;

fn corrupt(what: &str) -> Error {
    Error::SegmentCorrupt(format!("lexical docs: {what}"))
}

fn push_record(chunk: &ChunkRecord, out: &mut Vec<u8>) {
    out.extend_from_slice(&chunk.chunk_id.to_le_bytes());
    out.extend_from_slice(&chunk.byte_start.to_le_bytes());
    out.extend_from_slice(&chunk.byte_end.to_le_bytes());
    out.extend_from_slice(&(chunk.doc_id.len() as u32).to_le_bytes());
    out.extend_from_slice(&(chunk.body.len() as u32).to_le_bytes());
    let prefix_len = chunk.prefix.as_ref().map_or(NO_PREFIX, |p| p.len() as u32);
    out.extend_from_slice(&prefix_len.to_le_bytes());
    let mut metadata = Vec::new();
    if !chunk.metadata.is_empty() {
        metadata.extend_from_slice(&(chunk.metadata.len() as u32).to_le_bytes());
        for (key, value) in &chunk.metadata {
            for text in [key, value] {
                metadata.extend_from_slice(&(text.len() as u32).to_le_bytes());
                metadata.extend_from_slice(text.as_bytes());
            }
        }
    }
    out.extend_from_slice(&(metadata.len() as u32).to_le_bytes());
    out.extend_from_slice(chunk.doc_id.as_bytes());
    out.extend_from_slice(chunk.body.as_bytes());
    if let Some(prefix) = &chunk.prefix {
        out.extend_from_slice(prefix.as_bytes());
    }
    out.extend_from_slice(&metadata);
}

/// Encodes every chunk, in order, as one `docs.bin`.
pub fn encode<R: Borrow<ChunkRecord>>(chunks: &[R]) -> Vec<u8> {
    // (first_doc, raw_len, packed bytes) per block.
    let mut blocks: Vec<(u32, u32, Vec<u8>)> = Vec::new();
    let mut raw = Vec::with_capacity(2 * BLOCK_BYTES);
    let mut first = 0u32;
    let mut flush = |raw: &mut Vec<u8>, first: u32| {
        blocks.push((first, raw.len() as u32, lz4_flex::block::compress(raw)));
        raw.clear();
    };
    for (doc, chunk) in chunks.iter().enumerate() {
        if raw.is_empty() {
            first = doc as u32;
        }
        push_record(chunk.borrow(), &mut raw);
        if raw.len() >= BLOCK_BYTES {
            flush(&mut raw, first);
        }
    }
    if !raw.is_empty() {
        flush(&mut raw, first);
    }
    let index_len = blocks.len() * INDEX_ENTRY;
    let packed_len: usize = blocks.iter().map(|(_, _, packed)| packed.len()).sum();
    let mut out = Vec::with_capacity(HEADER + index_len + packed_len);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
    out.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
    let mut offset = (HEADER + index_len) as u64;
    for (first, raw_len, packed) in &blocks {
        out.extend_from_slice(&first.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&raw_len.to_le_bytes());
        out.extend_from_slice(&(packed.len() as u32).to_le_bytes());
        offset += packed.len() as u64;
    }
    for (_, _, packed) in blocks {
        out.extend_from_slice(&packed);
    }
    out
}

/// `docs.bin`, memory-mapped.
pub struct DocStore {
    map: Mmap,
    num_docs: u32,
    num_blocks: usize,
}

#[derive(Clone, Copy)]
struct BlockEntry {
    first_doc: u32,
    offset: usize,
    raw_len: usize,
    packed_len: usize,
}

impl DocStore {
    pub fn open(path: &Path) -> Result<DocStore> {
        let map = super::map_file(path, MAGIC)?;
        if map.len() < HEADER {
            return Err(corrupt("header is truncated"));
        }
        let num_docs = u32::from_le_bytes(map[8..12].try_into().unwrap());
        let num_blocks = u32::from_le_bytes(map[12..16].try_into().unwrap()) as usize;
        if map.len() < HEADER + num_blocks * INDEX_ENTRY {
            return Err(corrupt("block index is truncated"));
        }
        let store = DocStore { map, num_docs, num_blocks };
        let mut previous: Option<u32> = None;
        for block in 0..num_blocks {
            let entry = store.entry(block);
            let ordered = previous.map_or(entry.first_doc == 0, |p| entry.first_doc > p);
            let in_file = entry.offset.checked_add(entry.packed_len).is_some_and(|end| end <= store.map.len());
            // LZ4 cannot expand input by more than 255 times; a larger raw length is corrupt.
            let plausible = entry.raw_len <= entry.packed_len.saturating_mul(255) + 16;
            if !ordered || entry.first_doc >= num_docs || !in_file || !plausible {
                return Err(corrupt("block index is inconsistent"));
            }
            previous = Some(entry.first_doc);
        }
        if num_docs > 0 && num_blocks == 0 {
            return Err(corrupt("documents without blocks"));
        }
        Ok(store)
    }

    pub fn num_docs(&self) -> u32 {
        self.num_docs
    }

    fn entry(&self, block: usize) -> BlockEntry {
        let at = HEADER + block * INDEX_ENTRY;
        let bytes = &self.map[at..at + INDEX_ENTRY];
        let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        BlockEntry {
            first_doc: u32_at(0) as u32,
            offset: u64::from_le_bytes(bytes[4..12].try_into().unwrap()) as usize,
            raw_len: u32_at(12),
            packed_len: u32_at(16),
        }
    }

    fn block(&self, entry: BlockEntry) -> Result<Vec<u8>> {
        let packed = &self.map[entry.offset..entry.offset + entry.packed_len];
        let raw = lz4_flex::block::decompress(packed, entry.raw_len).map_err(|e| corrupt(&e.to_string()))?;
        if raw.len() != entry.raw_len {
            return Err(corrupt("block length disagrees with the index"));
        }
        Ok(raw)
    }

    /// The record of doc ordinal `doc`.
    pub fn get(&self, doc: u32) -> Result<StoredChunk> {
        if doc >= self.num_docs {
            return Err(corrupt("doc ordinal out of range"));
        }
        // The last block whose first doc is at most `doc`.
        let (mut lo, mut hi) = (0, self.num_blocks);
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.entry(mid).first_doc <= doc {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let entry = self.entry(lo);
        let raw = self.block(entry)?;
        let mut at = 0;
        for _ in entry.first_doc..doc {
            skip_record(&raw, &mut at)?;
        }
        read_record(&raw, &mut at)
    }

    /// Every record, in doc order.
    pub fn all(&self) -> Result<Vec<StoredChunk>> {
        let mut out = Vec::with_capacity(self.num_docs as usize);
        for block in 0..self.num_blocks {
            let raw = self.block(self.entry(block))?;
            let mut at = 0;
            while at < raw.len() {
                out.push(read_record(&raw, &mut at)?);
            }
        }
        if out.len() != self.num_docs as usize {
            return Err(corrupt("record count disagrees with the header"));
        }
        Ok(out)
    }
}

fn take<'a>(raw: &'a [u8], at: &mut usize, len: usize) -> Result<&'a [u8]> {
    let end = at.checked_add(len).filter(|end| *end <= raw.len()).ok_or_else(|| corrupt("record is truncated"))?;
    let bytes = &raw[*at..end];
    *at = end;
    Ok(bytes)
}

struct Head {
    chunk_id: u64,
    byte_start: u64,
    byte_end: u64,
    key_len: usize,
    body_len: usize,
    prefix_len: Option<usize>,
    metadata_len: usize,
}

fn read_head(raw: &[u8], at: &mut usize) -> Result<Head> {
    let bytes = take(raw, at, 40)?;
    let u64_at = |i: usize| u64::from_le_bytes(bytes[i..i + 8].try_into().unwrap());
    let u32_at = |i: usize| u32::from_le_bytes(bytes[i..i + 4].try_into().unwrap());
    let prefix = u32_at(32);
    Ok(Head {
        chunk_id: u64_at(0),
        byte_start: u64_at(8),
        byte_end: u64_at(16),
        key_len: u32_at(24) as usize,
        body_len: u32_at(28) as usize,
        prefix_len: (prefix != NO_PREFIX).then_some(prefix as usize),
        metadata_len: u32_at(36) as usize,
    })
}

fn skip_record(raw: &[u8], at: &mut usize) -> Result<()> {
    let head = read_head(raw, at)?;
    take(raw, at, head.key_len + head.body_len + head.prefix_len.unwrap_or(0) + head.metadata_len).map(|_| ())
}

fn read_record(raw: &[u8], at: &mut usize) -> Result<StoredChunk> {
    let head = read_head(raw, at)?;
    let text = |at: &mut usize, len: usize| -> Result<String> {
        String::from_utf8(take(raw, at, len)?.to_vec()).map_err(|_| corrupt("text is not UTF-8"))
    };
    let doc_key = text(at, head.key_len)?;
    let body = text(at, head.body_len)?;
    let prefix = head.prefix_len.map(|len| text(at, len)).transpose()?;
    let blob = take(raw, at, head.metadata_len)?;
    let mut metadata = Vec::new();
    if !blob.is_empty() {
        let mut inner = 0;
        let count = u32::from_le_bytes(take(blob, &mut inner, 4)?.try_into().unwrap());
        let field = |inner: &mut usize| -> Result<String> {
            let len = u32::from_le_bytes(take(blob, inner, 4)?.try_into().unwrap()) as usize;
            String::from_utf8(take(blob, inner, len)?.to_vec()).map_err(|_| corrupt("metadata is not UTF-8"))
        };
        for _ in 0..count {
            let key = field(&mut inner)?;
            metadata.push((key, field(&mut inner)?));
        }
        if inner != blob.len() {
            return Err(corrupt("metadata length disagrees with its pairs"));
        }
    }
    Ok(StoredChunk {
        chunk_id: head.chunk_id,
        doc_key,
        body,
        prefix,
        metadata,
        byte_start: head.byte_start,
        byte_end: head.byte_end,
    })
}
