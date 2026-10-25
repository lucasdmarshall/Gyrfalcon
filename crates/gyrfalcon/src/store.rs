//! Binary file codecs for the generation files and `terms.bin`, plus fsync helpers.
//!
//! Every integer is little-endian. Strings are a `u32` byte length followed by UTF-8.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::Path;

use crate::error::{Error, Result};

pub fn write_synced(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = File::create(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

/// Fsyncs a directory so a rename or a new entry inside it is durable. A no-op where
/// directories cannot be opened as files.
pub fn sync_dir(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        File::open(path)?.sync_all()?;
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
    Ok(())
}

/// Segment ids are zero-padded decimals.
pub fn segment_name(id: u64) -> String {
    format!("{id:08}")
}

pub fn parse_segment_name(name: &str) -> Option<u64> {
    if name.len() == 8 && name.bytes().all(|b| b.is_ascii_digit()) {
        name.parse().ok()
    } else {
        None
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
    what: &'a str,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8], what: &'a str) -> Self {
        Reader { bytes, at: 0, what }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.at + n > self.bytes.len() {
            return Err(Error::SegmentCorrupt(format!("{} is truncated", self.what)));
        }
        let out = &self.bytes[self.at..self.at + n];
        self.at += n;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn string(&mut self) -> Result<String> {
        let len = self.u32()? as usize;
        String::from_utf8(self.take(len)?.to_vec())
            .map_err(|_| Error::SegmentCorrupt(format!("{} holds invalid UTF-8", self.what)))
    }
}

fn put_string(out: &mut Vec<u8>, value: &str) {
    out.extend_from_slice(&(value.len() as u32).to_le_bytes());
    out.extend_from_slice(value.as_bytes());
}

/// `(term, doc_freq)` pairs: the collection statistics of a query's terms.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TermMap(pub BTreeMap<String, u64>);

impl TermMap {
    pub fn get(&self, term: &str) -> u64 {
        self.0.get(term).copied().unwrap_or(0)
    }
}

/// Sorted unique `u64` array: tombstone files and a segment's `ids.bin`.
pub fn encode_ids(ids: &[u64]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ids.len() * 8);
    for id in ids {
        out.extend_from_slice(&id.to_le_bytes());
    }
    out
}

pub fn decode_ids(bytes: &[u8]) -> Result<Vec<u64>> {
    if !bytes.len().is_multiple_of(8) {
        return Err(Error::SegmentCorrupt("id array length is not a multiple of 8".into()));
    }
    Ok(bytes.chunks_exact(8).map(|b| u64::from_le_bytes(b.try_into().unwrap())).collect())
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct CatalogRow {
    pub doc_id: String,
    pub chunk_id: u64,
    pub segment_id: u64,
    pub byte_start: u64,
    pub byte_end: u64,
}

/// Live rows sorted by `(doc_id, chunk_id)`.
#[derive(Debug, Clone, Default)]
pub struct Catalog {
    pub rows: Vec<CatalogRow>,
}

impl Catalog {
    pub fn from_rows(mut rows: Vec<CatalogRow>) -> Catalog {
        rows.sort();
        Catalog { rows }
    }

    pub fn rows_for_doc(&self, doc_id: &str) -> &[CatalogRow] {
        let start = self.rows.partition_point(|row| row.doc_id.as_str() < doc_id);
        let end = self.rows.partition_point(|row| row.doc_id.as_str() <= doc_id);
        &self.rows[start..end]
    }

    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&(self.rows.len() as u64).to_le_bytes());
        for row in &self.rows {
            put_string(&mut out, &row.doc_id);
            for value in [row.chunk_id, row.segment_id, row.byte_start, row.byte_end] {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
        out
    }

    pub fn decode(bytes: &[u8]) -> Result<Catalog> {
        let mut reader = Reader::new(bytes, "catalog");
        let count = reader.u64()?;
        let mut rows = Vec::with_capacity(count as usize);
        for _ in 0..count {
            rows.push(CatalogRow {
                doc_id: reader.string()?,
                chunk_id: reader.u64()?,
                segment_id: reader.u64()?,
                byte_start: reader.u64()?,
                byte_end: reader.u64()?,
            });
        }
        Ok(Catalog { rows })
    }
}

/// The collection tombstone bitmap: a sorted set of dead chunk ids.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Tombstones {
    ids: Vec<u64>,
}

impl Tombstones {
    pub fn from_ids(mut ids: Vec<u64>) -> Tombstones {
        ids.sort_unstable();
        ids.dedup();
        Tombstones { ids }
    }

    pub fn contains(&self, id: u64) -> bool {
        self.ids.binary_search(&id).is_ok()
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// The dead ids in `low..=high`, ascending.
    pub fn in_range(&self, low: u64, high: u64) -> &[u64] {
        let start = self.ids.partition_point(|id| *id < low);
        let end = self.ids.partition_point(|id| *id <= high);
        &self.ids[start..end.max(start)]
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn with(&self, more: &[u64]) -> Tombstones {
        let mut ids = self.ids.clone();
        ids.extend_from_slice(more);
        Tombstones::from_ids(ids)
    }

    pub fn without(&self, remove: &[u64]) -> Tombstones {
        let mut remove = remove.to_vec();
        remove.sort_unstable();
        Tombstones { ids: self.ids.iter().copied().filter(|id| remove.binary_search(id).is_err()).collect() }
    }

    pub fn encode(&self) -> Vec<u8> {
        encode_ids(&self.ids)
    }

    pub fn decode(bytes: &[u8]) -> Result<Tombstones> {
        Ok(Tombstones::from_ids(decode_ids(bytes)?))
    }
}
