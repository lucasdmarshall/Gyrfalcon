//! Small file codecs (tombstone generations, `ids.bin`) and fsync helpers. Every integer is
//! little-endian.

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

/// The chunks a search must not return: the dead ones, and under a filter every chunk outside
/// it. Searches ask `contains` per candidate; the dense cascade asks for its rows at once.
pub trait Excluded: Sync {
    fn contains(&self, id: u64) -> bool;

    /// The rows of a segment (whose ids, ascending, are `ids`) a search may return.
    fn rows(&self, ids: &[u64]) -> RowMask;
}

/// A segment's searchable rows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RowMask {
    /// Every row except these, ascending.
    Except(Vec<usize>),
    /// Only these rows, ascending.
    Only(Vec<usize>),
}

impl RowMask {
    pub fn allows(&self, row: usize) -> bool {
        match self {
            RowMask::Except(rows) => rows.binary_search(&row).is_err(),
            RowMask::Only(rows) => rows.binary_search(&row).is_ok(),
        }
    }
}

impl Excluded for Tombstones {
    fn contains(&self, id: u64) -> bool {
        Tombstones::contains(self, id)
    }

    fn rows(&self, ids: &[u64]) -> RowMask {
        let (Some(&low), Some(&high)) = (ids.first(), ids.last()) else { return RowMask::Except(Vec::new()) };
        RowMask::Except(self.in_range(low, high).iter().filter_map(|id| ids.binary_search(id).ok()).collect())
    }
}

/// A filter's chunks, less the dead ones.
pub struct Filtered<'a> {
    pub dead: &'a Tombstones,
    /// The chunk ids the filter matches, ascending.
    pub allowed: &'a [u64],
}

impl Excluded for Filtered<'_> {
    fn contains(&self, id: u64) -> bool {
        self.allowed.binary_search(&id).is_err() || self.dead.contains(id)
    }

    fn rows(&self, ids: &[u64]) -> RowMask {
        let (Some(&low), Some(&high)) = (ids.first(), ids.last()) else { return RowMask::Only(Vec::new()) };
        let start = self.allowed.partition_point(|id| *id < low);
        let end = self.allowed.partition_point(|id| *id <= high);
        RowMask::Only(
            self.allowed[start..end.max(start)]
                .iter()
                .filter(|id| !self.dead.contains(**id))
                .filter_map(|id| ids.binary_search(id).ok())
                .collect(),
        )
    }
}
