//! A segment's `keys.bin`: its chunks sorted by caller `doc_id`, so a delete or an update finds
//! a document's chunks without a collection-wide catalog.
//!
//! ```text
//! magic "GYRKEYS1", count u32, blob_len u32
//! chunk ids    u64 per entry, sorted by (doc_id, chunk_id)
//! key ends     u32 per entry, end offset of the entry's doc_id in the blob
//! blob         the doc_id bytes, back to back
//! ```

use std::path::Path;

use memmap2::Mmap;

use crate::error::{Error, Result};

const MAGIC: &[u8; 8] = b"GYRKEYS1";
const HEADER: usize = 16;

fn corrupt(what: &str) -> Error {
    Error::SegmentCorrupt(format!("keys.bin: {what}"))
}

/// Encodes `(doc_id, chunk_id)` pairs in any order.
pub fn encode(pairs: &[(&str, u64)]) -> Vec<u8> {
    let mut sorted = pairs.to_vec();
    sorted.sort_unstable();
    let blob_len: usize = sorted.iter().map(|(key, _)| key.len()).sum();
    let mut out = Vec::with_capacity(HEADER + sorted.len() * 12 + blob_len);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&(sorted.len() as u32).to_le_bytes());
    out.extend_from_slice(&(blob_len as u32).to_le_bytes());
    for (_, chunk_id) in &sorted {
        out.extend_from_slice(&chunk_id.to_le_bytes());
    }
    let mut end = 0u32;
    for (key, _) in &sorted {
        end += key.len() as u32;
        out.extend_from_slice(&end.to_le_bytes());
    }
    for (key, _) in &sorted {
        out.extend_from_slice(key.as_bytes());
    }
    out
}

/// `keys.bin`, memory-mapped.
pub struct DocKeys {
    map: Mmap,
    count: usize,
}

impl DocKeys {
    pub fn open(path: &Path) -> Result<DocKeys> {
        let file = std::fs::File::open(path)?;
        // SAFETY: sealed segment files never change; the reaper waits for readers to release.
        let map = unsafe { Mmap::map(&file)? };
        if map.len() < HEADER || &map[..8] != MAGIC {
            return Err(corrupt("no header"));
        }
        let count = u32::from_le_bytes(map[8..12].try_into().unwrap()) as usize;
        let blob_len = u32::from_le_bytes(map[12..16].try_into().unwrap()) as usize;
        if map.len() != HEADER + count * 12 + blob_len {
            return Err(corrupt("length disagrees with the header"));
        }
        let keys = DocKeys { map, count };
        let mut previous_end = 0;
        for at in 0..count {
            let end = keys.end(at);
            if end < previous_end || end > blob_len {
                return Err(corrupt("key offsets are out of order"));
            }
            if at > 0 && (keys.key(at - 1), keys.chunk_id(at - 1)) >= (keys.key(at), keys.chunk_id(at)) {
                return Err(corrupt("entries are not sorted"));
            }
            previous_end = end;
        }
        Ok(keys)
    }

    fn chunk_id(&self, at: usize) -> u64 {
        let offset = HEADER + at * 8;
        u64::from_le_bytes(self.map[offset..offset + 8].try_into().unwrap())
    }

    fn end(&self, at: usize) -> usize {
        let offset = HEADER + self.count * 8 + at * 4;
        u32::from_le_bytes(self.map[offset..offset + 4].try_into().unwrap()) as usize
    }

    fn key(&self, at: usize) -> &[u8] {
        let blob = HEADER + self.count * 12;
        let start = if at == 0 { 0 } else { self.end(at - 1) };
        &self.map[blob + start..blob + self.end(at)]
    }

    /// The stored chunk ids of `doc_id`, ascending, dead ones included.
    pub fn chunks_for(&self, doc_id: &str) -> Vec<u64> {
        let key = doc_id.as_bytes();
        let (mut lo, mut hi) = (0, self.count);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.key(mid) < key {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        (lo..self.count).take_while(|at| self.key(*at) == key).map(|at| self.chunk_id(at)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_every_chunk_of_a_key_and_nothing_else() {
        let pairs = [("b", 7), ("a", 3), ("b", 2), ("ab", 9), ("", 1), ("မြန်မာ", 4), ("b", 5)];
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.bin");
        std::fs::write(&path, encode(&pairs)).unwrap();
        let keys = DocKeys::open(&path).unwrap();
        assert_eq!(keys.chunks_for("b"), vec![2, 5, 7]);
        assert_eq!(keys.chunks_for("a"), vec![3]);
        assert_eq!(keys.chunks_for("ab"), vec![9]);
        assert_eq!(keys.chunks_for(""), vec![1]);
        assert_eq!(keys.chunks_for("မြန်မာ"), vec![4]);
        assert!(keys.chunks_for("c").is_empty());
        assert!(keys.chunks_for("a\0").is_empty());

        let bytes = std::fs::read(&path).unwrap();
        for cut in [0, 8, 15, bytes.len() - 1] {
            std::fs::write(&path, &bytes[..cut]).unwrap();
            assert!(DocKeys::open(&path).is_err(), "cut {cut}");
        }
    }

    #[test]
    fn an_empty_segment_has_no_keys() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("keys.bin");
        std::fs::write(&path, encode(&[])).unwrap();
        assert!(DocKeys::open(&path).unwrap().chunks_for("a").is_empty());
    }
}
