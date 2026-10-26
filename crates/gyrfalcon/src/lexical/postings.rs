//! One term's postings: ascending doc ordinals and term frequencies.
//!
//! ```text
//! df < 128    vint stream
//! df >= 128   skip entries, one per 128-doc block, then the block data
//!   skip      last_doc u32, offset u32, max_tf u32, min_norm u8, doc_bits u8, tf_bits u8
//!   block     bit-packed doc deltas, then bit-packed (tf - 1); a final partial block is a vint stream
//! ```
//!
//! A vint stream holds `delta << 1 | (tf == 1)` per doc, followed by `tf` when it is not 1.
//! Deltas start from the previous block's last doc, or 0. `max_tf` and `min_norm` bound the
//! block's best BM25 contribution whatever the collection's average length is.

use bitpacking::{BitPacker, BitPacker4x};

use crate::error::{Error, Result};

pub const BLOCK: usize = BitPacker4x::BLOCK_LEN;
const SKIP_BYTES: usize = 15;

pub fn corrupt(what: &str) -> Error {
    Error::SegmentCorrupt(format!("lexical postings: {what}"))
}

fn write_vint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 0x80 {
        out.push(value as u8 | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

fn read_vint(bytes: &[u8], at: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    for shift in (0..64).step_by(7) {
        let byte = *bytes.get(*at).ok_or_else(|| corrupt("truncated vint"))?;
        *at += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte < 0x80 {
            return Ok(value);
        }
    }
    Err(corrupt("vint too long"))
}

fn encode_vints(initial: u32, docs: &[u32], tfs: &[u32], out: &mut Vec<u8>) {
    let mut previous = initial;
    for (&doc, &tf) in docs.iter().zip(tfs) {
        write_vint(u64::from(doc - previous) << 1 | u64::from(tf == 1), out);
        if tf != 1 {
            write_vint(u64::from(tf), out);
        }
        previous = doc;
    }
}

fn decode_vints(initial: u32, bytes: &[u8], docs: &mut [u32], tfs: &mut [u32]) -> Result<()> {
    let mut at = 0;
    let mut previous = initial;
    for (doc, tf) in docs.iter_mut().zip(tfs.iter_mut()) {
        let code = read_vint(bytes, &mut at)?;
        let delta = u32::try_from(code >> 1).map_err(|_| corrupt("doc delta overflows"))?;
        *doc = previous.checked_add(delta).ok_or_else(|| corrupt("doc ordinal overflows"))?;
        *tf = if code & 1 == 1 {
            1
        } else {
            u32::try_from(read_vint(bytes, &mut at)?).map_err(|_| corrupt("term frequency overflows"))?
        };
        previous = *doc;
    }
    Ok(())
}

/// The largest `tf` and the smallest length norm among some postings.
pub fn bound(tfs: &[u32], docs: &[u32], norm: &impl Fn(u32) -> u8) -> (u32, u8) {
    let max_tf = tfs.iter().copied().max().unwrap_or(0);
    let min_norm = docs.iter().map(|&doc| norm(doc)).min().unwrap_or(u8::MAX);
    (max_tf, min_norm)
}

/// Appends one term's postings. `docs` ascend strictly, every `tf` is at least 1, and `norm`
/// gives a doc's length norm.
pub fn encode(docs: &[u32], tfs: &[u32], norm: &impl Fn(u32) -> u8, out: &mut Vec<u8>) {
    debug_assert!(docs.windows(2).all(|w| w[0] < w[1]) && tfs.iter().all(|&tf| tf >= 1));
    if docs.len() < BLOCK {
        encode_vints(0, docs, tfs, out);
        return;
    }
    let packer = BitPacker4x::new();
    let skip_at = out.len();
    out.resize(skip_at + docs.len().div_ceil(BLOCK) * SKIP_BYTES, 0);
    let data_at = out.len();
    let mut initial = 0;
    let mut packed = [0u8; 4 * BLOCK];
    let mut minus_one = [0u32; BLOCK];
    for (block, (docs, tfs)) in docs.chunks(BLOCK).zip(tfs.chunks(BLOCK)).enumerate() {
        let (max_tf, min_norm) = bound(tfs, docs, norm);
        let offset = (out.len() - data_at) as u32;
        let (doc_bits, tf_bits) = if docs.len() == BLOCK {
            let doc_bits = packer.num_bits_sorted(initial, docs);
            let written = packer.compress_sorted(initial, docs, &mut packed, doc_bits);
            out.extend_from_slice(&packed[..written]);
            for (slot, tf) in minus_one.iter_mut().zip(tfs) {
                *slot = tf - 1;
            }
            let tf_bits = packer.num_bits(&minus_one);
            let written = packer.compress(&minus_one, &mut packed, tf_bits);
            out.extend_from_slice(&packed[..written]);
            (doc_bits, tf_bits)
        } else {
            encode_vints(initial, docs, tfs, out);
            (0, 0)
        };
        let last_doc = *docs.last().unwrap();
        let skip = &mut out[skip_at + block * SKIP_BYTES..skip_at + (block + 1) * SKIP_BYTES];
        skip[0..4].copy_from_slice(&last_doc.to_le_bytes());
        skip[4..8].copy_from_slice(&offset.to_le_bytes());
        skip[8..12].copy_from_slice(&max_tf.to_le_bytes());
        skip[12] = min_norm;
        skip[13] = doc_bits;
        skip[14] = tf_bits;
        initial = last_doc;
    }
}

/// A block's skip entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Skip {
    pub last_doc: u32,
    offset: u32,
    pub max_tf: u32,
    pub min_norm: u8,
    doc_bits: u8,
    tf_bits: u8,
}

/// A borrowed, encoded posting list.
pub struct PostingList<'a> {
    bytes: &'a [u8],
    df: usize,
    /// Skip entries; empty when `df < BLOCK`.
    skips: &'a [u8],
    data: &'a [u8],
}

impl<'a> PostingList<'a> {
    pub fn new(bytes: &'a [u8], df: u32) -> Result<PostingList<'a>> {
        let df = df as usize;
        let skip_len = if df < BLOCK { 0 } else { df.div_ceil(BLOCK) * SKIP_BYTES };
        if bytes.len() < skip_len {
            return Err(corrupt("skip entries are truncated"));
        }
        let (skips, data) = bytes.split_at(skip_len);
        Ok(PostingList { bytes, df, skips, data })
    }

    pub fn blocks(&self) -> usize {
        self.df.div_ceil(BLOCK)
    }

    /// The skip entry of `block`; `None` for a list shorter than one block.
    pub fn skip(&self, block: usize) -> Option<Skip> {
        let entry = self.skips.get(block * SKIP_BYTES..(block + 1) * SKIP_BYTES)?;
        let u32_at = |at: usize| u32::from_le_bytes(entry[at..at + 4].try_into().unwrap());
        Some(Skip {
            last_doc: u32_at(0),
            offset: u32_at(4),
            max_tf: u32_at(8),
            min_norm: entry[12],
            doc_bits: entry[13],
            tf_bits: entry[14],
        })
    }

    /// Decodes `block` into `docs` and `tfs` and returns its length.
    pub fn decode(&self, block: usize, docs: &mut [u32; BLOCK], tfs: &mut [u32; BLOCK]) -> Result<usize> {
        let len = (self.df - block * BLOCK).min(BLOCK);
        if self.df < BLOCK {
            decode_vints(0, self.bytes, &mut docs[..len], &mut tfs[..len])?;
            return Ok(len);
        }
        let skip = self.skip(block).ok_or_else(|| corrupt("block out of range"))?;
        let initial = if block == 0 { 0 } else { self.skip(block - 1).map(|s| s.last_doc).unwrap_or(0) };
        let data = self.data.get(skip.offset as usize..).ok_or_else(|| corrupt("block offset out of range"))?;
        if len < BLOCK {
            decode_vints(initial, data, &mut docs[..len], &mut tfs[..len])?;
            return Ok(len);
        }
        if skip.doc_bits > 32 || skip.tf_bits > 32 {
            return Err(corrupt("bit width over 32"));
        }
        let doc_bytes = BitPacker4x::compressed_block_size(skip.doc_bits);
        let tf_bytes = BitPacker4x::compressed_block_size(skip.tf_bits);
        if data.len() < doc_bytes + tf_bytes {
            return Err(corrupt("block is truncated"));
        }
        let packer = BitPacker4x::new();
        packer.decompress_sorted(initial, &data[..doc_bytes], docs, skip.doc_bits);
        packer.decompress(&data[doc_bytes..doc_bytes + tf_bytes], tfs, skip.tf_bits);
        for tf in tfs.iter_mut() {
            *tf = tf.wrapping_add(1);
        }
        Ok(BLOCK)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip(docs: &[u32], tfs: &[u32]) {
        let norm = |doc: u32| (doc.wrapping_mul(2654435761) >> 24) as u8;
        let mut bytes = vec![0xAA; 3];
        encode(docs, tfs, &norm, &mut bytes);
        let list = PostingList::new(&bytes[3..], docs.len() as u32).unwrap();
        assert_eq!(list.blocks(), docs.len().div_ceil(BLOCK));
        let (mut got_docs, mut got_tfs) = (Vec::new(), Vec::new());
        let (mut block_docs, mut block_tfs) = ([0; BLOCK], [0; BLOCK]);
        for block in 0..list.blocks() {
            let len = list.decode(block, &mut block_docs, &mut block_tfs).unwrap();
            if let Some(skip) = list.skip(block) {
                assert_eq!(skip.last_doc, block_docs[len - 1]);
                let (max_tf, min_norm) = bound(&block_tfs[..len], &block_docs[..len], &norm);
                assert_eq!((skip.max_tf, skip.min_norm), (max_tf, min_norm));
            }
            got_docs.extend_from_slice(&block_docs[..len]);
            got_tfs.extend_from_slice(&block_tfs[..len]);
        }
        assert_eq!(got_docs, docs);
        assert_eq!(got_tfs, tfs);
    }

    #[test]
    fn postings_roundtrip_at_every_block_boundary() {
        let mut seed = 11u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for len in [1, 2, 127, 128, 129, 255, 256, 257, 1000] {
            for spread in [1u64, 3, 1000, 1 << 20] {
                let mut doc = 0u32;
                let docs: Vec<u32> = (0..len)
                    .map(|i| {
                        if i > 0 {
                            doc += 1 + (next() % spread) as u32;
                        }
                        doc
                    })
                    .collect();
                let tfs: Vec<u32> = (0..len).map(|_| 1 + (next() % 5) as u32 * (next() % 3) as u32).collect();
                roundtrip(&docs, &tfs);
            }
        }
        roundtrip(&[0, u32::MAX - 1], &[u32::MAX, 1]);
    }

    #[test]
    fn truncated_lists_are_errors_not_panics() {
        let docs: Vec<u32> = (0..300).collect();
        let tfs = vec![2; 300];
        let mut bytes = Vec::new();
        encode(&docs, &tfs, &|_| 0, &mut bytes);
        let (mut d, mut t) = ([0; BLOCK], [0; BLOCK]);
        for cut in [0, 10, 45, 60, bytes.len() - 1] {
            let result = PostingList::new(&bytes[..cut], 300).and_then(|list| {
                (0..list.blocks()).try_for_each(|block| list.decode(block, &mut d, &mut t).map(|_| ()))
            });
            assert!(result.is_err(), "cut {cut}");
        }
    }
}
