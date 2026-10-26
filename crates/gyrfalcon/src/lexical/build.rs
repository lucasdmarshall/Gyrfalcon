//! Builds a segment's lexical files from its chunk records in one pass over the tokens.
//!
//! Terms are interned as borrowed strings, each doc becomes `(term, tf)` pairs, and a counting
//! sort by term turns those into posting lists that are already in doc order. No token string
//! is copied.

use std::borrow::Borrow;
use std::path::Path;

use rustc_hash::FxHashMap;

use super::{docstore, fieldnorm_to_id, postings, DICT_ENTRY, DICT_HEADER, DICT_MAGIC, NORMS_MAGIC, POSTINGS_MAGIC};
use crate::error::{Error, Result};
use crate::segment::ChunkRecord;

/// What a segment's BM25 totals need from the build.
pub struct Built {
    /// Token occurrences, repeats included.
    pub num_tokens: u64,
}

/// The in-memory inverted index of one segment.
struct Inverted<'a> {
    /// Term strings by term id.
    names: Vec<&'a str>,
    /// Postings by term id: `docs[starts[t]..starts[t + 1]]`.
    starts: Vec<usize>,
    docs: Vec<u32>,
    tfs: Vec<u32>,
    /// One length norm per doc.
    norms: Vec<u8>,
    num_tokens: u64,
}

fn invert<R: Borrow<ChunkRecord>>(chunks: &[R]) -> Result<Inverted<'_>> {
    if u32::try_from(chunks.len()).is_err() {
        return Err(Error::Lexical("a segment holds at most u32::MAX chunks".into()));
    }
    let total: usize = chunks.iter().map(|c| c.borrow().terms.len()).sum();
    let mut ids: FxHashMap<&str, u32> = FxHashMap::with_capacity_and_hasher(total / 8, Default::default());
    let mut names: Vec<&str> = Vec::new();
    let mut df: Vec<u32> = Vec::new();
    // `(term, tf)` per doc, docs back to back; `ends[d]` closes doc `d`.
    let mut pairs: Vec<(u32, u32)> = Vec::with_capacity(total);
    let mut ends: Vec<usize> = Vec::with_capacity(chunks.len());
    let mut norms = Vec::with_capacity(chunks.len());
    let mut num_tokens = 0u64;
    for chunk in chunks {
        let terms = &chunk.borrow().terms;
        num_tokens += u64::from(terms.num_tokens());
        norms.push(fieldnorm_to_id(terms.num_tokens()));
        for (term, tf) in terms.iter() {
            let next = names.len() as u32;
            let id = *ids.entry(term).or_insert_with(|| {
                names.push(term);
                df.push(0);
                next
            });
            pairs.push((id, tf));
            df[id as usize] += 1;
        }
        ends.push(pairs.len());
    }

    let mut starts = Vec::with_capacity(df.len() + 1);
    starts.push(0);
    for count in &df {
        starts.push(starts.last().unwrap() + *count as usize);
    }
    let mut fill = starts.clone();
    let mut docs = vec![0u32; pairs.len()];
    let mut tfs = vec![0u32; pairs.len()];
    let mut begin = 0;
    for (doc, &end) in ends.iter().enumerate() {
        for &(term, tf) in &pairs[begin..end] {
            let slot = &mut fill[term as usize];
            docs[*slot] = doc as u32;
            tfs[*slot] = tf;
            *slot += 1;
        }
        begin = end;
    }
    Ok(Inverted { names, starts, docs, tfs, norms, num_tokens })
}

/// Writes `terms.dict`, `postings.bin`, `norms.bin`, and `docs.bin` under `dir`. The caller
/// fsyncs the directory tree.
pub fn write<R: Borrow<ChunkRecord> + Sync>(dir: &Path, chunks: &[R]) -> Result<Built> {
    std::fs::create_dir_all(dir)?;
    std::thread::scope(|scope| {
        let docs = scope.spawn(|| std::fs::write(dir.join("docs.bin"), docstore::encode(chunks)));
        let built = write_index(dir, chunks);
        docs.join().map_err(|_| Error::Lexical("docs.bin build panicked".into()))??;
        built
    })
}

fn write_index<R: Borrow<ChunkRecord>>(dir: &Path, chunks: &[R]) -> Result<Built> {
    let inverted = invert(chunks)?;
    let mut order: Vec<u32> = (0..inverted.names.len() as u32).collect();
    order.sort_unstable_by(|a, b| inverted.names[*a as usize].cmp(inverted.names[*b as usize]));

    let norms = &inverted.norms;
    let norm = |doc: u32| norms[doc as usize];
    let mut postings = Vec::with_capacity(8 + inverted.docs.len() * 2);
    postings.extend_from_slice(POSTINGS_MAGIC);
    let blob_len: usize = inverted.names.iter().map(|n| n.len()).sum();
    if u32::try_from(blob_len).is_err() {
        return Err(Error::Lexical("a segment's terms hold at most u32::MAX bytes".into()));
    }
    let mut dict = Vec::with_capacity(DICT_HEADER + order.len() * (4 + DICT_ENTRY) + blob_len);
    dict.extend_from_slice(DICT_MAGIC);
    dict.extend_from_slice(&(order.len() as u32).to_le_bytes());
    dict.extend_from_slice(&(chunks.len() as u32).to_le_bytes());
    dict.extend_from_slice(&(blob_len as u64).to_le_bytes());
    let mut end = 0u32;
    for &term in &order {
        end += inverted.names[term as usize].len() as u32;
        dict.extend_from_slice(&end.to_le_bytes());
    }
    for &term in &order {
        let range = inverted.starts[term as usize]..inverted.starts[term as usize + 1];
        let (docs, tfs) = (&inverted.docs[range.clone()], &inverted.tfs[range]);
        let (max_tf, min_norm) = postings::bound(tfs, docs, &norm);
        dict.extend_from_slice(&(postings.len() as u64).to_le_bytes());
        dict.extend_from_slice(&(docs.len() as u32).to_le_bytes());
        dict.extend_from_slice(&max_tf.to_le_bytes());
        dict.push(min_norm);
        postings::encode(docs, tfs, &norm, &mut postings);
    }
    for &term in &order {
        dict.extend_from_slice(inverted.names[term as usize].as_bytes());
    }
    let mut norms_file = Vec::with_capacity(8 + norms.len());
    norms_file.extend_from_slice(NORMS_MAGIC);
    norms_file.extend_from_slice(norms);

    std::fs::write(dir.join("terms.dict"), dict)?;
    std::fs::write(dir.join("postings.bin"), postings)?;
    std::fs::write(dir.join("norms.bin"), norms_file)?;
    Ok(Built { num_tokens: inverted.num_tokens })
}
