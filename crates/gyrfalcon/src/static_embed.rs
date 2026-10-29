//! Static (Model2Vec) embeddings: a token-to-vector table, mean-pooled. No transformer runs,
//! so embedding costs a tokenizer pass and a few hundred additions per token.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use tokenizers::Tokenizer;

use crate::embed::Embedder;
use crate::error::{Error, Result};

/// Model2Vec's default `max_length`. Like the `model2vec` package, a text is cut to
/// `max_length * median_token_length` characters, tokenized, truncated to `max_length` ids
/// (unknown tokens included), and the ids other than `[UNK]` are pooled.
pub const STATIC_MAX_TOKENS: usize = 512;

fn model_error(err: impl std::fmt::Display) -> Error {
    Error::ModelUnavailable(err.to_string())
}

/// The embeddings table. Little-endian f32 data, 4-byte aligned in the file, stays mapped.
/// Open reads the file once (129 MB for potion-retrieval-32M) so a query's token rows are
/// already cached; the rows themselves stay in the mapping. Other dtypes are converted into
/// memory.
enum Table {
    /// `file` is the mapped safetensors file. A cold query reads token rows through it; opening
    /// the path again would drop the page cache on FUSE.
    Mapped {
        map: memmap2::Mmap,
        file: std::fs::File,
        start: usize,
        len: usize,
    },
    Owned(Vec<f32>),
}

impl Table {
    fn floats(&self) -> &[f32] {
        match self {
            Table::Owned(floats) => floats,
            Table::Mapped { map, start, len, .. } => {
                let bytes = &map[*start..*start + *len * 4];
                // SAFETY: `from_dir` checked that the range is in the file, 4-byte aligned, and
                // little-endian f32 data on a little-endian target; every bit pattern is an f32.
                unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), *len) }
            }
        }
    }
}

/// A Model2Vec model directory: `config.json`, `model.safetensors`, `tokenizer.json`.
pub struct StaticEmbedder {
    tokenizer_path: PathBuf,
    /// Built on first use when `fast` handles most text, at once otherwise.
    tokenizer: OnceLock<Tokenizer>,
    /// Allocation-free path for BERT WordPiece tokenizers; `None` uses `tokenizer` alone.
    fast: Option<crate::wordpiece::FastWordPiece>,
    /// Characters kept per text: `STATIC_MAX_TOKENS * median vocabulary token length`.
    max_chars: usize,
    /// Row-major `[vocab, width]`.
    table: Table,
    width: usize,
    /// Leading components of each row that are pooled: `width`, or fewer for a Matryoshka
    /// model truncated by `truncated`.
    dimension: usize,
    normalize: bool,
    unk_id: Option<u32>,
    model_id: String,
}

/// The `embeddings` tensor header of a safetensors file: dtype, shape, and byte range.
fn load_tokenizer(path: &Path) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(path).map_err(model_error)?;
    tokenizer.with_padding(None);
    tokenizer.with_truncation(None).map_err(model_error)?;
    Ok(tokenizer)
}

fn embeddings_header(bytes: &[u8]) -> Result<(String, Vec<usize>, usize, usize)> {
    if bytes.len() < 8 {
        return Err(model_error("model.safetensors is truncated"));
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value =
        serde_json::from_slice(bytes.get(8..8 + header_len).ok_or_else(|| model_error("bad safetensors header"))?)?;
    let tensor = header
        .get("embeddings")
        .or_else(|| header.as_object().and_then(|o| o.iter().find(|(k, _)| *k != "__metadata__").map(|(_, v)| v)))
        .ok_or_else(|| model_error("model.safetensors has no tensor"))?;
    let dtype = tensor["dtype"].as_str().unwrap_or_default().to_string();
    let shape: Vec<usize> = tensor["shape"]
        .as_array()
        .map(|s| s.iter().filter_map(|d| d.as_u64()).map(|d| d as usize).collect())
        .unwrap_or_default();
    let offsets = tensor["data_offsets"].as_array().ok_or_else(|| model_error("tensor has no data_offsets"))?;
    let start = 8 + header_len + offsets[0].as_u64().unwrap_or(0) as usize;
    let end = 8 + header_len + offsets[1].as_u64().unwrap_or(0) as usize;
    if shape.len() != 2 || start > end || end > bytes.len() {
        return Err(model_error("embeddings tensor must be 2-D and inside the file"));
    }
    Ok((dtype, shape, start, end))
}

fn floats_of(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect()
}

fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half & 0x8000) as u32) << 16;
    let exponent = ((half >> 10) & 0x1f) as u32;
    let mantissa = (half & 0x3ff) as u32;
    let bits = match (exponent, mantissa) {
        (0, 0) => sign,
        (0, _) => {
            let mut e = 127 - 15 + 1;
            let mut m = mantissa;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, _) => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}

impl StaticEmbedder {
    /// This embedder pooling only the first `dimension` components of each row, for a model
    /// trained with Matryoshka loss (its leading components carry the most). A larger
    /// `dimension` changes nothing.
    pub fn truncated(mut self, dimension: usize) -> StaticEmbedder {
        self.dimension = self.dimension.min(dimension.max(1));
        self
    }

    /// The output dimension, read from the safetensors header without loading the table.
    pub fn read_dimension(dir: &Path) -> Result<u32> {
        use std::io::Read;
        let mut file = std::fs::File::open(dir.join("model.safetensors")).map_err(model_error)?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len).map_err(model_error)?;
        let mut head = vec![0u8; u64::from_le_bytes(len) as usize];
        file.read_exact(&mut head).map_err(model_error)?;
        let mut bytes = len.to_vec();
        bytes.extend(head);
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..])?;
        let shape = header
            .as_object()
            .and_then(|o| o.iter().find(|(k, _)| *k != "__metadata__"))
            .and_then(|(_, v)| v["shape"].as_array().cloned())
            .ok_or_else(|| model_error("model.safetensors has no tensor shape"))?;
        Ok(shape.get(1).and_then(|d| d.as_u64()).ok_or_else(|| model_error("tensor is not 2-D"))? as u32)
    }

    pub fn from_dir(dir: &Path, model_id: impl Into<String>) -> Result<StaticEmbedder> {
        for file in ["model.safetensors", "tokenizer.json"] {
            if !dir.join(file).is_file() {
                return Err(Error::ModelUnavailable(format!("no {file} under {}", dir.display())));
            }
        }
        let config: serde_json::Value = match std::fs::read(dir.join("config.json")) {
            Ok(bytes) => serde_json::from_slice(&bytes)?,
            Err(_) => serde_json::json!({}),
        };
        let normalize = config["normalize"].as_bool().unwrap_or(true);
        let file = std::fs::File::open(dir.join("model.safetensors")).map_err(model_error)?;
        // SAFETY: the model file is not modified while the process runs, as with segments.
        let map = unsafe { memmap2::Mmap::map(&file) }.map_err(model_error)?;
        // Queries touch scattered rows. Fault them through this mapping: opening the file
        // again to read it would drop the pages on FUSE, and the query would read them back.
        crate::warm::warm_map(&map);
        // Token rows are scattered. Once they are resident, a later miss should not pull the
        // neighboring rows in with them.
        let _ = map.advise(memmap2::Advice::Random);
        let (dtype, shape, start, end) = embeddings_header(&map)?;
        let len = shape[0] * shape[1];
        let data = &map[start..end];
        if data.len() != len * if dtype == "F16" { 2 } else { 4 } {
            return Err(model_error("embeddings tensor size does not match its shape"));
        }
        let table = match dtype.as_str() {
            "F32" if cfg!(target_endian = "little") && (data.as_ptr() as usize).is_multiple_of(4) => {
                Table::Mapped { map, file, start, len }
            }
            "F32" => Table::Owned(data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect()),
            "F16" => Table::Owned(data.as_chunks::<2>().0.iter().map(|b| f16_to_f32(u16::from_le_bytes(*b))).collect()),
            other => return Err(model_error(format!("unsupported embeddings dtype {other}"))),
        };
        let tokenizer_path = dir.join("tokenizer.json");
        let fast = crate::wordpiece::FastWordPiece::from_tokenizer_json(&std::fs::read(&tokenizer_path)?);
        let tokenizer = OnceLock::new();
        let (unk_id, median) = match &fast {
            Some(fast) => (fast.unk_literal, fast.median_token_chars),
            None => {
                let loaded = load_tokenizer(&tokenizer_path)?;
                // `int(np.median([len(token) for token in tokens]))`, lengths in characters.
                let mut lengths: Vec<usize> = loaded.get_vocab(true).keys().map(|t| t.chars().count()).collect();
                lengths.sort_unstable();
                let median = match lengths.len() {
                    0 => 1,
                    n if n % 2 == 1 => lengths[n / 2],
                    n => (lengths[n / 2 - 1] + lengths[n / 2]) / 2,
                };
                let unk_id = loaded.token_to_id("[UNK]");
                let _ = tokenizer.set(loaded);
                (unk_id, median)
            }
        };
        Ok(StaticEmbedder {
            tokenizer_path,
            tokenizer,
            fast,
            max_chars: STATIC_MAX_TOKENS * median.max(1),
            table,
            width: shape[1],
            dimension: shape[1],
            normalize,
            unk_id,
            model_id: model_id.into(),
        })
    }

    /// Token ids of `text` as `model2vec` computes them: cut to `max_chars` characters,
    /// tokenized without special tokens, truncated to `STATIC_MAX_TOKENS` ids.
    pub fn token_ids(&self, text: &str) -> Result<Vec<u32>> {
        let cut = match text.char_indices().nth(self.max_chars) {
            Some((at, _)) => &text[..at],
            None => text,
        };
        let mut ids = Vec::with_capacity(cut.len() / 4);
        let full = |text: &str, ids: &mut Vec<u32>| -> Result<()> {
            let encoding = self.tokenizer()?.encode(text, false).map_err(|e| Error::Embed(e.to_string()))?;
            ids.extend_from_slice(encoding.get_ids());
            Ok(())
        };
        match &self.fast {
            Some(fast) => fast.encode(cut, full, &mut ids, STATIC_MAX_TOKENS)?,
            None => full(cut, &mut ids)?,
        }
        ids.truncate(STATIC_MAX_TOKENS);
        Ok(ids)
    }

    fn tokenizer(&self) -> Result<&Tokenizer> {
        if let Some(tokenizer) = self.tokenizer.get() {
            return Ok(tokenizer);
        }
        // Two threads may both load it; one copy is kept.
        let _ = self.tokenizer.set(load_tokenizer(&self.tokenizer_path)?);
        Ok(self.tokenizer.get().expect("set above"))
    }

    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.pool(&self.token_ids(text)?))
    }

    fn pool(&self, ids: &[u32]) -> Vec<f32> {
        let kept: Vec<u32> = ids.iter().copied().filter(|id| Some(*id) != self.unk_id).collect();
        // A cold token row faults its page and the kernel's neighbors. Read those rows
        // directly when any of them is still on disk; a resident table takes the mapped path.
        if let Table::Mapped { map, file, start, .. } = &self.table {
            let ranges: Vec<(u64, usize)> = kept
                .iter()
                .map(|id| {
                    let at = *start as u64 + u64::from(*id) * self.width as u64 * 4;
                    (at, self.dimension * 4)
                })
                .collect();
            let as_usize: Vec<(usize, usize)> = ranges.iter().map(|(at, len)| (*at as usize, *len)).collect();
            if !crate::warm::ranges_resident(map, &as_usize) {
                if let Ok(buffers) = crate::warm::read_ranges(file, &ranges) {
                    let rows: Vec<Vec<f32>> = buffers.iter().map(|buf| floats_of(buf)).collect();
                    return Self::finish_pool(self.dimension, self.normalize, rows.iter().map(|row| row.as_slice()));
                }
            }
        }
        let dimension = self.dimension;
        let normalize = self.normalize;
        let width = self.width;
        let table = self.table.floats();
        let rows = kept.iter().filter_map(|id| {
            let start = *id as usize * width;
            table.get(start..start + dimension)
        });
        Self::finish_pool(dimension, normalize, rows)
    }

    fn finish_pool<'a>(dimension: usize, normalize: bool, rows: impl Iterator<Item = &'a [f32]>) -> Vec<f32> {
        let mut out = vec![0f32; dimension];
        let mut count = 0usize;
        for row in rows {
            for (o, v) in out.iter_mut().zip(row) {
                *o += v;
            }
            count += 1;
        }
        if count > 0 {
            out.iter_mut().for_each(|v| *v /= count as f32);
        }
        if normalize {
            let norm = out.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 0.0 {
                out.iter_mut().for_each(|v| *v /= norm);
            }
        }
        out
    }
}

impl Embedder for StaticEmbedder {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        if texts.len() < 64 || threads == 1 {
            return texts.iter().map(|t| self.embed_one(t)).collect();
        }
        let per_thread = texts.len().div_ceil(threads);
        std::thread::scope(|scope| {
            let parts: Vec<_> = texts
                .chunks(per_thread)
                .map(|part| scope.spawn(move || part.iter().map(|t| self.embed_one(t)).collect::<Result<Vec<_>>>()))
                .collect();
            let mut out = Vec::with_capacity(texts.len());
            for part in parts {
                out.extend(part.join().map_err(|_| Error::Embed("static embed thread panicked".into()))??);
            }
            Ok(out)
        })
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn dimension(&self) -> u32 {
        self.dimension as u32
    }
}
