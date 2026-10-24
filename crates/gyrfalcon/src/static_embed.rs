//! Static (Model2Vec) embeddings: a token-to-vector table, mean-pooled. No transformer runs,
//! so embedding costs a tokenizer pass and a few hundred additions per token.

use std::path::Path;

use tokenizers::Tokenizer;

use crate::embed::Embedder;
use crate::error::{Error, Result};

/// Tokens per text, as in Model2Vec's default `max_length`.
pub const STATIC_MAX_TOKENS: usize = 512;

fn model_error(err: impl std::fmt::Display) -> Error {
    Error::ModelUnavailable(err.to_string())
}

/// A Model2Vec model directory: `config.json`, `model.safetensors`, `tokenizer.json`.
pub struct StaticEmbedder {
    tokenizer: Tokenizer,
    /// Row-major `[vocab, dimension]`.
    table: Vec<f32>,
    dimension: usize,
    normalize: bool,
    unk_id: Option<u32>,
    model_id: String,
}

/// The `embeddings` tensor header of a safetensors file: dtype, shape, and byte range.
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
    if shape.len() != 2 || end > bytes.len() {
        return Err(model_error("embeddings tensor must be 2-D and inside the file"));
    }
    Ok((dtype, shape, start, end))
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
        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(model_error)?;
        tokenizer.with_padding(None);
        tokenizer.with_truncation(None).map_err(model_error)?;
        let bytes = std::fs::read(dir.join("model.safetensors")).map_err(model_error)?;
        let (dtype, shape, start, end) = embeddings_header(&bytes)?;
        let data = &bytes[start..end];
        let table: Vec<f32> = match dtype.as_str() {
            "F32" => data.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
            "F16" => data.chunks_exact(2).map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
            other => return Err(model_error(format!("unsupported embeddings dtype {other}"))),
        };
        if table.len() != shape[0] * shape[1] {
            return Err(model_error("embeddings tensor size does not match its shape"));
        }
        let unk_id = tokenizer.token_to_id("[UNK]");
        Ok(StaticEmbedder { tokenizer, table, dimension: shape[1], normalize, unk_id, model_id: model_id.into() })
    }

    fn pool(&self, ids: &[u32]) -> Vec<f32> {
        let mut out = vec![0f32; self.dimension];
        let mut count = 0usize;
        for id in ids.iter().filter(|id| Some(**id) != self.unk_id).take(STATIC_MAX_TOKENS) {
            let start = *id as usize * self.dimension;
            let Some(row) = self.table.get(start..start + self.dimension) else { continue };
            for (o, v) in out.iter_mut().zip(row) {
                *o += v;
            }
            count += 1;
        }
        if count > 0 {
            out.iter_mut().for_each(|v| *v /= count as f32);
        }
        if self.normalize {
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
        let inputs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let encodings = self.tokenizer.encode_batch(inputs, false).map_err(|e| Error::Embed(e.to_string()))?;
        Ok(encodings.iter().map(|e| self.pool(e.get_ids())).collect())
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn dimension(&self) -> u32 {
        self.dimension as u32
    }
}
