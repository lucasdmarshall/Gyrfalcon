use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::store;

/// 4: BM25 statistics come from segment dictionaries (no `bm25/` or `terms.bin`); 3 added
/// dense `codes.bin`; 2 held a USearch graph; 1 also held Tantivy.
pub const FORMAT_VERSION: u32 = 4;
pub const ANALYZER_ID: &str = "gyr-text-v1";
pub const DEFAULT_MODEL_ID: &str = "BAAI/bge-small-en-v1.5";
pub const DEFAULT_DIMENSION: u32 = 384;
/// The static (Model2Vec) embedder of the lite tier.
pub const STATIC_MODEL_ID: &str = "minishlab/potion-retrieval-32M";
pub const DEFAULT_RERANKER_ID: &str = "Xenova/ms-marco-MiniLM-L-6-v2";
pub const SEGMENTS_PER_TIER: u32 = 8;
pub const MAX_MERGE_CHUNKS: u64 = 100_000;
pub const DEAD_RATIO_LIMIT: f64 = 0.20;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Quantization {
    #[serde(rename = "i8")]
    I8,
    #[serde(rename = "f16")]
    F16,
    #[serde(rename = "f32")]
    F32,
    #[serde(rename = "b1")]
    B1,
}

impl Quantization {
    pub fn as_str(self) -> &'static str {
        match self {
            Quantization::I8 => "i8",
            Quantization::F16 => "f16",
            Quantization::F32 => "f32",
            Quantization::B1 => "b1",
        }
    }

    pub fn parse(value: &str) -> Option<Quantization> {
        match value {
            "i8" => Some(Quantization::I8),
            "f16" => Some(Quantization::F16),
            "f32" => Some(Quantization::F32),
            "b1" => Some(Quantization::B1),
            _ => None,
        }
    }

    /// Bytes per vector component in `vectors.bin`. Binary mode stores the int8 matrix.
    pub fn bytes_per_component(self) -> usize {
        match self {
            Quantization::I8 | Quantization::B1 => 1,
            Quantization::F16 => 2,
            Quantization::F32 => 4,
        }
    }
}

/// The process config an index is created with and must match on open.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexConfig {
    pub model_id: String,
    pub dimension: u32,
    pub quantization: Quantization,
    pub analyzer_id: String,
}

impl Default for IndexConfig {
    fn default() -> Self {
        IndexConfig {
            model_id: DEFAULT_MODEL_ID.to_string(),
            dimension: DEFAULT_DIMENSION,
            quantization: Quantization::I8,
            analyzer_id: ANALYZER_ID.to_string(),
        }
    }
}

impl IndexConfig {
    /// The v1 defaults: `BAAI/bge-small-en-v1.5`, 384, `i8`, `gyr-text-v1`.
    pub fn test_default() -> Self {
        IndexConfig::default()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Manifest {
    pub format_version: u32,
    pub model_id: String,
    pub dimension: u32,
    pub quantization: Quantization,
    pub metric: String,
    pub analyzer_id: String,
    pub tombstone_generation: u64,
    pub catalog_generation: u64,
    pub bm25_num_docs: u64,
    pub bm25_num_tokens: u64,
    pub segments_per_tier: u32,
    pub max_merge_chunks: u64,
    pub dead_ratio_limit: f64,
    pub epoch: u64,
    pub sealed_segments: Vec<String>,
    /// Next unallocated chunk id. Chunk ids are never reused, even after a merge drops the highest one.
    pub next_chunk_id: u64,
    /// Next unallocated segment id.
    pub next_segment_id: u64,
}

impl Manifest {
    pub fn new(config: &IndexConfig) -> Manifest {
        Manifest {
            format_version: FORMAT_VERSION,
            model_id: config.model_id.clone(),
            dimension: config.dimension,
            quantization: config.quantization,
            metric: "cosine".to_string(),
            analyzer_id: config.analyzer_id.clone(),
            tombstone_generation: 0,
            catalog_generation: 0,
            bm25_num_docs: 0,
            bm25_num_tokens: 0,
            segments_per_tier: SEGMENTS_PER_TIER,
            max_merge_chunks: MAX_MERGE_CHUNKS,
            dead_ratio_limit: DEAD_RATIO_LIMIT,
            epoch: 1,
            sealed_segments: Vec::new(),
            next_chunk_id: 0,
            next_segment_id: 0,
        }
    }

    pub fn read(path: &Path) -> Result<Manifest> {
        let bytes = match std::fs::read(path) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Err(Error::IndexNotFound(path.parent().unwrap_or(path).to_path_buf()))
            }
            Err(err) => return Err(err.into()),
        };
        Ok(serde_json::from_slice(&bytes)?)
    }

    /// Writes `manifest.json.tmp`, fsyncs it, and renames it over `manifest.json`.
    pub fn publish(&self, root: &Path) -> Result<()> {
        let tmp = root.join("manifest.json.tmp");
        store::write_synced(&tmp, &serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(&tmp, root.join("manifest.json"))?;
        store::sync_dir(root)?;
        Ok(())
    }

    pub fn config(&self) -> IndexConfig {
        IndexConfig {
            model_id: self.model_id.clone(),
            dimension: self.dimension,
            quantization: self.quantization,
            analyzer_id: self.analyzer_id.clone(),
        }
    }
}

/// Compares the four identity fields. Order: model, dimension, quantization, analyzer.
pub fn check_compatible(manifest: &Manifest, expected: &IndexConfig) -> Result<()> {
    use crate::error::{mismatch, IncompatibleField};
    if manifest.format_version != FORMAT_VERSION {
        return mismatch(
            IncompatibleField::FormatVersion,
            &manifest.format_version.to_string(),
            &FORMAT_VERSION.to_string(),
        );
    }
    if manifest.model_id != expected.model_id {
        return mismatch(IncompatibleField::ModelId, &manifest.model_id, &expected.model_id);
    }
    if manifest.dimension != expected.dimension {
        return mismatch(
            IncompatibleField::Dimension,
            &manifest.dimension.to_string(),
            &expected.dimension.to_string(),
        );
    }
    if manifest.quantization != expected.quantization {
        return mismatch(
            IncompatibleField::Quantization,
            manifest.quantization.as_str(),
            expected.quantization.as_str(),
        );
    }
    if manifest.analyzer_id != expected.analyzer_id {
        return mismatch(IncompatibleField::AnalyzerId, &manifest.analyzer_id, &expected.analyzer_id);
    }
    Ok(())
}
