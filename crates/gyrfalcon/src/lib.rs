//! Gyrfalcon: a sealed-segment hybrid retrieval library.
//!
//! One process owns one collection. `index_documents` seals immutable segment directories
//! and publishes `manifest.json` by rename.

mod chunk;
mod embed;
mod epoch;
mod error;
mod fuse;
mod index;
mod lexical;
mod manifest;
mod prepare;
mod rerank;
mod retrieve;
mod segment;
mod store;
mod tombstone;
mod vector;

pub use chunk::{chunk, chunk_with, Span};
pub use error::{Error, IncompatibleField, Result};
pub use index::{Document, Index, IndexReport, FAN_OUT_WARNING, SEAL_EVERY};
pub use manifest::{IndexConfig, Manifest, Quantization, ANALYZER_ID, DEFAULT_DIMENSION, DEFAULT_MODEL_ID, DEFAULT_RERANKER_ID};
pub use prepare::{prepare, prepare_with_prefix, PreparedText, Token};
pub use embed::{testing, Embedder, HashEmbedder};
pub use epoch::{FileDeleter, FsDeleter, ReaderGuard, Reaper};
pub use fuse::rrf;
pub use lexical::{GyrTextTokenizer, LexicalIndex, ScoredId, TantivyLexical};
pub use rerank::{NoopReranker, Reranker, RERANK_K};
pub use retrieve::{DenseHits, Hit, Hits, RetrieveOptions, LEG_LIMIT};
pub use tombstone::DeleteReport;
pub use vector::{FallbackStep, VectorSegment};
