//! Gyrfalcon: a sealed-segment hybrid retrieval library.
//!
//! One process owns one collection. `index_documents` seals immutable segment directories
//! and publishes `manifest.json` by rename.

mod chunk;
mod embed;
mod epoch;
mod error;
mod fuse;
mod http;
mod index;
mod lexical;
mod manifest;
mod merge;
mod ort_session;
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
pub use embed::{testing, EmbedSession, Embedder, HashEmbedder, HttpEmbedder, OrtEmbedder, ScriptedSession, SessionEmbedder};
pub use epoch::{FileDeleter, FsDeleter, ReaderGuard, Reaper};
pub use fuse::rrf;
pub use merge::{dead_ratio_target, select_tier};
pub use lexical::{GyrTextTokenizer, LexicalIndex, ScoredId, TantivyLexical};
pub use ort_session::OrtModel;
pub use rerank::{NoopReranker, OrtReranker, Reranker, ScriptedReranker, RERANK_K};
pub use retrieve::{DenseHits, Hit, Hits, RetrieveOptions, LEG_LIMIT};
pub use tombstone::DeleteReport;
pub use vector::{FallbackStep, VectorSegment};
