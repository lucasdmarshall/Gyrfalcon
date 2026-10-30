//! Gyrfalcon: a sealed-segment hybrid retrieval library.
//!
//! One process owns one collection. `index_documents` seals immutable segment directories
//! and publishes `manifest.json` by rename.

mod answer;
pub mod bench;
mod buffer;
mod chunk;
mod embed;
mod epoch;
mod error;
mod filter;
mod fuse;
mod http;
mod index;
mod keys;
mod lexical;
mod manifest;
mod merge;
mod ort_session;
pub mod planner;
mod prepare;
mod rerank;
mod retrieve;
mod segment;
mod static_embed;
mod store;
mod tier;
mod tombstone;
mod vector;
mod wal;
mod warm;
mod wordpiece;

pub use answer::{
    complete, fill_answer_prompt, fill_judge_prompt, AnswerEvent, AnswerOptions, AnswerStream, ChatConfig,
    ANSWER_PROMPT, CONTEXT_K, JUDGE_PROMPT,
};
pub use chunk::{chunk, chunk_with, Span};
pub use embed::{
    testing, EmbedSession, Embedder, HashEmbedder, HttpEmbedder, OrtEmbedder, ScriptedSession, SessionEmbedder,
};
pub use epoch::{FileDeleter, FsDeleter, ReaderGuard, Reaper};
pub use error::{Error, IncompatibleField, Result};
pub use filter::Filter;
pub use fuse::{
    max_fusion, rrf, rrf_weighted, score_fusion, Fusion, FULL_RESCORE_DENSE_WEIGHT, LITE_DENSE_WEIGHT,
    RESCORE_DENSE_WEIGHT,
};
pub use index::{
    Document, Index, IndexReport, BUFFERED_WRITE_MAX, FAN_OUT_WARNING, SEAL_EVERY, WRITE_BUFFER, WRITE_BUFFER_BYTES,
};
pub use lexical::ScoredId;
pub use manifest::{
    IndexConfig, Manifest, Quantization, ANALYZER_ID, DEFAULT_DIMENSION, DEFAULT_MODEL_ID, DEFAULT_RERANKER_ID,
    STATIC_MODEL_ID,
};
pub use merge::{dead_ratio_target, select_tier};
pub use ort_session::OrtModel;
pub use prepare::{analyze, prepare, prepare_with_prefix, tokenize, PreparedText, Token};
pub use rerank::{NoopReranker, OrtReranker, Reranker, ScriptedReranker, RERANK_K};
pub use retrieve::{DenseHits, Hit, Hits, RetrieveOptions, LEG_LIMIT};
pub use static_embed::{StaticEmbedder, STATIC_MAX_TOKENS};
pub use tier::{ModelFile, Tier};
pub use tombstone::DeleteReport;
pub use vector::{FallbackStep, VectorSegment};
