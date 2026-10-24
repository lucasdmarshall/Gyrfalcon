//! Gyrfalcon: a sealed-segment hybrid retrieval library.
//!
//! One process owns one collection. `index_documents` seals immutable segment directories
//! and publishes `manifest.json` by rename.

mod answer;
pub mod bench;
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
mod static_embed;
mod store;
mod tombstone;
mod vector;

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
pub use fuse::rrf;
pub use index::{Document, Index, IndexReport, FAN_OUT_WARNING, SEAL_EVERY};
pub use lexical::{GyrTextTokenizer, LexicalIndex, ScoredId, TantivyLexical};
pub use manifest::{
    IndexConfig, Manifest, Quantization, ANALYZER_ID, DEFAULT_DIMENSION, DEFAULT_MODEL_ID, DEFAULT_RERANKER_ID,
    STATIC_MODEL_ID,
};
pub use merge::{dead_ratio_target, select_tier};
pub use ort_session::OrtModel;
pub use prepare::{prepare, prepare_with_prefix, PreparedText, Token};
pub use rerank::{NoopReranker, OrtReranker, Reranker, ScriptedReranker, RERANK_K};
pub use retrieve::{DenseHits, Hit, Hits, RetrieveOptions, LEG_LIMIT};
pub use static_embed::{StaticEmbedder, STATIC_MAX_TOKENS};
pub use tombstone::DeleteReport;
pub use vector::{FallbackStep, VectorSegment};
