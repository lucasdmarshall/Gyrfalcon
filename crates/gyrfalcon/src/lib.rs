//! Gyrfalcon: a sealed-segment hybrid retrieval library.
//!
//! One process owns one collection. `index_documents` seals immutable segment directories
//! and publishes `manifest.json` by rename.

mod chunk;
mod error;
mod index;
mod manifest;
mod prepare;
mod segment;
mod store;

pub use chunk::{chunk, chunk_with, Span};
pub use error::{Error, IncompatibleField, Result};
pub use index::{Document, Index, IndexReport, FAN_OUT_WARNING, SEAL_EVERY};
pub use manifest::{IndexConfig, Manifest, Quantization, ANALYZER_ID, DEFAULT_DIMENSION, DEFAULT_MODEL_ID, DEFAULT_RERANKER_ID};
pub use prepare::{prepare, prepare_with_prefix, PreparedText, Token};
