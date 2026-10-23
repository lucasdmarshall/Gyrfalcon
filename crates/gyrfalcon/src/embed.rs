//! Embedders. `OrtEmbedder` is the production default; `HashEmbedder` is the model-free
//! stand-in an index uses until a session is attached.

use std::path::Path;
use std::sync::{Arc, Mutex};

use crate::error::{mismatch, Error, IncompatibleField, Result};
use crate::ort_session::OrtModel;
use crate::prepare::tokenize;

pub trait Embedder: Send + Sync {
    /// Embeds `embed_text` values unchanged. Returns one `f32` vector of `dimension()` each.
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn model_id(&self) -> &str;
    fn dimension(&self) -> u32;
    /// Texts a recording test double has seen. Production embedders record nothing.
    fn recorded(&self) -> Vec<String> {
        Vec::new()
    }
}

/// Deterministic feature hashing of `gyr-text-v1` tokens. No model, no network. Used by
/// tests, the bench smoke run, and `gyr --embedder hash`.
#[derive(Debug, Clone)]
pub struct HashEmbedder {
    dimension: u32,
}

impl HashEmbedder {
    pub fn new(dimension: u32) -> HashEmbedder {
        HashEmbedder { dimension }
    }

    pub fn embed_one(&self, text: &str) -> Vec<f32> {
        let dimension = self.dimension.max(1) as usize;
        let mut vector = vec![0f32; dimension];
        for token in tokenize(text) {
            let hash = fnv1a(token.text.as_bytes());
            let sign = if hash >> 63 == 0 { 1.0 } else { -1.0 };
            vector[(hash % dimension as u64) as usize] += sign;
        }
        if vector.iter().all(|v| *v == 0.0) {
            vector[0] = 1.0;
        }
        vector
    }
}

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    hash
}

impl Embedder for HashEmbedder {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|text| self.embed_one(text)).collect())
    }

    fn model_id(&self) -> &str {
        "gyr-hash"
    }

    fn dimension(&self) -> u32 {
        self.dimension
    }
}

/// The model behind an embedder: an output width and a batch run.
pub trait EmbedSession: Send + Sync {
    fn width(&self) -> usize;
    fn run(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
}

/// Test double of an ort session: a fixed width and deterministic vectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScriptedSession {
    pub width: u32,
}

impl EmbedSession for ScriptedSession {
    fn width(&self) -> usize {
        self.width as usize
    }

    fn run(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        HashEmbedder::new(self.width).embed(texts)
    }
}

impl EmbedSession for OrtModel {
    fn width(&self) -> usize {
        self.output_width().max(0) as usize
    }

    fn run(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed(texts)
    }
}

/// An `Embedder` over any session.
pub struct SessionEmbedder<S> {
    session: S,
    model_id: String,
}

impl<S: EmbedSession> SessionEmbedder<S> {
    pub fn new(session: S, model_id: impl Into<String>) -> Self {
        SessionEmbedder { session, model_id: model_id.into() }
    }

    pub fn session(&self) -> &S {
        &self.session
    }
}

impl<S: EmbedSession> Embedder for SessionEmbedder<S> {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.session.run(texts)
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn dimension(&self) -> u32 {
        self.session.width() as u32
    }
}

/// The default local embedder: ort, ONNX weights for `BAAI/bge-small-en-v1.5`, CLS pooling.
pub type OrtEmbedder = SessionEmbedder<OrtModel>;

impl OrtEmbedder {
    /// Loads `model.onnx` and `tokenizer.json` from `dir`. A missing file is `ModelUnavailable`.
    pub fn from_dir(dir: &Path, model_id: impl Into<String>) -> Result<OrtEmbedder> {
        Ok(SessionEmbedder::new(OrtModel::load(dir)?, model_id))
    }
}

/// Optional embedder that POSTs batches to an OpenAI-compatible embeddings URL. Off unless
/// configured. Each response must have the manifest dimension.
#[derive(Debug, Clone)]
pub struct HttpEmbedder {
    url: String,
    dimension: u32,
    model: String,
    api_key: Option<String>,
}

impl HttpEmbedder {
    pub fn new(url: impl Into<String>, dimension: u32) -> HttpEmbedder {
        HttpEmbedder { url: url.into(), dimension, model: crate::DEFAULT_MODEL_ID.to_string(), api_key: None }
    }

    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Rejects a response width that is not the configured (manifest) dimension. Does not
    /// touch the network.
    pub fn check_width(&self, width: usize) -> Result<()> {
        if width != self.dimension as usize {
            return mismatch(IncompatibleField::Dimension, &self.dimension.to_string(), &width.to_string());
        }
        Ok(())
    }
}

impl Embedder for HttpEmbedder {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let body = serde_json::json!({ "model": self.model, "input": texts });
        let (status, bytes) = crate::http::post_json(&self.url, self.api_key.as_deref(), &body)
            .map_err(|err| Error::Embed(err.to_string()))?;
        if !(200..300).contains(&status) {
            return Err(Error::Embed(format!("embeddings endpoint returned {status}")));
        }
        let value: serde_json::Value = serde_json::from_slice(&bytes)?;
        let data = value["data"].as_array().ok_or_else(|| Error::Embed("response has no data array".into()))?;
        let mut vectors = Vec::with_capacity(data.len());
        for item in data {
            let vector: Vec<f32> = item["embedding"]
                .as_array()
                .ok_or_else(|| Error::Embed("item has no embedding".into()))?
                .iter()
                .map(|v| v.as_f64().unwrap_or(0.0) as f32)
                .collect();
            self.check_width(vector.len())?;
            vectors.push(vector);
        }
        Ok(vectors)
    }

    fn model_id(&self) -> &str {
        &self.model
    }

    fn dimension(&self) -> u32 {
        self.dimension
    }
}

/// Test doubles shared by the integration tests.
pub mod testing {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counts `embed` calls and delegates to `HashEmbedder`.
    #[derive(Debug, Clone)]
    pub struct CountingEmbedder {
        inner: HashEmbedder,
        calls: Arc<AtomicUsize>,
    }

    impl CountingEmbedder {
        pub fn new(dimension: u32) -> CountingEmbedder {
            CountingEmbedder { inner: HashEmbedder::new(dimension), calls: Arc::new(AtomicUsize::new(0)) }
        }

        pub fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    impl Embedder for CountingEmbedder {
        fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.inner.embed(texts)
        }

        fn model_id(&self) -> &str {
            "counting"
        }

        fn dimension(&self) -> u32 {
            self.inner.dimension
        }
    }

    /// Records every text it receives.
    #[derive(Debug, Clone)]
    pub struct RecordingEmbedder {
        inner: HashEmbedder,
        texts: Arc<Mutex<Vec<String>>>,
    }

    impl RecordingEmbedder {
        pub fn new(dimension: u32) -> RecordingEmbedder {
            RecordingEmbedder { inner: HashEmbedder::new(dimension), texts: Arc::new(Mutex::new(Vec::new())) }
        }
    }

    impl Embedder for RecordingEmbedder {
        fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.texts.lock().unwrap().extend(texts.iter().cloned());
            self.inner.embed(texts)
        }

        fn model_id(&self) -> &str {
            "recording"
        }

        fn dimension(&self) -> u32 {
            self.inner.dimension
        }

        fn recorded(&self) -> Vec<String> {
            self.texts.lock().unwrap().clone()
        }
    }
}
