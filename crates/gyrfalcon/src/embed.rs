//! Embedders. `OrtEmbedder` is the production default; `HashEmbedder` is the model-free
//! stand-in an index uses until a session is attached.

use std::sync::{Arc, Mutex};

use crate::error::Result;
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
