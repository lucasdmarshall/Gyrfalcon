//! End-to-end lite ingest timing on a JSONL corpus, plus a per-stage breakdown.
//!
//! `cargo run --release --example profile_ingest -p gyrfalcon -- CORPUS.jsonl MODEL_DIR`
//!
//! The corpus is either Open RAG Bench chunks (`doc_key`, `text`) or a BEIR corpus
//! (`_id`, `title`, `text`). MODEL_DIR is a Model2Vec model such as potion-retrieval-32M.
use std::time::Instant;

use gyrfalcon::{Document, Embedder, Index, IndexConfig, StaticEmbedder};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[1]).unwrap();
    let docs: Vec<Document> = text
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .map(|v| {
            let id = v["doc_key"].as_str().or(v["_id"].as_str()).unwrap_or("x");
            let title = v["title"].as_str().unwrap_or("").trim();
            let body = v["text"].as_str().unwrap_or("");
            Document::body(id, if title.is_empty() { body.to_string() } else { format!("{title}\n\n{body}") })
        })
        .collect();
    let bytes: usize = docs.iter().map(|d| d.text.len()).sum();
    let mb = bytes as f64 / 1e6;
    println!("{} docs, {:.1} MB", docs.len(), mb);

    let dir = std::path::Path::new(&args[2]);
    let embedder = StaticEmbedder::from_dir(dir, "x").unwrap();
    let dimension = embedder.dimension();

    // The VM is noisy: every figure is the best of three runs.
    let best = |f: &mut dyn FnMut()| {
        (0..3)
            .map(|_| {
                let t = Instant::now();
                f();
                t.elapsed().as_secs_f64()
            })
            .fold(f64::INFINITY, f64::min)
    };
    let mut prepared = Vec::new();
    let s = best(&mut || prepared = docs.iter().map(|d| gyrfalcon::prepare(&d.text)).collect());
    println!("stage prepare (1 thread)      {s:7.2} s {:7.1} MB/s", mb / s);

    let texts: Vec<String> = prepared.iter().map(|p| p.embed_text.clone()).collect();
    let s = best(&mut || {
        for batch in texts.chunks(32) {
            embedder.embed(batch).unwrap();
        }
    });
    println!("stage static embed (batch 32) {s:7.2} s {:7.1} MB/s", mb / s);

    let config = IndexConfig { dimension, model_id: gyrfalcon::STATIC_MODEL_ID.into(), ..IndexConfig::test_default() };
    let embedder = std::sync::Arc::new(embedder);
    let s = best(&mut || {
        let mut index = Index::create_in_temp(&config).unwrap();
        index.set_embedder(Box::new(Shared(embedder.clone())));
        index.auto_merge = false;
        for batch in docs.chunks(10_000) {
            index.index_sections(batch).unwrap();
        }
    });
    println!("END-TO-END lite index_sections {s:7.2} s {:7.1} MB/s {:8.0} docs/s", mb / s, docs.len() as f64 / s);
}

/// Lets the timed runs share one loaded model.
struct Shared(std::sync::Arc<StaticEmbedder>);

impl Embedder for Shared {
    fn embed(&self, texts: &[String]) -> gyrfalcon::Result<Vec<Vec<f32>>> {
        self.0.embed(texts)
    }

    fn model_id(&self) -> &str {
        self.0.model_id()
    }

    fn dimension(&self) -> u32 {
        self.0.dimension()
    }
}
