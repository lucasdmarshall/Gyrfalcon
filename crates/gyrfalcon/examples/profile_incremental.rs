//! Incremental ingest: the same corpus indexed in batches of different sizes, each batch one
//! `index_sections` call (one seal and publish), as an application adding documents would.
//!
//! `cargo run --release --example profile_incremental -p gyrfalcon -- CORPUS.jsonl MODEL_DIR BATCH...`
use std::time::Instant;

use gyrfalcon::{Document, Embedder, Index, IndexConfig, StaticEmbedder};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let docs: Vec<Document> = std::fs::read_to_string(&args[1])
        .unwrap()
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .map(|v| {
            let id = v["doc_key"].as_str().or(v["_id"].as_str()).unwrap_or("x");
            let title = v["title"].as_str().unwrap_or("").trim();
            let body = v["text"].as_str().unwrap_or("");
            Document::body(id, if title.is_empty() { body.to_string() } else { format!("{title}\n\n{body}") })
        })
        .collect();
    let embedder = std::sync::Arc::new(StaticEmbedder::from_dir(std::path::Path::new(&args[2]), "x").unwrap());
    for batch in args[3..].iter().map(|b| b.parse::<usize>().unwrap()) {
        let config = IndexConfig {
            dimension: embedder.dimension(),
            model_id: gyrfalcon::STATIC_MODEL_ID.into(),
            ..IndexConfig::test_default()
        };
        let mut index = Index::create_in_temp(&config).unwrap();
        index.set_embedder(Box::new(Shared(embedder.clone())));
        let start = Instant::now();
        let mut last = Instant::now();
        let mut done = 0;
        let mut slowest = 0f64;
        for (i, chunk) in docs.chunks(batch).enumerate() {
            let t = Instant::now();
            index.index_sections(chunk).unwrap();
            slowest = slowest.max(t.elapsed().as_secs_f64());
            done += chunk.len();
            if last.elapsed().as_secs_f64() > 10.0 || done == docs.len() {
                println!(
                    "batch {batch:6}: {done:6} docs, {:7.1} s, last call {:7.1} ms, segments {}",
                    start.elapsed().as_secs_f64(),
                    t.elapsed().as_secs_f64() * 1e3,
                    index.manifest().sealed_segments.len()
                );
                last = Instant::now();
            }
            if start.elapsed().as_secs_f64() > 240.0 {
                println!("batch {batch:6}: stopped after {i} calls, {done} docs");
                break;
            }
        }
        println!(
            "batch {batch:6}: TOTAL {:.1} s, {:.0} docs/s, slowest call {:.1} ms",
            start.elapsed().as_secs_f64(),
            done as f64 / start.elapsed().as_secs_f64(),
            slowest * 1e3
        );
    }
}

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
