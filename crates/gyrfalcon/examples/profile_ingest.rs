//! Times each ingest stage on a JSONL corpus: `cargo run --release --example profile_ingest -- CHUNKS.jsonl MODEL_DIR`.
use std::time::Instant;

use gyrfalcon::{Document, Embedder, Index, IndexConfig, StaticEmbedder};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[1]).unwrap();
    let docs: Vec<Document> = text
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .map(|v| Document::body(v["doc_key"].as_str().unwrap_or("x"), v["text"].as_str().unwrap_or("")))
        .collect();
    let bytes: usize = docs.iter().map(|d| d.text.len()).sum();
    println!("{} docs, {:.1} MB", docs.len(), bytes as f64 / 1e6);

    let t = Instant::now();
    let prepared: Vec<_> = docs.iter().map(|d| gyrfalcon::prepare(&d.text)).collect();
    println!("prepare        {:7.2} s", t.elapsed().as_secs_f64());

    let dir = std::path::Path::new(&args[2]);
    let embedder = StaticEmbedder::from_dir(dir, "x").unwrap();
    let texts: Vec<String> = prepared.iter().map(|p| p.embed_text.clone()).collect();
    let t = Instant::now();
    for batch in texts.chunks(32) {
        embedder.embed(batch).unwrap();
    }
    println!("static embed   {:7.2} s (batch 32)", t.elapsed().as_secs_f64());
    let t = Instant::now();
    embedder.embed(&texts).unwrap();
    println!("static embed   {:7.2} s (one batch)", t.elapsed().as_secs_f64());

    let config = IndexConfig { dimension: embedder.dimension(), ..IndexConfig::test_default() };
    let mut index = Index::create_in_temp(&config).unwrap();
    index.set_embedder(Box::new(gyrfalcon::HashEmbedder::new(embedder.dimension())));
    index.auto_merge = false;
    let t = Instant::now();
    index.index_sections(&docs).unwrap();
    println!("seal (hash embed, tantivy + usearch + files) {:7.2} s", t.elapsed().as_secs_f64());

    // Publish cost as the collection grows: many small batches.
    let mut index = Index::create_in_temp(&config).unwrap();
    index.set_embedder(Box::new(gyrfalcon::HashEmbedder::new(embedder.dimension())));
    index.auto_merge = false;
    let small: Vec<Document> = (0..400).map(|i| Document::body(format!("s{i}"), "tiny doc")).collect();
    for round in 0..8 {
        index.index_sections(&docs[round * 2000..(round + 1) * 2000]).unwrap();
        let t = Instant::now();
        index.index_sections(&small[round * 50..(round + 1) * 50]).unwrap();
        println!("  50-doc write after {:6} chunks: {:6.1} ms", (round + 1) * 2000, t.elapsed().as_secs_f64() * 1000.0);
    }
}
