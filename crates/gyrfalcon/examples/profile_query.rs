//! BM25 leg latency and nDCG@10 on a BEIR dataset, plus every query's top 100 for diffing
//! two builds.
//!
//! `cargo run --release --example profile_query -p gyrfalcon -- BEIR_DIR [TOP100_OUT]`
//!
//! BEIR_DIR holds `corpus.jsonl`, `queries.jsonl`, and `qrels/test.tsv`. Ingest uses the hash
//! embedder, so only the lexical leg is meaningful. The VM is noisy: latency is the best of
//! three passes.
use std::collections::HashMap;
use std::io::Write;
use std::time::Instant;

use gyrfalcon::{Document, Index, IndexConfig};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::Path::new(&args[1]);
    let read = |name: &str| -> Vec<serde_json::Value> {
        std::fs::read_to_string(dir.join(name)).unwrap().lines().filter_map(|l| serde_json::from_str(l).ok()).collect()
    };
    let corpus = read("corpus.jsonl");
    let keys: Vec<String> = corpus.iter().map(|v| v["_id"].as_str().unwrap().to_string()).collect();
    let docs: Vec<Document> = corpus
        .iter()
        .map(|v| {
            let title = v["title"].as_str().unwrap_or("").trim();
            let body = v["text"].as_str().unwrap_or("");
            let text = if title.is_empty() { body.to_string() } else { format!("{title}\n\n{body}") };
            Document::body(v["_id"].as_str().unwrap(), text)
        })
        .collect();
    let mut qrels: HashMap<String, HashMap<String, u32>> = HashMap::new();
    for line in std::fs::read_to_string(dir.join("qrels/test.tsv")).unwrap().lines().skip(1) {
        let parts: Vec<&str> = line.split('\t').collect();
        qrels.entry(parts[0].into()).or_default().insert(parts[1].into(), parts[2].parse().unwrap());
    }
    let queries: Vec<(String, String)> = read("queries.jsonl")
        .iter()
        .map(|v| (v["_id"].as_str().unwrap().to_string(), v["text"].as_str().unwrap().to_string()))
        .filter(|(id, _)| qrels.contains_key(id))
        .collect();

    let mut index = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    index.auto_merge = false;
    let t = Instant::now();
    for batch in docs.chunks(10_000) {
        index.index_sections(batch).unwrap();
    }
    println!(
        "{} docs in {} segments, ingest {:.2} s",
        docs.len(),
        index.manifest().sealed_segments.len(),
        t.elapsed().as_secs_f64()
    );

    let prepared: Vec<_> = queries.iter().map(|(_, text)| gyrfalcon::prepare(text).tokens).collect();
    let mut best = vec![f64::INFINITY; queries.len()];
    let mut results = Vec::new();
    for _ in 0..3 {
        results.clear();
        for (slot, tokens) in best.iter_mut().zip(&prepared) {
            let t = Instant::now();
            let hits = index.lexical_search(tokens, 100).unwrap();
            *slot = slot.min(t.elapsed().as_secs_f64() * 1e6);
            results.push(hits);
        }
    }
    let mut sorted = best.clone();
    sorted.sort_by(f64::total_cmp);
    let pct = |p: f64| sorted[((sorted.len() - 1) as f64 * p) as usize];
    println!(
        "{} queries: mean {:.0} us  p50 {:.0}  p95 {:.0}  p99 {:.0}  max {:.0}",
        queries.len(),
        best.iter().sum::<f64>() / best.len() as f64,
        pct(0.5),
        pct(0.95),
        pct(0.99),
        pct(1.0)
    );

    // nDCG@10 with graded gains, as pytrec_eval computes it; chunk id i is corpus line i.
    let mut ndcg = 0.0;
    for ((id, _), hits) in queries.iter().zip(&results) {
        let judged = &qrels[id];
        let dcg: f64 = hits
            .iter()
            .take(10)
            .enumerate()
            .map(|(rank, hit)| {
                let gain = judged.get(&keys[hit.chunk_id as usize]).copied().unwrap_or(0) as f64;
                gain / (rank as f64 + 2.0).log2()
            })
            .sum();
        let mut ideal: Vec<u32> = judged.values().copied().collect();
        ideal.sort_unstable_by(|a, b| b.cmp(a));
        let idcg: f64 = ideal.iter().take(10).enumerate().map(|(r, g)| *g as f64 / (r as f64 + 2.0).log2()).sum();
        ndcg += if idcg > 0.0 { dcg / idcg } else { 0.0 };
    }
    println!("BM25 nDCG@10 {:.4}", ndcg / queries.len() as f64);

    if let Some(out) = args.get(2) {
        let mut file = std::io::BufWriter::new(std::fs::File::create(out).unwrap());
        for ((id, _), hits) in queries.iter().zip(&results) {
            let line: Vec<String> = hits.iter().map(|h| format!("{}:{:.4}", h.chunk_id, h.score)).collect();
            writeln!(file, "{id}\t{}", line.join(" ")).unwrap();
        }
    }
}
