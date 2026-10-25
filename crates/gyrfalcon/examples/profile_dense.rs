//! Dense-leg evidence: the engine's graph-free cascade against an exact scan and USearch HNSW,
//! on real embeddings.
//!
//! `cargo run --release --example profile_dense -p gyrfalcon -- BEIR_DIR MODEL_DIR [SEGMENT_ROWS]`
//!
//! MODEL_DIR is a Model2Vec directory or an ONNX one (`model.onnx`, needs `ORT_DYLIB_PATH`).
//! Embeddings are cached in the temp directory. Ground truth is the exact int8 cosine top 100,
//! which is what the stored rows can express. Each variant reports build time, query latency
//! (best of 3), and recall@10/50/100 against it.
use std::time::Instant;

use gyrfalcon::{Document, Embedder, Index, IndexConfig, Quantization, StaticEmbedder};

const K: usize = 100;

fn quantize(v: &[f32]) -> Vec<i8> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let scale = if norm > 0.0 { 127.0 / norm } else { 0.0 };
    v.iter().map(|x| (x * scale).round().clamp(-127.0, 127.0) as i8).collect()
}

fn cosine_i8(a: &[i8], b: &[i8]) -> f32 {
    let (mut dot, mut na, mut nb) = (0i32, 0i32, 0i32);
    for (x, y) in a.iter().zip(b) {
        dot += i32::from(*x) * i32::from(*y);
        na += i32::from(*x) * i32::from(*x);
        nb += i32::from(*y) * i32::from(*y);
    }
    1.0 - dot as f32 / ((na as f32).sqrt() * (nb as f32).sqrt()).max(f32::MIN_POSITIVE)
}

fn recall(found: &[u64], truth: &[u64], at: usize) -> f64 {
    let truth: std::collections::HashSet<_> = truth.iter().take(at).collect();
    found.iter().take(at).filter(|id| truth.contains(id)).count() as f64 / truth.len().max(1) as f64
}

fn measure(name: &str, build_s: f64, truth: &[Vec<u64>], mut search: impl FnMut(usize) -> Vec<u64>) {
    let queries = truth.len();
    let mut best = vec![f64::INFINITY; queries];
    let mut found = vec![Vec::new(); queries];
    for _ in 0..3 {
        for q in 0..queries {
            let t = Instant::now();
            found[q] = search(q);
            best[q] = best[q].min(t.elapsed().as_secs_f64() * 1e6);
        }
    }
    let mut sorted = best.clone();
    sorted.sort_by(f64::total_cmp);
    let at = |k: usize| (0..queries).map(|q| recall(&found[q], &truth[q], k)).sum::<f64>() / queries as f64;
    println!(
        "{name:<30} build {build_s:7.3} s  mean {:6.0} us  p95 {:6.0} us  R@10 {:.4}  R@50 {:.4}  R@100 {:.4}",
        best.iter().sum::<f64>() / queries as f64,
        sorted[(queries - 1) * 95 / 100],
        at(10),
        at(50),
        at(100)
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let dir = std::path::Path::new(&args[1]);
    let model = std::path::Path::new(&args[2]);
    let segment_rows: usize = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(100_000);
    let read = |name: &str| -> Vec<serde_json::Value> {
        std::fs::read_to_string(dir.join(name)).unwrap().lines().filter_map(|l| serde_json::from_str(l).ok()).collect()
    };
    let texts: Vec<String> = read("corpus.jsonl")
        .iter()
        .map(|v| {
            let title = v["title"].as_str().unwrap_or("").trim();
            let body = v["text"].as_str().unwrap_or("");
            if title.is_empty() {
                body.to_string()
            } else {
                format!("{title}\n\n{body}")
            }
        })
        .collect();
    let mut query_texts: Vec<String> =
        read("queries.jsonl").iter().map(|v| v["text"].as_str().unwrap().to_string()).collect();
    query_texts.truncate(1000);

    let embedder: Box<dyn Embedder> = if model.join("model.onnx").is_file() {
        Box::new(gyrfalcon::OrtEmbedder::from_dir(model, "x").unwrap())
    } else {
        Box::new(StaticEmbedder::from_dir(model, "x").unwrap())
    };
    let cache = |suffix: &str| {
        std::env::temp_dir().join(format!(
            "gyr-dense-{}-{}.{suffix}.f32",
            dir.file_name().unwrap().to_string_lossy(),
            model.file_name().unwrap().to_string_lossy()
        ))
    };
    let embed_all = |texts: &[String], suffix: &str| -> Vec<Vec<f32>> {
        let path = cache(suffix);
        if let Ok(bytes) = std::fs::read(&path) {
            let dim = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
            let floats: Vec<f32> =
                bytes[4..].chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            return floats.chunks(dim).map(|c| c.to_vec()).collect();
        }
        let t = Instant::now();
        let out: Vec<Vec<f32>> = texts.chunks(64).flat_map(|b| embedder.embed(b).unwrap()).collect();
        eprintln!("embedded {} texts in {:.1} s", texts.len(), t.elapsed().as_secs_f64());
        let mut bytes = (out[0].len() as u32).to_le_bytes().to_vec();
        for x in out.iter().flatten() {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(&path, bytes).unwrap();
        out
    };
    let raw = embed_all(&texts, "corpus");
    let raw_queries = embed_all(&query_texts, "queries");
    let rows: Vec<Vec<i8>> = raw.iter().map(|v| quantize(v)).collect();
    let queries: Vec<Vec<i8>> = raw_queries.iter().map(|v| quantize(v)).collect();
    let dim = rows[0].len();
    println!("{} rows, {} queries, {dim} dims", rows.len(), queries.len());

    let exact = |q: &[i8]| -> Vec<u64> {
        let mut scored: Vec<(u64, f32)> = rows.iter().enumerate().map(|(i, r)| (i as u64, cosine_i8(q, r))).collect();
        scored.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        scored.truncate(K);
        scored.into_iter().map(|(i, _)| i).collect()
    };
    let truth: Vec<Vec<u64>> = queries.iter().map(|q| exact(q)).collect();
    measure("flat int8 exact", 0.0, &truth, |q| exact(&queries[q]));

    // USearch HNSW as format version 2 built it.
    let options = usearch::IndexOptions {
        dimensions: dim,
        metric: usearch::MetricKind::Cos,
        quantization: usearch::ScalarKind::I8,
        connectivity: 16,
        expansion_add: 128,
        expansion_search: 64,
        multi: false,
    };
    let hnsw = usearch::Index::new(&options).unwrap();
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    hnsw.reserve_capacity_and_threads(rows.len(), threads).unwrap();
    let all: Vec<usize> = (0..rows.len()).collect();
    let t = Instant::now();
    std::thread::scope(|s| {
        for part in all.chunks(rows.len().div_ceil(threads)) {
            let (hnsw, rows) = (&hnsw, &rows);
            s.spawn(move || part.iter().for_each(|&i| hnsw.add::<i8>(i as u64, &rows[i]).unwrap()));
        }
    });
    let build = t.elapsed().as_secs_f64();
    measure("usearch hnsw M16 ef128/64", build, &truth, |q| hnsw.search::<i8>(&queries[q], K).unwrap().keys);

    // The engine, through the public API; chunk id i is row i.
    let config = IndexConfig { dimension: dim as u32, quantization: Quantization::I8, ..IndexConfig::test_default() };
    let mut index = Index::create_in_temp(&config).unwrap();
    index.auto_merge = false;
    let docs: Vec<(Document, Vec<f32>)> =
        raw.iter().enumerate().map(|(i, v)| (Document::body(format!("d{i}"), "x"), v.clone())).collect();
    let t = Instant::now();
    for batch in docs.chunks(segment_rows) {
        index.index_documents_with_vectors(batch).unwrap();
    }
    let build = t.elapsed().as_secs_f64();
    let segments = index.manifest().sealed_segments.len();
    measure(&format!("gyrfalcon cascade, {segments} seg"), build, &truth, |q| {
        index.vector_search(&raw_queries[q], K).unwrap().ids
    });
    // The retrieve path asks each leg for LEG_LIMIT (50); R@100 is not meaningful here.
    measure(&format!("gyrfalcon cascade limit 50, {segments} seg"), build, &truth, |q| {
        index.vector_search(&raw_queries[q], 50).unwrap().ids
    });
}
