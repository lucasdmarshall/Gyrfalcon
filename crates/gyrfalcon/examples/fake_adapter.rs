//! A stand-in adapter runner for tests. It does not index or parse anything: it copies the
//! fixture chunk count and question ids into `$BENCH_REPORTS/$BENCH_SYSTEM.json` (default
//! `bench/reports`) in the `AccuracyReport` schema, with the first ten chunk keys as every
//! question's retrieval list.

use std::path::PathBuf;

use gyrfalcon::bench::{AccuracyReport, QuestionResult};

fn lines(path: PathBuf) -> Vec<serde_json::Value> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn main() {
    let fixture = PathBuf::from(std::env::var("BENCH_FIXTURE").expect("BENCH_FIXTURE"));
    let system = std::env::var("BENCH_SYSTEM").expect("BENCH_SYSTEM");
    let reports = PathBuf::from(std::env::var("BENCH_REPORTS").unwrap_or_else(|_| "bench/reports".into()));
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(fixture.join("meta.json")).unwrap()).unwrap();
    let chunks = lines(fixture.join("chunks.jsonl"));
    let keys: Vec<String> = chunks.iter().take(10).map(|c| c["doc_key"].as_str().unwrap().to_string()).collect();
    let questions = lines(fixture.join("questions.jsonl"))
        .iter()
        .map(|q| QuestionResult {
            query_id: q["query_id"].as_str().unwrap().to_string(),
            relevant: q["relevant"].as_str().unwrap().to_string(),
            doc_keys: keys.clone(),
            ..QuestionResult::default()
        })
        .collect();
    let report = AccuracyReport {
        system: system.clone(),
        generator_id: "none".into(),
        answer_prompt_sha256: gyrfalcon::bench::sha256_hex(gyrfalcon::ANSWER_PROMPT.as_bytes()),
        judge_id: "none".into(),
        question_id_sha256: meta["question_id_sha256"].as_str().unwrap_or_default().to_string(),
        dataset_revision: meta["dataset_revision"].as_str().unwrap_or_default().to_string(),
        chunk_count: chunks.len(),
        questions,
        ..AccuracyReport::default()
    };
    std::fs::create_dir_all(&reports).unwrap();
    std::fs::write(reports.join(format!("{system}.json")), serde_json::to_vec_pretty(&report).unwrap()).unwrap();
}
