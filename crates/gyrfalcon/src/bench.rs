//! Benchmark report structs and the harness math. The `gyr bench` commands fill these.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::error::{Error, Result};
use crate::vector::FallbackStep;

/// Machine and index identity every report records.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BenchEnvironment {
    pub cpu_model: String,
    pub physical_cores: usize,
    pub ort_intra_threads: usize,
    pub model_id: String,
    pub dimension: u32,
    pub quantization: String,
    pub analyzer_id: String,
    pub git_revision: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_revision: Option<String>,
}

impl BenchEnvironment {
    pub fn detect(config: &crate::IndexConfig) -> BenchEnvironment {
        let cpuinfo = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
        let cpu_model = cpuinfo
            .lines()
            .find(|line| line.starts_with("model name"))
            .and_then(|line| line.split_once(':'))
            .map(|(_, value)| value.trim().to_string())
            .unwrap_or_else(|| std::env::consts::ARCH.to_string());
        let mut cores: Vec<(String, String)> = Vec::new();
        let (mut physical, mut core) = (String::new(), String::new());
        for line in cpuinfo.lines() {
            if let Some((key, value)) = line.split_once(':') {
                match key.trim() {
                    "physical id" => physical = value.trim().to_string(),
                    "core id" => core = value.trim().to_string(),
                    _ => {}
                }
            } else if !physical.is_empty() || !core.is_empty() {
                cores.push((std::mem::take(&mut physical), std::mem::take(&mut core)));
            }
        }
        cores.sort();
        cores.dedup();
        let physical_cores = if cores.is_empty() {
            std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
        } else {
            cores.len()
        };
        let git_revision = std::env::var("GYR_GIT_REVISION").ok().unwrap_or_else(|| {
            std::process::Command::new("git")
                .args(["rev-parse", "HEAD"])
                .output()
                .ok()
                .filter(|out| out.status.success())
                .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_string())
                .unwrap_or_else(|| "unknown".into())
        });
        BenchEnvironment {
            cpu_model,
            physical_cores,
            ort_intra_threads: crate::ort_session::intra_threads(),
            model_id: config.model_id.clone(),
            dimension: config.dimension,
            quantization: config.quantization.as_str().to_string(),
            analyzer_id: config.analyzer_id.clone(),
            git_revision,
            dataset_revision: None,
        }
    }
}

/// Percent of timed queries per deepest HNSW step. All six keys are present and sum to 100.
pub fn fallback_pct(steps: &[FallbackStep]) -> BTreeMap<String, f64> {
    let mut out: BTreeMap<String, f64> = FallbackStep::ALL.iter().map(|step| (step.label().to_string(), 0.0)).collect();
    if steps.is_empty() {
        out.insert(FallbackStep::E64.label().to_string(), 100.0);
        return out;
    }
    for step in steps {
        *out.get_mut(step.label()).unwrap() += 1.0;
    }
    for value in out.values_mut() {
        *value = 100.0 * *value / steps.len() as f64;
    }
    out
}

/// Nearest-rank percentile of an ascending list.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = ((p / 100.0) * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LatencyColumn {
    /// 1 is the single-flight column.
    pub in_flight: usize,
    pub p50_ms: f64,
    pub p99_ms: f64,
    pub qps: f64,
    pub fallback_pct: BTreeMap<String, f64>,
}

impl LatencyColumn {
    pub fn from_samples(
        in_flight: usize,
        mut millis: Vec<f64>,
        elapsed_seconds: f64,
        steps: &[FallbackStep],
    ) -> LatencyColumn {
        millis.sort_by(|a, b| a.total_cmp(b));
        LatencyColumn {
            in_flight,
            p50_ms: percentile(&millis, 50.0),
            p99_ms: percentile(&millis, 99.0),
            qps: if elapsed_seconds > 0.0 { millis.len() as f64 / elapsed_seconds } else { 0.0 },
            fallback_pct: fallback_pct(steps),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RetrieveReport {
    pub environment: BenchEnvironment,
    pub rerank: bool,
    pub segment_count: usize,
    pub tombstone_count: usize,
    pub warmup: usize,
    pub timed: usize,
    pub chunks: usize,
    /// The single-flight column's step distribution.
    pub fallback_pct: BTreeMap<String, f64>,
    pub single_flight: LatencyColumn,
    pub concurrent: Vec<LatencyColumn>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub time_to_request: Option<TimeToRequestReport>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct TimeToRequestReport {
    /// Always `discarded`: a local server accepts the bytes and runs no model.
    pub llm: String,
    pub columns: Vec<LatencyColumn>,
}

/// The percentage function on a scripted list of deepest steps.
pub fn bench_retrieve_scripted(steps: &[FallbackStep]) -> RetrieveReport {
    let pct = fallback_pct(steps);
    RetrieveReport {
        timed: steps.len(),
        fallback_pct: pct.clone(),
        single_flight: LatencyColumn { in_flight: 1, fallback_pct: pct, ..LatencyColumn::default() },
        ..RetrieveReport::default()
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct IngestRow {
    pub headline: bool,
    pub auto_merge: bool,
    pub contextual_prefix: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_llm: Option<String>,
    pub files: usize,
    pub chunks: usize,
    pub docs_per_sec: f64,
    pub chunks_per_sec: f64,
    pub seal_wall_seconds: f64,
    pub merge_wall_seconds: f64,
    pub segment_count: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct IngestReport {
    pub environment: BenchEnvironment,
    pub rows: Vec<IngestRow>,
}

pub const RAM_KEYS: [&str; 10] = [
    "usearch_bytes",
    "vectors_bin_bytes",
    "tantivy_bytes",
    "catalog_bytes",
    "tombstone_bytes",
    "bm25_bytes",
    "rss_baseline",
    "rss_index_open",
    "rss_model_loaded",
    "rss_model_dropped",
];

/// The key set of the RAM report, every value 0. `gyr bench ram` fills the values.
pub fn bench_ram_shape() -> Map<String, Value> {
    RAM_KEYS.iter().map(|key| (key.to_string(), json!(0u64))).collect()
}

/// Bytes of the collection's files, grouped the way the RAM report lists them.
pub fn collection_bytes(root: &std::path::Path) -> Map<String, Value> {
    fn size(path: &std::path::Path) -> u64 {
        match std::fs::metadata(path) {
            Ok(meta) if meta.is_dir() => {
                std::fs::read_dir(path).map(|entries| entries.flatten().map(|e| size(&e.path())).sum()).unwrap_or(0)
            }
            Ok(meta) => meta.len(),
            Err(_) => 0,
        }
    }
    let mut out = bench_ram_shape();
    let segments = root.join("segments");
    let (mut usearch, mut vectors, mut tantivy) = (0, 0, 0);
    if let Ok(entries) = std::fs::read_dir(&segments) {
        for entry in entries.flatten() {
            usearch += size(&entry.path().join("usearch.index"));
            vectors += size(&entry.path().join("vectors.bin"));
            tantivy += size(&entry.path().join("tantivy"));
        }
    }
    out.insert("usearch_bytes".into(), json!(usearch));
    out.insert("vectors_bin_bytes".into(), json!(vectors));
    out.insert("tantivy_bytes".into(), json!(tantivy));
    out.insert("catalog_bytes".into(), json!(size(&root.join("catalog"))));
    out.insert("tombstone_bytes".into(), json!(size(&root.join("tombstones"))));
    out.insert("bm25_bytes".into(), json!(size(&root.join("bm25"))));
    out
}

/// Resident set size of this process in bytes (Linux), else 0.
pub fn rss_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find(|line| line.starts_with("VmRSS:"))
                .and_then(|line| line.split_whitespace().nth(1))
                .and_then(|kb| kb.parse::<u64>().ok())
        })
        .map(|kb| kb * 1024)
        .unwrap_or(0)
}

/// Keeps the first occurrence of each `doc_key` and drops later copies, preserving order.
pub fn dedupe_doc_keys(keys: &[String]) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    keys.iter().filter(|key| seen.insert(key.as_str())).cloned().collect()
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct RetrievalScore {
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub mrr: f64,
    pub ndcg_at_10: f64,
}

/// Per-question metrics for one relevant `doc_key` at 1-based rank `r`.
pub fn retrieval_metrics(rank: Option<u32>) -> RetrievalScore {
    match rank {
        None => RetrievalScore::default(),
        Some(r) => RetrievalScore {
            recall_at_5: if r <= 5 { 1.0 } else { 0.0 },
            recall_at_10: if r <= 10 { 1.0 } else { 0.0 },
            mrr: 1.0 / r as f64,
            ndcg_at_10: if r <= 10 { 1.0 / ((r + 1) as f64).log2() } else { 0.0 },
        },
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct QuestionResult {
    pub query_id: String,
    pub relevant: String,
    /// Ordered retrieval output before the harness dedupe; at least 10.
    pub doc_keys: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub answer: Option<String>,
    /// `Some(true)` pass, `Some(false)` fail, `None` judge error or no answer pass.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pass: Option<bool>,
}

/// One system's run on the text-only slice. Adapters write this schema.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct AccuracyReport {
    pub system: String,
    pub generator_id: String,
    pub answer_prompt_sha256: String,
    pub judge_id: String,
    pub question_id_sha256: String,
    pub dataset_revision: String,
    pub chunk_count: usize,
    pub questions: Vec<QuestionResult>,
    #[serde(default)]
    pub passes: usize,
    #[serde(default)]
    pub judge_errors: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrievalRow {
    pub system: String,
    #[serde(flatten)]
    pub score: RetrievalScore,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnswerRow {
    pub system: String,
    pub passes: usize,
    pub judge_errors: usize,
    pub pass_rate: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AccuracyComparison {
    pub retrieval: Option<Vec<RetrievalRow>>,
    /// Present only when every report shares the generator, prompt, judge, questions, and dataset.
    pub answer: Option<Vec<AnswerRow>>,
    pub json: Value,
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// `question_id_sha256`: the hash of the sorted query UUIDs, one per line.
pub fn question_id_sha256(ids: &[String]) -> String {
    let mut sorted = ids.to_vec();
    sorted.sort();
    sha256_hex(sorted.join("\n").as_bytes())
}

/// Means over every question of the harness-deduped list.
pub fn score_report(report: &AccuracyReport) -> RetrievalScore {
    let mut total = RetrievalScore::default();
    for question in &report.questions {
        let deduped = dedupe_doc_keys(&question.doc_keys);
        let rank = deduped.iter().position(|key| *key == question.relevant).map(|at| at as u32 + 1);
        let score = retrieval_metrics(rank);
        total.recall_at_5 += score.recall_at_5;
        total.recall_at_10 += score.recall_at_10;
        total.mrr += score.mrr;
        total.ndcg_at_10 += score.ndcg_at_10;
    }
    let n = report.questions.len().max(1) as f64;
    RetrievalScore {
        recall_at_5: total.recall_at_5 / n,
        recall_at_10: total.recall_at_10 / n,
        mrr: total.mrr / n,
        ndcg_at_10: total.ndcg_at_10 / n,
    }
}

/// Builds the retrieval table and, when the generation setup matches, the answer table.
/// Aborts when chunk counts differ or any list is shorter than 10 before dedupe. No field
/// holds a previously published full-set score.
pub fn compare_accuracy(reports: &[AccuracyReport]) -> Result<AccuracyComparison> {
    let abort = |message: String| Err(Error::Io(std::io::Error::other(message)));
    let Some(first) = reports.first() else { return abort("no reports to compare".into()) };
    if let Some(other) = reports.iter().find(|r| r.chunk_count != first.chunk_count) {
        return abort(format!(
            "chunk counts differ: {} has {}, {} has {}",
            first.system, first.chunk_count, other.system, other.chunk_count
        ));
    }
    for report in reports {
        if let Some(short) = report.questions.iter().find(|q| q.doc_keys.len() < 10) {
            return abort(format!(
                "{} returned {} doc_keys for {}",
                report.system,
                short.doc_keys.len(),
                short.query_id
            ));
        }
    }
    let retrieval: Vec<RetrievalRow> =
        reports.iter().map(|r| RetrievalRow { system: r.system.clone(), score: score_report(r) }).collect();
    let same_setup = reports.iter().all(|r| {
        r.generator_id == first.generator_id
            && r.answer_prompt_sha256 == first.answer_prompt_sha256
            && r.judge_id == first.judge_id
            && r.question_id_sha256 == first.question_id_sha256
            && r.dataset_revision == first.dataset_revision
    });
    let answer = same_setup.then(|| {
        reports
            .iter()
            .map(|r| AnswerRow {
                system: r.system.clone(),
                passes: r.passes,
                judge_errors: r.judge_errors,
                pass_rate: r.passes as f64 / r.questions.len().max(1) as f64,
            })
            .collect::<Vec<_>>()
    });
    let mut json = json!({
        "questions": first.questions.len(),
        "dataset_revision": first.dataset_revision,
        "question_id_sha256": first.question_id_sha256,
        "chunk_count": first.chunk_count,
        "retrieval": retrieval,
    });
    if let Some(answer) = &answer {
        json["answer"] = json!({
            "generator_id": first.generator_id,
            "judge_id": first.judge_id,
            "answer_prompt_sha256": first.answer_prompt_sha256,
            "rows": answer,
        });
    }
    Ok(AccuracyComparison { retrieval: Some(retrieval), answer, json })
}

/// Graded relevance of one query: document id to gain. Gains of 0 or less are not relevant.
pub type Qrels = std::collections::HashMap<String, i64>;

/// nDCG@k with linear gains and `log2(rank + 1)` discounts, as `pytrec_eval` computes it for
/// BEIR and MTEB.
pub fn ndcg_at(ranked: &[String], qrels: &Qrels, k: usize) -> f64 {
    let gain = |id: &String| qrels.get(id).copied().unwrap_or(0).max(0) as f64;
    let dcg: f64 = ranked.iter().take(k).enumerate().map(|(i, id)| gain(id) / ((i + 2) as f64).log2()).sum();
    let mut ideal: Vec<f64> = qrels.values().map(|g| (*g).max(0) as f64).filter(|g| *g > 0.0).collect();
    ideal.sort_by(|a, b| b.total_cmp(a));
    let idcg: f64 = ideal.iter().take(k).enumerate().map(|(i, g)| g / ((i + 2) as f64).log2()).sum();
    if idcg > 0.0 {
        dcg / idcg
    } else {
        0.0
    }
}

/// Share of relevant documents found in the top `k`.
pub fn recall_at(ranked: &[String], qrels: &Qrels, k: usize) -> f64 {
    let relevant = qrels.values().filter(|g| **g > 0).count();
    if relevant == 0 {
        return 0.0;
    }
    let found = ranked.iter().take(k).filter(|id| qrels.get(*id).is_some_and(|g| *g > 0)).count();
    found as f64 / relevant as f64
}

/// Reciprocal rank of the first relevant document within the top `k`.
pub fn mrr_at(ranked: &[String], qrels: &Qrels, k: usize) -> f64 {
    ranked
        .iter()
        .take(k)
        .position(|id| qrels.get(id).is_some_and(|g| *g > 0))
        .map(|at| 1.0 / (at + 1) as f64)
        .unwrap_or(0.0)
}
