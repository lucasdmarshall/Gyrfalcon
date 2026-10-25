//! `gyr bench retrieve | ingest | ram | accuracy`. Every report is JSON.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use clap::Subcommand;
use gyrfalcon::bench::{
    collection_bytes, compare_accuracy, question_id_sha256, rss_bytes, score_report, sha256_hex, AccuracyReport,
    BenchEnvironment, IngestReport, IngestRow, LatencyColumn, QuestionResult, RetrieveReport, TimeToRequestReport,
};
use gyrfalcon::{
    complete, fill_answer_prompt, fill_judge_prompt, AnswerEvent, AnswerOptions, ChatConfig, Document, Error,
    FallbackStep, Fusion, HashEmbedder, Hit, Index, RetrieveOptions, ANSWER_PROMPT,
};
use serde_json::{json, Value};

use crate::ModelArgs;

#[derive(Subcommand)]
pub enum BenchCommand {
    /// Single-flight and concurrent retrieve clocks with fallback percentages.
    Retrieve {
        /// Seal the corpus into this many segments (auto-merge off when above 1).
        #[arg(long, default_value_t = 1)]
        segments: usize,
        /// Synthetic documents to generate when --corpus is not given.
        #[arg(long, default_value_t = 20_000)]
        docs: usize,
        /// Directory of plain-text files to index instead of synthetic documents.
        #[arg(long)]
        corpus: Option<PathBuf>,
        #[arg(long, default_value_t = 1000)]
        queries: usize,
        #[arg(long, default_value_t = 200)]
        warmup: usize,
        #[arg(long, value_delimiter = ',', default_value = "4,16,32")]
        concurrency: Vec<usize>,
        #[arg(long)]
        rerank: bool,
        /// Also clock time-to-request against a local discard server.
        #[arg(long)]
        time_to_request: bool,
        #[arg(long)]
        out: Option<PathBuf>,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Docs/sec with the merger included, plus a seal-only row.
    Ingest {
        #[arg(long, default_value_t = 16)]
        files: usize,
        #[arg(long, default_value_t = 10_000)]
        chunks_per_file: usize,
        #[arg(long)]
        out: Option<PathBuf>,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// BEIR datasets: nDCG@10, recall@100, and MRR@10 comparable with BEIR and MTEB numbers.
    Beir {
        /// Dataset directories with corpus.jsonl, queries.jsonl, and qrels/{split}.tsv.
        #[arg(long, num_args = 1.., required = true)]
        dataset: Vec<PathBuf>,
        #[arg(long, default_value = "test")]
        split: String,
        /// Fusions to score on the same index: configured, rrf, lexical, dense, score:W, rescore:W.
        #[arg(long, value_delimiter = ',', default_value = "configured")]
        fusions: Vec<String>,
        /// Ranked documents kept per query.
        #[arg(long, default_value_t = 100)]
        depth: usize,
        #[arg(long)]
        rerank: bool,
        #[arg(long)]
        out: Option<PathBuf>,
        /// Writes each ranked list as `{dataset}-{fusion}.tsv` (query, doc, rank, score) here,
        /// for scoring with another evaluator.
        #[arg(long)]
        run_dir: Option<PathBuf>,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Planner traces: every step's scores and timings per query, for replay (P1).
    Trace {
        /// BEIR dataset directories.
        #[arg(long, num_args = 1..)]
        dataset: Vec<PathBuf>,
        /// Open RAG Bench text-only fixture directory, traced as dataset `orb`.
        #[arg(long)]
        orb_fixture: Option<PathBuf>,
        #[arg(long, default_value = "bench/traces")]
        out_dir: PathBuf,
        #[arg(long, default_value_t = 100)]
        depth: usize,
        /// Cross-encoder model directory; without it traces carry no rerank scores.
        #[arg(long)]
        rerank_dir: Option<PathBuf>,
        #[arg(long, default_value_t = 32)]
        rerank_k: usize,
        #[arg(long, default_value_t = 256)]
        rerank_tokens: usize,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// Replays policies over traces and reports quality, replayed latency, and paired
    /// bootstrap intervals against a reference policy.
    Replay {
        #[arg(long, num_args = 1.., required = true)]
        traces: Vec<PathBuf>,
        #[arg(long, default_value = "validation")]
        split: String,
        /// lexical, dense, rrf, rescore:W, or rescore:W:K (rerank the top K).
        #[arg(long, value_delimiter = ',', required = true)]
        policies: Vec<String>,
        #[arg(long)]
        reference: Option<String>,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Embedder throughput on fixture chunks and single-query latency, without indexing.
    Embed {
        #[arg(long, default_value = "bench/fixtures/text-only")]
        fixture: PathBuf,
        /// Chunks to embed for the throughput figure.
        #[arg(long, default_value_t = 2000)]
        chunks: usize,
        #[arg(long, default_value_t = 32)]
        batch: usize,
        /// Questions to embed one at a time for the latency figure.
        #[arg(long, default_value_t = 500)]
        queries: usize,
        #[arg(long)]
        out: Option<PathBuf>,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// File bytes and RSS of a synthetic collection.
    Ram {
        #[arg(long, default_value_t = 1_000_000)]
        vectors: usize,
        /// Internal: measure RSS of this already-built collection in a fresh process.
        #[arg(long, hide = true)]
        measure_collection: Option<PathBuf>,
        #[arg(long)]
        out: Option<PathBuf>,
        #[command(flatten)]
        model: ModelArgs,
    },
    /// The text-only Open RAG Bench slice: --prepare, run, or --compare.
    Accuracy {
        /// Build the fixture from --dataset.
        #[arg(long)]
        prepare: bool,
        /// Compare these adapter reports.
        #[arg(long, num_args = 1..)]
        compare: Vec<PathBuf>,
        /// `official/pdf/arxiv` directory of vectara/open_ragbench.
        #[arg(long)]
        dataset: Option<PathBuf>,
        #[arg(long, default_value = "unknown")]
        dataset_revision: String,
        #[arg(long, default_value = "bench/fixtures/text-only")]
        fixture: PathBuf,
        #[arg(long, default_value_t = 1914, hide = true)]
        expected_questions: usize,
        #[arg(long, default_value = "gyrfalcon")]
        system: String,
        #[arg(long)]
        rerank: bool,
        #[arg(long, env = "GYR_JUDGE_ENDPOINT")]
        judge_endpoint: Option<String>,
        #[arg(long, env = "GYR_JUDGE_MODEL", default_value = "default")]
        judge_model: String,
        #[arg(long)]
        out: Option<PathBuf>,
        #[command(flatten)]
        model: ModelArgs,
    },
}

fn other(message: impl Into<String>) -> Error {
    Error::Io(std::io::Error::other(message.into()))
}

fn emit(value: &Value, out: Option<&Path>) -> Result<(), Error> {
    let text = serde_json::to_string_pretty(value)?;
    if let Some(path) = out {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, &text)?;
    }
    println!("{text}");
    Ok(())
}

/// Deterministic xorshift for corpus and query generation.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1)
    }

    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

const SYLLABLES: [&str; 16] =
    ["ka", "ri", "to", "mu", "sel", "van", "dor", "lin", "qua", "zen", "por", "ex", "tal", "nim", "ora", "bex"];

fn word(id: usize) -> String {
    let mut out = String::new();
    let mut n = id + 16;
    while n > 0 {
        out.push_str(SYLLABLES[n % 16]);
        n /= 16;
    }
    out
}

fn sentence(rng: &mut Rng, words: usize) -> String {
    (0..words).map(|_| word(rng.below(4000))).collect::<Vec<_>>().join(" ")
}

fn corpus(docs: usize, dir: Option<&Path>) -> Result<Vec<Document>, Error> {
    if let Some(dir) = dir {
        let mut paths: Vec<PathBuf> =
            std::fs::read_dir(dir)?.flatten().map(|e| e.path()).filter(|p| p.is_file()).collect();
        paths.sort();
        return paths
            .iter()
            .map(|path| Ok(Document::body(path.to_string_lossy(), std::fs::read_to_string(path)?)))
            .collect();
    }
    let mut rng = Rng::new(7);
    Ok((0..docs)
        .map(|i| {
            let words = 40 + rng.below(40);
            Document::body(format!("doc{i}"), format!("{}.", sentence(&mut rng, words)))
        })
        .collect())
}

/// Warmup plus timed queries drawn from the corpus, shuffled with seed 1.
fn query_list(docs: &[Document], count: usize) -> Vec<String> {
    let mut rng = Rng::new(1);
    let mut queries: Vec<String> = (0..count)
        .map(|_| {
            let doc = &docs[rng.below(docs.len())];
            let words: Vec<&str> = doc.text.split_whitespace().collect();
            let take = 2 + rng.below(4);
            let start = rng.below(words.len().saturating_sub(take).max(1));
            words.iter().skip(start).take(take).copied().collect::<Vec<_>>().join(" ")
        })
        .collect();
    for i in (1..queries.len()).rev() {
        let j = rng.below(i + 1);
        queries.swap(i, j);
    }
    queries
}

/// Indexes `docs` into `segments` sealed segments, as evenly as the document count allows.
fn build_layout(index: &mut Index, docs: &[Document], segments: usize) -> Result<(), Error> {
    if segments <= 1 {
        index.index_documents(docs)?;
        index.wait_until_merge_idle();
        return Ok(());
    }
    index.auto_merge = false;
    let groups = segments.min(docs.len()).max(1);
    for g in 0..groups {
        let start = g * docs.len() / groups;
        let end = (g + 1) * docs.len() / groups;
        index.index_documents(&docs[start..end])?;
    }
    Ok(())
}

fn timed_retrieve(index: &Index, query: &str, opts: RetrieveOptions) -> Result<(f64, FallbackStep), Error> {
    let started = Instant::now();
    let hits = index.retrieve(query, opts)?;
    Ok((started.elapsed().as_secs_f64() * 1000.0, hits.fallback))
}

/// Runs `queries` with `in_flight` workers sharing the list.
fn column<F>(in_flight: usize, queries: &[String], run: F) -> Result<LatencyColumn, Error>
where
    F: Fn(&str) -> Result<(f64, FallbackStep), Error> + Sync,
{
    let next = AtomicUsize::new(0);
    let samples = Mutex::new(Vec::with_capacity(queries.len()));
    let first_error = Mutex::new(None);
    let started = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..in_flight.max(1) {
            scope.spawn(|| loop {
                let at = next.fetch_add(1, Ordering::SeqCst);
                if at >= queries.len() {
                    return;
                }
                match run(&queries[at]) {
                    Ok(sample) => samples.lock().unwrap().push(sample),
                    Err(err) => {
                        first_error.lock().unwrap().get_or_insert(err);
                        return;
                    }
                }
            });
        }
    });
    let elapsed = started.elapsed().as_secs_f64();
    if let Some(err) = first_error.into_inner().unwrap() {
        return Err(err);
    }
    let samples = samples.into_inner().unwrap();
    let millis = samples.iter().map(|(ms, _)| *ms).collect();
    let steps: Vec<FallbackStep> = samples.iter().map(|(_, step)| *step).collect();
    Ok(LatencyColumn::from_samples(in_flight.max(1), millis, elapsed, &steps))
}

/// Accepts chat requests, reads them, and closes without running a model.
fn discard_server() -> Result<String, Error> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?;
    std::thread::spawn(move || {
        for stream in listener.incoming().flatten() {
            std::thread::spawn(move || {
                let mut reader = BufReader::new(stream);
                let mut length = 0usize;
                let mut line = String::new();
                while reader.read_line(&mut line).unwrap_or(0) > 0 {
                    if line == "\r\n" {
                        break;
                    }
                    if let Some((name, value)) = line.split_once(':') {
                        if name.eq_ignore_ascii_case("content-length") {
                            length = value.trim().parse().unwrap_or(0);
                        }
                    }
                    line.clear();
                }
                let mut body = vec![0; length];
                let _ = reader.read_exact(&mut body);
            });
        }
    });
    Ok(format!("http://{addr}/v1/chat/completions"))
}

#[allow(clippy::too_many_arguments)]
fn bench_retrieve(
    segments: usize,
    docs: usize,
    corpus_dir: Option<&Path>,
    queries: usize,
    warmup: usize,
    concurrency: &[usize],
    rerank: bool,
    time_to_request: bool,
    model: &ModelArgs,
) -> Result<Value, Error> {
    let dir = tempfile::tempdir()?;
    let mut index = model.open(&dir.path().join("collection"), true)?;
    let docs = corpus(docs, corpus_dir)?;
    build_layout(&mut index, &docs, segments)?;
    let list = query_list(&docs, warmup + queries);
    let (warm, timed) = list.split_at(warmup);
    let opts = RetrieveOptions { rerank, limit: 10 };
    for query in warm {
        index.retrieve(query, opts)?;
    }
    let single = column(1, timed, |q| timed_retrieve(&index, q, opts))?;
    let mut concurrent = Vec::new();
    for in_flight in concurrency {
        concurrent.push(column(*in_flight, timed, |q| timed_retrieve(&index, q, opts))?);
    }
    let ttr = if time_to_request {
        index.set_chat_endpoint(discard_server()?);
        let answer = |q: &str| -> Result<(f64, FallbackStep), Error> {
            let mut stream = index.answer(q, AnswerOptions { rerank, ..AnswerOptions::default() })?;
            match stream.next() {
                Some(AnswerEvent::TimeToRequest { ms }) => Ok((ms, FallbackStep::E64)),
                Some(AnswerEvent::Error(err)) => Err(err),
                _ => Err(other("answer did not report time_to_request")),
            }
        };
        let mut columns = vec![column(1, timed, answer)?, column(16, timed, answer)?];
        for column in &mut columns {
            column.fallback_pct.clear();
        }
        Some(TimeToRequestReport { llm: "discarded".into(), columns })
    } else {
        None
    };
    let report = RetrieveReport {
        environment: BenchEnvironment::detect(index.config()),
        rerank,
        segment_count: index.segment_chunk_counts().len(),
        tombstone_count: index.tombstone_len(),
        warmup,
        timed: timed.len(),
        chunks: index.segment_chunk_counts().iter().sum::<u64>() as usize,
        fallback_pct: single.fallback_pct.clone(),
        single_flight: single,
        concurrent,
        time_to_request: ttr,
    };
    Ok(serde_json::to_value(report)?)
}

/// Paragraphs of 1,500 to about 1,650 bytes: each becomes one chunk under the 2048/256 chunker.
fn ingest_file(rng: &mut Rng, chunks: usize) -> String {
    let mut paragraphs = Vec::with_capacity(chunks);
    for _ in 0..chunks {
        let mut paragraph = String::new();
        while paragraph.len() < 1_500 {
            paragraph.push_str(&sentence(rng, 12));
            paragraph.push_str(". ");
        }
        paragraphs.push(paragraph.trim_end().to_string());
    }
    paragraphs.join("\n\n")
}

fn ingest_row(files: &[PathBuf], auto_merge: bool, prefix: bool, model: &ModelArgs) -> Result<IngestRow, Error> {
    let dir = tempfile::tempdir()?;
    let mut index = model.open(&dir.path().join("collection"), true)?;
    index.auto_merge = auto_merge;
    let started = Instant::now();
    let mut chunks = 0;
    for path in files {
        let text = std::fs::read_to_string(path)?;
        let mut doc = Document::body(path.to_string_lossy(), text);
        doc.contextual_prefix = prefix;
        chunks += index.index_documents(std::slice::from_ref(&doc))?.chunk_ids.len();
    }
    let seal = started.elapsed().as_secs_f64();
    index.wait_until_merge_idle();
    let total = started.elapsed().as_secs_f64();
    if let Some(err) = index.merge_error() {
        return Err(other(format!("merger failed: {err}")));
    }
    Ok(IngestRow {
        headline: auto_merge && !prefix,
        auto_merge,
        contextual_prefix: prefix,
        prefix_llm: prefix.then(|| model.chat_model.clone()),
        files: files.len(),
        chunks,
        docs_per_sec: files.len() as f64 / total,
        chunks_per_sec: chunks as f64 / total,
        seal_wall_seconds: seal,
        merge_wall_seconds: total - seal,
        segment_count: index.segment_chunk_counts().len(),
    })
}

fn bench_ingest(files: usize, chunks_per_file: usize, model: &ModelArgs) -> Result<Value, Error> {
    let fixture = tempfile::tempdir()?;
    let mut rng = Rng::new(3);
    let mut paths = Vec::with_capacity(files);
    for i in 0..files {
        let path = fixture.path().join(format!("file{i:02}.txt"));
        std::fs::write(&path, ingest_file(&mut rng, chunks_per_file))?;
        paths.push(path);
    }
    let mut rows = vec![ingest_row(&paths, true, false, model)?, ingest_row(&paths, false, false, model)?];
    if model.chat_endpoint.is_some() {
        rows.push(ingest_row(&paths, true, true, model)?);
    }
    let report = IngestReport { environment: BenchEnvironment::detect(&model.config()), rows };
    Ok(serde_json::to_value(report)?)
}

fn bench_ram(vectors: usize, model: &ModelArgs) -> Result<Value, Error> {
    let dir = tempfile::tempdir()?;
    let root = dir.path().join("collection");
    {
        let mut index = gyrfalcon::Index::create(&root, &model.config())?;
        index.auto_merge = false;
        let mut left = vectors;
        while left > 0 {
            let n = left.min(gyrfalcon::SEAL_EVERY);
            index.seal_n_chunks(n, "ram")?;
            left -= n;
        }
    }
    // RSS is measured in a fresh process, so memory the build used does not count.
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    args.push("--measure-collection".into());
    args.push(root.to_string_lossy().into_owned());
    let output = std::process::Command::new(std::env::current_exe()?).args(&args).output()?;
    if !output.status.success() {
        return Err(other(format!("RSS measurement failed: {}", String::from_utf8_lossy(&output.stderr))));
    }
    let measured: Value = serde_json::from_slice(&output.stdout)?;
    let config = model.config();
    let mut report = collection_bytes(&root);
    for key in ["rss_baseline", "rss_index_open", "rss_model_loaded", "rss_model_dropped"] {
        report.insert(key.into(), measured[key].clone());
    }
    report.insert("vectors".into(), json!(vectors));
    report.insert("tombstone_count".into(), json!(0));
    report.insert("vector_payload_planning_bytes".into(), json!(vectors as u64 * config.dimension as u64));
    report.insert("environment".into(), serde_json::to_value(BenchEnvironment::detect(&config))?);
    Ok(Value::Object(report))
}

/// Child side of `gyr bench ram`: open the collection without a model, retrieve, then load and
/// drop the model, reading RSS after each step.
fn measure_ram(root: &Path, model: &ModelArgs) -> Result<Value, Error> {
    let baseline = rss_bytes();
    let config = model.config();
    let mut index = Index::open(root, &config)?;
    index.set_embedder(Box::new(HashEmbedder::new(config.dimension)));
    index.retrieve("ram", RetrieveOptions { rerank: false, limit: 10 })?;
    let index_open = rss_bytes();
    index.set_embedder(model.load_embedder(&config)?);
    index.retrieve("ram", RetrieveOptions { rerank: false, limit: 10 })?;
    let loaded = rss_bytes();
    index.set_embedder(Box::new(HashEmbedder::new(config.dimension)));
    let dropped = rss_bytes();
    Ok(json!({
        "rss_baseline": baseline,
        "rss_index_open": index_open,
        "rss_model_loaded": loaded,
        "rss_model_dropped": dropped,
    }))
}

fn read_qrels(path: &Path) -> Result<HashMap<String, gyrfalcon::bench::Qrels>, Error> {
    let mut qrels: HashMap<String, gyrfalcon::bench::Qrels> = HashMap::new();
    for line in std::fs::read_to_string(path)?.lines().skip(1) {
        let mut fields = line.split('\t');
        if let (Some(query), Some(doc), Some(score)) = (fields.next(), fields.next(), fields.next()) {
            qrels.entry(query.to_string()).or_default().insert(doc.to_string(), score.trim().parse().unwrap_or(0));
        }
    }
    Ok(qrels)
}

/// One dataset to trace: its documents and each query with its split and judgments.
struct TraceSet {
    name: String,
    docs: Vec<Document>,
    queries: Vec<(String, String, String, gyrfalcon::bench::Qrels)>,
}

/// The protocol's splits: SciFact validates on `train`, NFCorpus and FiQA on `dev`, others
/// split `test` by query-id hash.
fn beir_trace_set(dataset: &Path) -> Result<TraceSet, Error> {
    let name = dataset.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let validation_file = match name.as_str() {
        "scifact" => Some("train"),
        "nfcorpus" | "fiqa" => Some("dev"),
        _ => None,
    };
    let test = read_qrels(&dataset.join("qrels/test.tsv"))?;
    let validation = match validation_file {
        Some(file) => read_qrels(&dataset.join(format!("qrels/{file}.tsv")))?,
        None => HashMap::new(),
    };
    let texts: HashMap<String, String> = jsonl(&dataset.join("queries.jsonl"))?
        .iter()
        .filter_map(|q| Some((q["_id"].as_str()?.to_string(), q["text"].as_str()?.to_string())))
        .collect();
    let mut queries = Vec::new();
    for (split, judged) in [("test", &test), ("validation", &validation)] {
        for (id, qrels) in judged {
            let Some(text) = texts.get(id) else { continue };
            let split = if validation_file.is_none() { gyrfalcon::planner::hash_split(id) } else { split };
            queries.push((split.to_string(), id.clone(), text.clone(), qrels.clone()));
        }
    }
    queries.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
    let docs = jsonl(&dataset.join("corpus.jsonl"))?
        .iter()
        .map(|doc| {
            let title = doc["title"].as_str().unwrap_or_default().trim();
            let text = doc["text"].as_str().unwrap_or_default();
            let body = if title.is_empty() { text.to_string() } else { format!("{title}\n\n{text}") };
            Document::body(doc["_id"].as_str().unwrap_or_default(), body)
        })
        .collect();
    Ok(TraceSet { name, docs, queries })
}

fn orb_trace_set(fixture: &Path) -> Result<TraceSet, Error> {
    let docs = jsonl(&fixture.join("chunks.jsonl"))?
        .iter()
        .map(|c| Document::body(c["doc_key"].as_str().unwrap_or_default(), c["text"].as_str().unwrap_or_default()))
        .collect();
    let mut queries: Vec<_> = jsonl(&fixture.join("questions.jsonl"))?
        .iter()
        .map(|q| {
            let id = q["query_id"].as_str().unwrap_or_default().to_string();
            let qrels = [(q["relevant"].as_str().unwrap_or_default().to_string(), 1)].into_iter().collect();
            (
                gyrfalcon::planner::hash_split(&id).to_string(),
                id,
                q["query"].as_str().unwrap_or_default().to_string(),
                qrels,
            )
        })
        .collect();
    queries.sort_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
    Ok(TraceSet { name: "orb".into(), docs, queries })
}

fn bench_trace(
    datasets: &[PathBuf],
    orb_fixture: Option<&Path>,
    out_dir: &Path,
    depth: usize,
    reranker: Option<(&dyn gyrfalcon::Reranker, usize)>,
    model: &ModelArgs,
) -> Result<Value, Error> {
    let mut sets = Vec::new();
    for dataset in datasets {
        sets.push(beir_trace_set(dataset)?);
    }
    if let Some(fixture) = orb_fixture {
        sets.push(orb_trace_set(fixture)?);
    }
    let label = model.model_id().replace('/', "_");
    let dir = out_dir.join(&label);
    std::fs::create_dir_all(&dir)?;
    let mut written = Vec::new();
    for set in sets {
        let collection = tempfile::tempdir()?;
        let mut index = model.open(&collection.path().join("collection"), true)?;
        let started = Instant::now();
        for batch in set.docs.chunks(10_000) {
            index.index_sections(batch)?;
        }
        index.wait_until_merge_idle();
        eprintln!("{}: indexed {} docs in {:.0}s", set.name, set.docs.len(), started.elapsed().as_secs_f64());
        // Warm the page cache and stored-field caches, so pass 1 and the live check that
        // follows it both measure a warm index.
        for (_, _, text, _) in &set.queries {
            index.retrieve(text, RetrieveOptions { rerank: false, limit: 100 })?;
        }
        // Pass 1: scores and step timings, with nothing else running between queries.
        let mut traces = Vec::with_capacity(set.queries.len());
        for (split, id, text, qrels) in &set.queries {
            let mut trace = index.trace_query(text, depth, None)?;
            trace.dataset = set.name.clone();
            trace.split = split.clone();
            trace.query_id = id.clone();
            trace.qrels = qrels.clone();
            traces.push(trace);
        }
        calibrate_overhead(&mut index, &set.queries, &mut traces)?;
        let test_traces: Vec<(String, gyrfalcon::planner::QueryTrace)> = set
            .queries
            .iter()
            .zip(&traces)
            .filter(|(_, t)| t.split == "test")
            .map(|((_, _, text, _), t)| (text.clone(), t.clone()))
            .collect();
        let verification = verify_replay(&mut index, &test_traces)?;
        // Pass 2: cross-encoder scores.
        if let Some((reranker, k)) = reranker {
            for (done, ((_, _, text, _), trace)) in set.queries.iter().zip(traces.iter_mut()).enumerate() {
                index.trace_rerank(trace, text, reranker, k)?;
                if (done + 1) % 200 == 0 {
                    eprintln!("{}: reranked {}/{} queries", set.name, done + 1, set.queries.len());
                }
            }
        }
        let path = dir.join(format!("{}.jsonl", set.name));
        let mut file = std::io::BufWriter::new(std::fs::File::create(&path)?);
        for trace in &traces {
            serde_json::to_writer(&mut file, trace)?;
            file.write_all(b"\n")?;
        }
        file.flush()?;
        written.push(json!({
            "dataset": set.name,
            "queries": set.queries.len(),
            "path": path,
            "replay_vs_live": verification,
        }));
    }
    Ok(json!({ "model_id": model.model_id(), "traces": written }))
}

/// Sets every trace's fixed per-call overhead to the median gap between a live retrieve and
/// its replayed steps, measured on validation queries only.
fn calibrate_overhead(
    index: &mut Index,
    queries: &[(String, String, String, gyrfalcon::bench::Qrels)],
    traces: &mut [gyrfalcon::planner::QueryTrace],
) -> Result<(), Error> {
    use gyrfalcon::planner::{replay, Policy};
    let configured = (index.rrf_weights, index.fusion);
    index.rrf_weights = [1.0, 1.0];
    index.fusion = Fusion::Rescore { dense_weight: 0.2 };
    let mut gaps = Vec::new();
    for ((split, _, text, _), trace) in queries.iter().zip(traces.iter()) {
        if split != "validation" {
            continue;
        }
        let started = Instant::now();
        index.retrieve(text, RetrieveOptions { rerank: false, limit: 100 })?;
        let live = started.elapsed().as_secs_f64() * 1000.0;
        let (_, steps) = replay(trace, Policy::Rescore { dense_weight: 0.2, rerank_k: 0 }, 100);
        gaps.push(live - steps);
    }
    (index.rrf_weights, index.fusion) = configured;
    gaps.sort_by(|a, b| a.total_cmp(b));
    let overhead = gyrfalcon::bench::percentile(&gaps, 50.0).max(0.0);
    eprintln!("  calibrated overhead {overhead:.3} ms from {} validation queries", gaps.len());
    for trace in traces.iter_mut() {
        trace.timings.overhead = overhead;
    }
    Ok(())
}

/// The protocol's replay check: on the traced index, each policy's live test nDCG@10 and
/// p50 latency against its replay. Replay is trusted within 1% nDCG and 10% p50.
fn verify_replay(index: &mut Index, traces: &[(String, gyrfalcon::planner::QueryTrace)]) -> Result<Value, Error> {
    use gyrfalcon::planner::{score, summarize, Policy};
    let configured = (index.rrf_weights, index.fusion);
    let mut rows = Vec::new();
    for (spec, weights, fusion) in [
        ("lexical", [1.0, 0.0], Fusion::Rrf),
        ("dense", [0.0, 1.0], Fusion::Rrf),
        ("rrf", [1.0, 1.0], Fusion::Rrf),
        ("rescore:0.2", [1.0, 1.0], Fusion::Rescore { dense_weight: 0.2 }),
        ("rescore:0.5", [1.0, 1.0], Fusion::Rescore { dense_weight: 0.5 }),
    ] {
        index.rrf_weights = weights;
        index.fusion = fusion;
        let (mut ndcg, mut millis) = (0.0, Vec::with_capacity(traces.len()));
        for (text, trace) in traces {
            let started = Instant::now();
            let hits = index.retrieve(text, RetrieveOptions { rerank: false, limit: 100 })?;
            millis.push(started.elapsed().as_secs_f64() * 1000.0);
            let ranked: Vec<String> =
                gyrfalcon::bench::dedupe_doc_keys(&hits.hits.iter().map(|h| h.doc_id.clone()).collect::<Vec<_>>())
                    .into_iter()
                    .filter(|doc| *doc != trace.query_id)
                    .collect();
            ndcg += gyrfalcon::bench::ndcg_at(&ranked, &trace.qrels, 10);
        }
        let live_ndcg = ndcg / traces.len().max(1) as f64;
        let live = LatencyColumn::from_samples(1, millis, 0.0, &[]);
        let policy = Policy::parse(spec).unwrap();
        let replayed = summarize(&traces.iter().map(|(_, t)| score(t, policy)).collect::<Vec<_>>());
        let ndcg_ok = (replayed.ndcg_at_10 - live_ndcg).abs() <= 0.01 * live_ndcg.max(1e-9);
        // Live lexical-only and dense-only still run both legs, so only fused policies compare.
        let p50_ok = (replayed.p50_ms - live.p50_ms).abs() <= 0.10 * live.p50_ms.max(1e-9) || !spec.starts_with('r');
        eprintln!(
            "  {spec}: live nDCG {live_ndcg:.4} replay {:.4}; live p50 {:.3} ms replay {:.3} ms",
            replayed.ndcg_at_10, live.p50_ms, replayed.p50_ms
        );
        rows.push(json!({
            "policy": spec,
            "live_ndcg_at_10": live_ndcg,
            "replay_ndcg_at_10": replayed.ndcg_at_10,
            "live_p50_ms": live.p50_ms,
            "replay_p50_ms": replayed.p50_ms,
            "ndcg_within_1pct": ndcg_ok,
            "p50_within_10pct": p50_ok,
        }));
    }
    (index.rrf_weights, index.fusion) = configured;
    Ok(Value::Array(rows))
}

fn bench_replay(
    traces: &[PathBuf],
    split: &str,
    policies: &[String],
    reference_spec: Option<&str>,
) -> Result<Value, Error> {
    use gyrfalcon::planner::{paired_bootstrap, score, summarize, Policy, QueryTrace};
    let parse = |spec: &str| Policy::parse(spec).ok_or_else(|| other(format!("unknown policy {spec}")));
    let reference = reference_spec.map(parse).transpose()?;
    let mut datasets = Vec::new();
    for path in traces {
        let mut by_dataset: HashMap<String, Vec<QueryTrace>> = HashMap::new();
        for line in std::fs::read_to_string(path)?.lines().filter(|l| !l.trim().is_empty()) {
            let trace: QueryTrace = serde_json::from_str(line)?;
            if trace.split == split {
                by_dataset.entry(trace.dataset.clone()).or_default().push(trace);
            }
        }
        let mut names: Vec<_> = by_dataset.keys().cloned().collect();
        names.sort();
        for name in names {
            let set = &by_dataset[&name];
            let reference_scores: Option<Vec<f64>> =
                reference.map(|p| set.iter().map(|t| score(t, p).ndcg_at_10).collect());
            let mut rows = Vec::new();
            for spec in policies {
                let policy = parse(spec)?;
                let scores: Vec<_> = set.iter().map(|t| score(t, policy)).collect();
                let mut row = serde_json::to_value(summarize(&scores))?;
                row["policy"] = json!(spec);
                if let Some(base) = &reference_scores {
                    let ndcg: Vec<f64> = scores.iter().map(|s| s.ndcg_at_10).collect();
                    let (mean, low, high) = paired_bootstrap(&ndcg, base, 10_000, 1);
                    row["ndcg_vs_reference"] =
                        json!({ "mean": mean, "low": low, "high": high, "significant": low > 0.0 || high < 0.0 });
                }
                rows.push(row);
            }
            datasets.push(json!({
                "dataset": name,
                "split": split,
                "queries": set.len(),
                "reference": reference_spec,
                "policies": rows,
            }));
        }
    }
    Ok(json!({ "datasets": datasets }))
}

/// `[lexical, dense]` RRF weights and fusion for one `--fusions` entry.
fn parse_fusion(spec: &str, configured: Fusion) -> Result<([f32; 2], Fusion), Error> {
    let weight = |text: &str| text.parse::<f32>().map_err(|_| other(format!("bad fusion weight in {spec}")));
    Ok(match spec.split_once(':') {
        None if spec == "configured" => ([1.0, 1.0], configured),
        None if spec == "rrf" => ([1.0, 1.0], Fusion::Rrf),
        None if spec == "lexical" => ([1.0, 0.0], Fusion::Rrf),
        None if spec == "dense" => ([0.0, 1.0], Fusion::Rrf),
        Some(("score", w)) => ([1.0, 1.0], Fusion::Score { dense_weight: weight(w)? }),
        Some(("rescore", w)) => ([1.0, 1.0], Fusion::Rescore { dense_weight: weight(w)? }),
        _ => return Err(other(format!("unknown fusion {spec}"))),
    })
}

/// Where `bench_beir` keeps its report (after each dataset) and its ranked lists.
struct BeirOutputs<'a> {
    report: Option<&'a Path>,
    runs: Option<&'a Path>,
}

fn bench_beir(
    datasets: &[PathBuf],
    split: &str,
    fusions: &[String],
    depth: usize,
    rerank: bool,
    model: &ModelArgs,
    outputs: BeirOutputs,
) -> Result<Value, Error> {
    let mut results = Vec::new();
    let mut environment = None;
    for dataset in datasets {
        let name = dataset.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let mut qrels: HashMap<String, gyrfalcon::bench::Qrels> = HashMap::new();
        for line in std::fs::read_to_string(dataset.join("qrels").join(format!("{split}.tsv")))?.lines().skip(1) {
            let mut fields = line.split('\t');
            if let (Some(query), Some(doc), Some(score)) = (fields.next(), fields.next(), fields.next()) {
                let score: i64 = score.trim().parse().unwrap_or(0);
                qrels.entry(query.to_string()).or_default().insert(doc.to_string(), score);
            }
        }
        let docs: Vec<Document> = jsonl(&dataset.join("corpus.jsonl"))?
            .iter()
            .map(|doc| {
                let title = doc["title"].as_str().unwrap_or_default().trim();
                let text = doc["text"].as_str().unwrap_or_default();
                let body = if title.is_empty() { text.to_string() } else { format!("{title}\n\n{text}") };
                Document::body(doc["_id"].as_str().unwrap_or_default(), body)
            })
            .collect();
        let mut queries: Vec<(String, String)> = jsonl(&dataset.join("queries.jsonl"))?
            .iter()
            .filter_map(|q| Some((q["_id"].as_str()?.to_string(), q["text"].as_str()?.to_string())))
            .filter(|(id, _)| qrels.contains_key(id))
            .collect();
        queries.sort();

        let dir = tempfile::tempdir()?;
        let mut index = model.open(&dir.path().join("collection"), true)?;
        let started = Instant::now();
        let mut done = 0;
        for batch in docs.chunks(10_000) {
            index.index_sections(batch)?;
            done += batch.len();
            eprintln!("{name}: indexed {done}/{} docs, {:.0}s", docs.len(), started.elapsed().as_secs_f64());
        }
        index.wait_until_merge_idle();
        let index_seconds = started.elapsed().as_secs_f64();
        environment.get_or_insert_with(|| BenchEnvironment::detect(index.config()));
        let configured = index.fusion;
        let mut rows = Vec::new();
        for spec in fusions {
            let (weights, fusion) = parse_fusion(spec, configured)?;
            index.rrf_weights = weights;
            index.fusion = fusion;
            let (mut ndcg, mut recall, mut mrr) = (0.0, 0.0, 0.0);
            let mut millis = Vec::with_capacity(queries.len());
            let mut run = String::new();
            for (id, text) in &queries {
                let started = Instant::now();
                let hits = index.retrieve(text, RetrieveOptions { rerank, limit: depth })?;
                millis.push(started.elapsed().as_secs_f64() * 1000.0);
                // BEIR and MTEB ignore a retrieved document whose id is the query id (ArguAna).
                let ranked: Vec<String> =
                    gyrfalcon::bench::dedupe_doc_keys(&hits.hits.iter().map(|h| h.doc_id.clone()).collect::<Vec<_>>())
                        .into_iter()
                        .filter(|doc| doc != id)
                        .collect();
                for (rank, doc) in ranked.iter().enumerate() {
                    run.push_str(&format!("{id}\t{doc}\t{}\t{}\n", rank + 1, ranked.len() - rank));
                }
                let judged = &qrels[id];
                ndcg += gyrfalcon::bench::ndcg_at(&ranked, judged, 10);
                recall += gyrfalcon::bench::recall_at(&ranked, judged, 100);
                mrr += gyrfalcon::bench::mrr_at(&ranked, judged, 10);
            }
            if let Some(run_dir) = outputs.runs {
                std::fs::create_dir_all(run_dir)?;
                std::fs::write(run_dir.join(format!("{name}-{}.tsv", spec.replace(':', "_"))), run)?;
            }
            let n = queries.len().max(1) as f64;
            let latency = LatencyColumn::from_samples(1, millis, 0.0, &[]);
            rows.push(json!({
                "fusion": spec,
                "ndcg_at_10": ndcg / n,
                "recall_at_100": recall / n,
                "mrr_at_10": mrr / n,
                "query_p50_ms": latency.p50_ms,
                "query_p99_ms": latency.p99_ms,
            }));
            eprintln!("{name} {spec}: nDCG@10 {:.4}", ndcg / n);
        }
        results.push(json!({
            "dataset": name,
            "split": split,
            "docs": docs.len(),
            "queries": queries.len(),
            "index_seconds": index_seconds,
            "docs_per_sec": docs.len() as f64 / index_seconds,
            "rerank": rerank,
            "results": rows,
        }));
        // Keep finished datasets on disk if a later one is interrupted.
        if let Some(path) = outputs.report {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let partial = json!({ "environment": environment, "datasets": results });
            std::fs::write(path, serde_json::to_vec_pretty(&partial)?)?;
        }
    }
    Ok(json!({ "environment": environment, "datasets": results }))
}

fn bench_embed(fixture: &Path, chunks: usize, batch: usize, queries: usize, model: &ModelArgs) -> Result<Value, Error> {
    let dir = tempfile::tempdir()?;
    let index = model.open(&dir.path().join("collection"), true)?;
    let texts: Vec<String> = jsonl(&fixture.join("chunks.jsonl"))?
        .iter()
        .filter_map(|c| c["text"].as_str().map(|t| gyrfalcon::prepare(t).embed_text))
        .filter(|t| !t.is_empty())
        .take(chunks)
        .collect();
    let questions: Vec<String> = jsonl(&fixture.join("questions.jsonl"))?
        .iter()
        .filter_map(|q| q["query"].as_str().map(str::to_string))
        .take(queries)
        .collect();
    let embedder = index.embedder();
    let started = Instant::now();
    for part in texts.chunks(batch.max(1)) {
        embedder.embed(part)?;
    }
    let seconds = started.elapsed().as_secs_f64();
    let mut millis = Vec::with_capacity(questions.len());
    for question in &questions {
        let started = Instant::now();
        embedder.embed(std::slice::from_ref(question))?;
        millis.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    let column = LatencyColumn::from_samples(1, millis, 0.0, &[]);
    Ok(json!({
        "environment": BenchEnvironment::detect(index.config()),
        "chunks": texts.len(),
        "batch": batch,
        "chunks_per_sec": texts.len() as f64 / seconds,
        "mean_chunk_bytes": texts.iter().map(String::len).sum::<usize>() as f64 / texts.len().max(1) as f64,
        "query_p50_ms": column.p50_ms,
        "query_p99_ms": column.p99_ms,
        "queries": questions.len(),
    }))
}

fn read_json(path: &Path) -> Result<Value, Error> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

/// Writes `chunks.jsonl`, `questions.jsonl`, and `meta.json` for the text-only slice.
fn prepare_accuracy(dataset: &Path, fixture: &Path, revision: &str, expected: usize) -> Result<Value, Error> {
    let queries = read_json(&dataset.join("queries.json"))?;
    let answers = read_json(&dataset.join("answers.json"))?;
    let qrels = read_json(&dataset.join("qrels.json"))?;
    let queries = queries.as_object().ok_or_else(|| other("queries.json is not an object"))?;
    let mut text_ids: Vec<String> =
        queries.iter().filter(|(_, q)| q["source"].as_str() == Some("text")).map(|(id, _)| id.clone()).collect();
    text_ids.sort();
    if text_ids.len() != expected {
        return Err(other(format!("expected {expected} text-only questions, found {}", text_ids.len())));
    }
    let mut papers: Vec<PathBuf> = std::fs::read_dir(dataset.join("corpus"))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    papers.sort();
    let mut sections_per_paper = HashMap::new();
    let mut chunks = String::new();
    let mut chunk_count = 0;
    for path in &papers {
        let paper_id = path.file_stem().unwrap().to_string_lossy().to_string();
        let paper = read_json(path)?;
        let mut push = |key: String, text: &str| {
            chunks.push_str(&json!({ "doc_key": key, "text": text }).to_string());
            chunks.push('\n');
            chunk_count += 1;
        };
        push(format!("{paper_id}#title"), paper["title"].as_str().unwrap_or_default());
        push(format!("{paper_id}#abstract"), paper["abstract"].as_str().unwrap_or_default());
        let sections = paper["sections"].as_array().cloned().unwrap_or_default();
        for (i, section) in sections.iter().enumerate() {
            push(format!("{paper_id}#{i}"), section["text"].as_str().unwrap_or_default());
        }
        sections_per_paper.insert(paper_id, sections.len());
    }
    let mut questions = String::new();
    for id in &text_ids {
        let qrel = qrels.get(id).and_then(Value::as_object).ok_or_else(|| other(format!("no qrel for {id}")))?;
        if qrel.len() != 2 || !qrel.contains_key("doc_id") || !qrel.contains_key("section_id") {
            return Err(other(format!("qrel for {id} is not {{doc_id, section_id}}")));
        }
        let doc_id = qrel["doc_id"].as_str().map(str::to_string).unwrap_or_else(|| qrel["doc_id"].to_string());
        let section =
            qrel["section_id"].as_u64().ok_or_else(|| other(format!("section_id for {id} is not an index")))?;
        let count = sections_per_paper
            .get(&doc_id)
            .copied()
            .ok_or_else(|| other(format!("qrel {id} names unknown paper {doc_id}")))?;
        if section as usize >= count {
            return Err(other(format!("qrel {id} points at section {section} of {count} in {doc_id}")));
        }
        questions.push_str(
            &json!({
                "query_id": id,
                "query": queries[id]["query"],
                "answer": answers.get(id).cloned().unwrap_or(Value::Null),
                "relevant": format!("{doc_id}#{section}"),
            })
            .to_string(),
        );
        questions.push('\n');
    }
    std::fs::create_dir_all(fixture)?;
    std::fs::write(fixture.join("chunks.jsonl"), chunks)?;
    std::fs::write(fixture.join("questions.jsonl"), questions)?;
    let meta = json!({
        "dataset_revision": revision,
        "question_id_sha256": question_id_sha256(&text_ids),
        "questions": text_ids.len(),
        "chunks": chunk_count,
    });
    std::fs::write(fixture.join("meta.json"), serde_json::to_vec_pretty(&meta)?)?;
    Ok(meta)
}

fn jsonl(path: &Path) -> Result<Vec<Value>, Error> {
    let text = std::fs::read_to_string(path)?;
    text.lines().filter(|l| !l.trim().is_empty()).map(|l| Ok(serde_json::from_str(l)?)).collect()
}

fn run_accuracy(
    fixture: &Path,
    system: &str,
    rerank: bool,
    judge: Option<ChatConfig>,
    model: &ModelArgs,
) -> Result<AccuracyReport, Error> {
    let meta = read_json(&fixture.join("meta.json"))?;
    let chunks = jsonl(&fixture.join("chunks.jsonl"))?;
    let questions = jsonl(&fixture.join("questions.jsonl"))?;
    let dir = tempfile::tempdir()?;
    let mut index = model.open(&dir.path().join("collection"), true)?;
    let mut bodies = HashMap::new();
    let docs: Vec<Document> = chunks
        .iter()
        .map(|chunk| {
            let key = chunk["doc_key"].as_str().unwrap_or_default().to_string();
            let text = chunk["text"].as_str().unwrap_or_default().to_string();
            bodies.insert(key.clone(), text.clone());
            Document::body(key, text)
        })
        .collect();
    let mut chunk_count = 0;
    for batch in docs.chunks(10_000) {
        chunk_count += index.index_sections(batch)?.chunk_ids.len();
    }
    index.wait_until_merge_idle();
    eprintln!("chunks {chunk_count}");
    let chat = model.chat_endpoint.as_ref().map(|endpoint| ChatConfig {
        endpoint: endpoint.clone(),
        model: model.chat_model.clone(),
        api_key: std::env::var("GYR_CHAT_API_KEY").ok(),
    });
    let mut report = AccuracyReport {
        system: system.to_string(),
        generator_id: chat.as_ref().map(|c| c.model.clone()).unwrap_or_else(|| "none".into()),
        answer_prompt_sha256: sha256_hex(ANSWER_PROMPT.as_bytes()),
        judge_id: judge.as_ref().map(|c| c.model.clone()).unwrap_or_else(|| "none".into()),
        question_id_sha256: meta["question_id_sha256"].as_str().unwrap_or_default().to_string(),
        dataset_revision: meta["dataset_revision"].as_str().unwrap_or_default().to_string(),
        chunk_count,
        ..AccuracyReport::default()
    };
    for question in &questions {
        let query = question["query"].as_str().unwrap_or_default();
        let hits = index.retrieve(query, RetrieveOptions { rerank, limit: 10 })?;
        let doc_keys: Vec<String> = hits.hits.iter().map(|hit| hit.doc_id.clone()).collect();
        let mut result = QuestionResult {
            query_id: question["query_id"].as_str().unwrap_or_default().to_string(),
            relevant: question["relevant"].as_str().unwrap_or_default().to_string(),
            doc_keys: doc_keys.clone(),
            ..QuestionResult::default()
        };
        if let Some(chat) = &chat {
            let context: Vec<Hit> = gyrfalcon::bench::dedupe_doc_keys(&doc_keys)
                .into_iter()
                .take(5)
                .map(|key| Hit {
                    chunk_id: 0,
                    body: bodies.get(&key).cloned().unwrap_or_default(),
                    doc_id: key,
                    prefix: None,
                    byte_start: 0,
                    byte_end: 0,
                    score: 0.0,
                })
                .collect();
            let answer = complete(chat, &fill_answer_prompt(query, &context)).unwrap_or_default();
            if let Some(judge) = &judge {
                let gold =
                    question["answer"].as_str().map(str::to_string).unwrap_or_else(|| question["answer"].to_string());
                match complete(judge, &fill_judge_prompt(query, &gold, &answer)) {
                    Ok(verdict) => match verdict.trim().to_lowercase().trim_matches(|c: char| !c.is_alphabetic()) {
                        "pass" => {
                            result.pass = Some(true);
                            report.passes += 1;
                        }
                        "fail" => result.pass = Some(false),
                        _ => report.judge_errors += 1,
                    },
                    Err(_) => report.judge_errors += 1,
                }
            }
            result.answer = Some(answer);
        }
        report.questions.push(result);
    }
    Ok(report)
}

pub fn run(command: BenchCommand) -> Result<(), Error> {
    match command {
        BenchCommand::Retrieve {
            segments,
            docs,
            corpus,
            queries,
            warmup,
            concurrency,
            rerank,
            time_to_request,
            out,
            model,
        } => {
            let report = bench_retrieve(
                segments,
                docs,
                corpus.as_deref(),
                queries,
                warmup,
                &concurrency,
                rerank,
                time_to_request,
                &model,
            )?;
            emit(&report, out.as_deref())
        }
        BenchCommand::Ingest { files, chunks_per_file, out, model } => {
            emit(&bench_ingest(files, chunks_per_file, &model)?, out.as_deref())
        }
        BenchCommand::Beir { dataset, split, fusions, depth, rerank, out, run_dir, model } => {
            let outputs = BeirOutputs { report: out.as_deref(), runs: run_dir.as_deref() };
            let report = bench_beir(&dataset, &split, &fusions, depth, rerank, &model, outputs)?;
            emit(&report, out.as_deref())
        }
        BenchCommand::Trace { dataset, orb_fixture, out_dir, depth, rerank_dir, rerank_k, rerank_tokens, model } => {
            let reranker = match &rerank_dir {
                Some(dir) => Some(gyrfalcon::OrtReranker::from_dir_with_max_tokens(dir, rerank_tokens)?),
                None => None,
            };
            let reranker = reranker.as_ref().map(|r| (r as &dyn gyrfalcon::Reranker, rerank_k));
            emit(&bench_trace(&dataset, orb_fixture.as_deref(), &out_dir, depth, reranker, &model)?, None)
        }
        BenchCommand::Replay { traces, split, policies, reference, out } => {
            emit(&bench_replay(&traces, &split, &policies, reference.as_deref())?, out.as_deref())
        }
        BenchCommand::Embed { fixture, chunks, batch, queries, out, model } => {
            emit(&bench_embed(&fixture, chunks, batch, queries, &model)?, out.as_deref())
        }
        BenchCommand::Ram { measure_collection: Some(root), model, .. } => {
            println!("{}", measure_ram(&root, &model)?);
            Ok(())
        }
        BenchCommand::Ram { vectors, out, model, .. } => emit(&bench_ram(vectors, &model)?, out.as_deref()),
        BenchCommand::Accuracy {
            prepare,
            compare,
            dataset,
            dataset_revision,
            fixture,
            expected_questions,
            system,
            rerank,
            judge_endpoint,
            judge_model,
            out,
            model,
        } => {
            if prepare {
                let dataset = dataset.ok_or_else(|| other("--prepare needs --dataset"))?;
                let meta = prepare_accuracy(&dataset, &fixture, &dataset_revision, expected_questions)?;
                return emit(&meta, None);
            }
            if !compare.is_empty() {
                let reports = compare
                    .iter()
                    .map(|path| Ok(serde_json::from_slice::<AccuracyReport>(&std::fs::read(path)?)?))
                    .collect::<Result<Vec<_>, Error>>()?;
                let compared = compare_accuracy(&reports)?;
                return emit(&compared.json, out.as_deref());
            }
            let judge = judge_endpoint.map(|endpoint| ChatConfig {
                endpoint,
                model: judge_model,
                api_key: std::env::var("GYR_JUDGE_API_KEY").ok(),
            });
            let report = run_accuracy(&fixture, &system, rerank, judge, &model)?;
            let out = out.unwrap_or_else(|| PathBuf::from(format!("bench/reports/{system}.json")));
            let mut value = serde_json::to_value(&report)?;
            value["retrieval"] = serde_json::to_value(score_report(&report))?;
            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(&out, serde_json::to_vec_pretty(&value)?)?;
            std::io::stdout().write_all(serde_json::to_string_pretty(&value["retrieval"])?.as_bytes())?;
            println!();
            Ok(())
        }
    }
}
