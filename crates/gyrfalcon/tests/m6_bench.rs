mod common;

use std::path::Path;
use std::process::Command;

use common::{example_binary, gyr};
use gyrfalcon::bench::{
    bench_retrieve_scripted, compare_accuracy, dedupe_doc_keys, retrieval_metrics, AccuracyReport, QuestionResult,
};
use gyrfalcon::FallbackStep;

#[test]
fn fallback_percentages_sum_to_100() {
    let report =
        bench_retrieve_scripted(&[FallbackStep::E64, FallbackStep::E128, FallbackStep::Exact, FallbackStep::E64]);
    let sum = report.fallback_pct.values().sum::<f64>();
    assert!((sum - 100.0).abs() < 1e-6);
    assert!((report.fallback_pct["64"] - 50.0).abs() < 1e-6);
    assert!((report.fallback_pct["exact"] - 25.0).abs() < 1e-6);
    assert_eq!(report.fallback_pct.len(), 6);
}

fn assert_fallback_keys(column: &serde_json::Value) {
    let pct = column["fallback_pct"].as_object().unwrap();
    for key in ["64", "128", "256", "512", "1024", "exact"] {
        assert!(pct.contains_key(key), "{key}");
    }
    let sum: f64 = pct.values().map(|v| v.as_f64().unwrap()).sum();
    assert!((sum - 100.0).abs() < 1e-6, "{sum}");
}

#[test]
fn bench_retrieve_reports_both_segment_layouts() {
    for (segments, expected) in [("1", 1), ("50", 20)] {
        let out = gyr(&[
            "bench",
            "retrieve",
            "--segments",
            segments,
            "--docs",
            "20",
            "--queries",
            "40",
            "--warmup",
            "5",
            "--concurrency",
            "4",
            "--embedder",
            "hash",
            "--time-to-request",
        ]);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let report: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        assert_eq!(report["segment_count"], expected);
        assert_eq!(report["timed"], 40);
        assert_fallback_keys(&report);
        assert_fallback_keys(&report["single_flight"]);
        assert_fallback_keys(&report["concurrent"][0]);
        assert_eq!(report["concurrent"][0]["in_flight"], 4);
        assert!(
            report["single_flight"]["p99_ms"].as_f64().unwrap() >= report["single_flight"]["p50_ms"].as_f64().unwrap()
        );
        assert_eq!(report["time_to_request"]["llm"], "discarded");
        assert_eq!(report["environment"]["model_id"], "gyr-hash");
        assert_eq!(report["environment"]["quantization"], "i8");
        assert!(report["environment"]["git_revision"].is_string());
    }
}

#[test]
fn bench_ingest_waits_for_the_merger() {
    let out = gyr(&["bench", "ingest", "--files", "8", "--chunks-per-file", "10", "--embedder", "hash"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let report: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    let rows = report["rows"].as_array().unwrap();
    let headline = rows.iter().find(|row| row["headline"] == true).unwrap();
    assert_eq!(headline["auto_merge"], true);
    assert_eq!(headline["chunks"], 80);
    assert_eq!(headline["files"], 8);
    assert!(headline["merge_wall_seconds"].as_f64().unwrap() > 0.0);
    assert_eq!(headline["segment_count"], 1);
    let seal_only = rows.iter().find(|row| row["auto_merge"] == false).unwrap();
    assert_eq!(seal_only["headline"], false);
    assert_eq!(seal_only["segment_count"], 8);
    for key in ["docs_per_sec", "chunks_per_sec", "seal_wall_seconds", "merge_wall_seconds"] {
        assert!(headline[key].is_number(), "{key}");
    }
}

#[test]
fn ram_report_lists_the_byte_counts() {
    let report = gyrfalcon::bench::bench_ram_shape();
    for key in [
        "usearch_bytes",
        "vectors_bin_bytes",
        "lexical_bytes",
        "catalog_bytes",
        "tombstone_bytes",
        "bm25_bytes",
        "rss_model_loaded",
        "rss_model_dropped",
    ] {
        assert!(report.get(key).is_some(), "{key}");
    }
}

#[test]
fn bench_ram_fills_the_byte_counts() {
    let out = gyr(&["bench", "ram", "--vectors", "2000", "--embedder", "hash"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let report: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(report["vectors_bin_bytes"], 2000 * 384);
    assert!(report["usearch_bytes"].as_u64().unwrap() > 0);
    assert!(report["lexical_bytes"].as_u64().unwrap() > 0);
    assert!(report["rss_model_loaded"].as_u64().unwrap() > 0);
    assert!(report["rss_index_open"].as_u64().unwrap() >= report["rss_baseline"].as_u64().unwrap());
    assert_eq!(report["tombstone_count"], 0);
}

#[test]
fn harness_dedupes_before_metrics() {
    let keys = dedupe_doc_keys(&["a#0".into(), "a#0".into(), "b#1".into()]);
    assert_eq!(keys, vec!["a#0".to_string(), "b#1".to_string()]);
    let at_one = retrieval_metrics(Some(1));
    assert_eq!(at_one.recall_at_5, 1.0);
    assert_eq!(at_one.ndcg_at_10, 1.0);
    let at_eleven = retrieval_metrics(Some(11));
    assert_eq!(at_eleven.recall_at_10, 0.0);
    assert_eq!(at_eleven.mrr, 1.0 / 11.0);
    assert_eq!(at_eleven.ndcg_at_10, 0.0);
    let at_three = retrieval_metrics(Some(3));
    assert_eq!(at_three.ndcg_at_10, 1.0 / 4f64.log2());
    assert_eq!(retrieval_metrics(None).mrr, 0.0);
}

fn four_text_only_reports() -> Vec<AccuracyReport> {
    ["gyrfalcon", "rustyrag", "swiftide", "llamaindex"]
        .iter()
        .map(|system| AccuracyReport {
            system: system.to_string(),
            generator_id: "judge-a".into(),
            answer_prompt_sha256: "p".into(),
            judge_id: "judge-a".into(),
            question_id_sha256: "q".into(),
            dataset_revision: "r".into(),
            chunk_count: 100,
            questions: vec![QuestionResult {
                query_id: "q1".into(),
                relevant: "x#2".into(),
                // A duplicate before the relevant key: rank is 2 after the dedupe, not 3.
                doc_keys: ["x#1", "x#1", "x#2", "x#3", "x#4", "x#5", "x#6", "x#7", "x#8", "x#9"]
                    .map(String::from)
                    .to_vec(),
                ..QuestionResult::default()
            }],
            passes: 1,
            ..AccuracyReport::default()
        })
        .collect()
}

#[test]
fn answer_table_requires_the_same_generator() {
    let reports = four_text_only_reports();
    let same = compare_accuracy(&reports).unwrap();
    assert!(same.answer.is_some());
    assert_eq!(same.retrieval.as_ref().unwrap()[0].score.mrr, 0.5);

    let mut reports = four_text_only_reports();
    reports[1].generator_id = "other".into();
    let compared = compare_accuracy(&reports).unwrap();
    assert!(compared.answer.is_none());
    assert!(compared.retrieval.is_some());
    assert!(compared.json.get("published_full_set").is_none());
    assert!(compared.json.get("answer").is_none());
}

#[test]
fn comparison_aborts_on_chunk_count_or_short_lists() {
    let mut reports = four_text_only_reports();
    reports[2].chunk_count = 99;
    assert!(compare_accuracy(&reports).is_err());
    let mut reports = four_text_only_reports();
    reports[3].questions[0].doc_keys.truncate(9);
    assert!(compare_accuracy(&reports).is_err());
}

fn write_json(path: &Path, value: serde_json::Value) {
    std::fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
}

/// Two papers of four sections each; three text questions and one image question.
fn tiny_dataset(dir: &Path) {
    std::fs::create_dir_all(dir.join("corpus")).unwrap();
    let topics = [
        ["borrow checker ownership", "lifetimes and references", "unsafe blocks", "traits and generics"],
        ["sourdough starter", "proofing dough", "oven temperature", "crust and crumb"],
    ];
    for (p, sections) in topics.iter().enumerate() {
        write_json(
            &dir.join(format!("corpus/paper{p}.json")),
            serde_json::json!({
                "title": format!("Paper {p}"),
                "abstract": format!("Abstract of paper {p}"),
                "sections": sections.iter().map(|s| serde_json::json!({ "text": format!("This section covers {s} in detail.") })).collect::<Vec<_>>(),
            }),
        );
    }
    write_json(
        &dir.join("queries.json"),
        serde_json::json!({
            "u1": { "query": "how do lifetimes and references work", "source": "text", "type": "abstractive" },
            "u2": { "query": "what oven temperature", "source": "text", "type": "extractive" },
            "u3": { "query": "sourdough starter", "source": "text", "type": "extractive" },
            "u4": { "query": "figure of crumb", "source": "image", "type": "extractive" },
        }),
    );
    write_json(&dir.join("answers.json"), serde_json::json!({ "u1": "a", "u2": "b", "u3": "c", "u4": "d" }));
    write_json(
        &dir.join("qrels.json"),
        serde_json::json!({
            "u1": { "doc_id": "paper0", "section_id": 1 },
            "u2": { "doc_id": "paper1", "section_id": 2 },
            "u3": { "doc_id": "paper1", "section_id": 0 },
            "u4": { "doc_id": "paper1", "section_id": 3 },
        }),
    );
}

#[test]
fn accuracy_prepare_run_adapters_and_compare() {
    let dir = tempfile::tempdir().unwrap();
    let dataset = dir.path().join("dataset");
    let fixture = dir.path().join("fixture");
    let reports = dir.path().join("reports");
    tiny_dataset(&dataset);

    // The count gate aborts on the real-slice default.
    let refused = gyr(&[
        "bench",
        "accuracy",
        "--prepare",
        "--dataset",
        dataset.to_str().unwrap(),
        "--fixture",
        fixture.to_str().unwrap(),
    ]);
    assert!(!refused.status.success());
    assert!(String::from_utf8_lossy(&refused.stderr).contains("1914"));

    let prepared = gyr(&[
        "bench",
        "accuracy",
        "--prepare",
        "--dataset",
        dataset.to_str().unwrap(),
        "--fixture",
        fixture.to_str().unwrap(),
        "--expected-questions",
        "3",
        "--dataset-revision",
        "test-rev",
    ]);
    assert!(prepared.status.success(), "{}", String::from_utf8_lossy(&prepared.stderr));
    let meta: serde_json::Value = serde_json::from_slice(&prepared.stdout).unwrap();
    assert_eq!(meta["questions"], 3);
    assert_eq!(meta["chunks"], 12);
    let questions = std::fs::read_to_string(fixture.join("questions.jsonl")).unwrap();
    assert!(questions.contains("\"relevant\":\"paper0#1\""));

    let gyrfalcon_report = reports.join("gyrfalcon.json");
    let ran = gyr(&[
        "bench",
        "accuracy",
        "--fixture",
        fixture.to_str().unwrap(),
        "--out",
        gyrfalcon_report.to_str().unwrap(),
        "--embedder",
        "hash",
    ]);
    assert!(ran.status.success(), "{}", String::from_utf8_lossy(&ran.stderr));
    let report: serde_json::Value = serde_json::from_slice(&std::fs::read(&gyrfalcon_report).unwrap()).unwrap();
    assert_eq!(report["chunk_count"], 12);
    assert_eq!(report["questions"].as_array().unwrap().len(), 3);
    assert!(report["retrieval"]["recall_at_10"].as_f64().unwrap() > 0.0);

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fake = example_binary("fake_adapter");
    for system in ["rustyrag", "swiftide", "llamaindex"] {
        let script = root.join(format!("bench/adapters/{system}/run.sh"));
        let without = Command::new("sh")
            .arg(&script)
            .env("BENCH_FIXTURE", &fixture)
            .env_remove("BENCH_ADAPTER_CMD")
            .env("BENCH_REPORTS", &reports)
            .output()
            .unwrap();
        assert_eq!(without.status.code(), Some(2));
        assert!(String::from_utf8_lossy(&without.stdout).contains("chunks 12"));
        assert!(!reports.join(format!("{system}.json")).exists());
        let with = Command::new("sh")
            .arg(&script)
            .env("BENCH_FIXTURE", &fixture)
            .env("BENCH_ADAPTER_CMD", &fake)
            .env("BENCH_REPORTS", &reports)
            .output()
            .unwrap();
        assert!(with.status.success(), "{}", String::from_utf8_lossy(&with.stderr));
        assert!(reports.join(format!("{system}.json")).exists());
    }

    let compared = gyr(&[
        "bench",
        "accuracy",
        "--compare",
        reports.join("gyrfalcon.json").to_str().unwrap(),
        reports.join("rustyrag.json").to_str().unwrap(),
        reports.join("swiftide.json").to_str().unwrap(),
        reports.join("llamaindex.json").to_str().unwrap(),
    ]);
    assert!(compared.status.success(), "{}", String::from_utf8_lossy(&compared.stderr));
    let table: serde_json::Value = serde_json::from_slice(&compared.stdout).unwrap();
    assert_eq!(table["retrieval"].as_array().unwrap().len(), 4);
    assert!(table.get("published_full_set").is_none());
    // Same "none" generator and judge on every report, so the answer table is present.
    assert_eq!(table["answer"]["rows"].as_array().unwrap().len(), 4);
}

#[test]
fn beir_metrics_match_hand_values() {
    use gyrfalcon::bench::{mrr_at, ndcg_at, recall_at, Qrels};
    let qrels: Qrels = [("a".to_string(), 2), ("b".to_string(), 1), ("z".to_string(), 0)].into_iter().collect();
    let ranked: Vec<String> = ["x", "b", "a"].map(String::from).to_vec();
    // DCG = 1/log2(3) + 2/log2(4); IDCG = 2/log2(2) + 1/log2(3).
    let dcg = 1.0 / 3f64.log2() + 2.0 / 2.0;
    let idcg = 2.0 + 1.0 / 3f64.log2();
    assert!((ndcg_at(&ranked, &qrels, 10) - dcg / idcg).abs() < 1e-12);
    assert_eq!(recall_at(&ranked, &qrels, 2), 0.5);
    assert_eq!(recall_at(&ranked, &qrels, 100), 1.0);
    assert_eq!(mrr_at(&ranked, &qrels, 10), 0.5);
    assert_eq!(ndcg_at(&["a".to_string(), "b".to_string()], &qrels, 10), 1.0);
}
