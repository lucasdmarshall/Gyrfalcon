use gyrfalcon::planner::{hash_split, paired_bootstrap, replay, score, Policy, QueryTrace};
use gyrfalcon::{Document, Index, IndexConfig, RetrieveOptions, ScriptedReranker};

fn trace() -> QueryTrace {
    let mut trace = QueryTrace { query_id: "q".into(), ..QueryTrace::default() };
    trace.lexical = vec![("a".into(), 9.0), ("b".into(), 5.0), ("c".into(), 1.0)];
    trace.dense = vec![("c".into(), 0.9), ("b".into(), 0.6)];
    trace.exact = [("a", 0.1), ("b", 0.6), ("c", 0.9)].into_iter().map(|(d, s)| (d.to_string(), s)).collect();
    trace.chunks = [("a", 0), ("b", 1), ("c", 2)].into_iter().map(|(d, c)| (d.to_string(), c)).collect();
    trace.rerank = [("a", 0.0), ("b", 5.0), ("c", 1.0)].into_iter().map(|(d, s)| (d.to_string(), s)).collect();
    trace.qrels = [("b".to_string(), 1)].into_iter().collect();
    trace.timings.lexical = 1.0;
    trace.timings.embed = 2.0;
    trace.timings.rerank_per_pair = 10.0;
    trace
}

#[test]
fn replay_rebuilds_each_policy() {
    let t = trace();
    assert_eq!(replay(&t, Policy::Lexical, 10).0, vec!["a", "b", "c"]);
    assert_eq!(replay(&t, Policy::Dense, 10).0, vec!["c", "b"]);
    // a: 0.8*1 + 0.2*0 = 0.8; b: 0.8*0.5 + 0.2*0.625 = 0.525; c: 0 + 0.2 = 0.2.
    assert_eq!(replay(&t, Policy::Rescore { dense_weight: 0.2, rerank_k: 0 }, 10).0, vec!["a", "b", "c"]);
    assert_eq!(replay(&t, Policy::Rescore { dense_weight: 0.8, rerank_k: 0 }, 10).0, vec!["c", "b", "a"]);
    // Rerank the top 2 of [a, b, c]: b (5.0) before a (0.0); c stays third.
    let (ranked, cost) = replay(&t, Policy::Rescore { dense_weight: 0.2, rerank_k: 2 }, 10);
    assert_eq!(ranked, vec!["b", "a", "c"]);
    assert!(cost >= 1.0 + 2.0 + 20.0);
    assert_eq!(score(&t, Policy::Rescore { dense_weight: 0.2, rerank_k: 2 }).mrr_at_10, 1.0);
    assert_eq!(score(&t, Policy::Lexical).mrr_at_10, 0.5);
    // BM25 / 9 plus cosine at dense weight 0.9: c 0.821, b 0.596, a 0.190.
    assert_eq!(replay(&t, Policy::RescoreMax { dense_weight: 0.9, rerank_k: 0 }, 10).0, vec!["c", "b", "a"]);
    // Rerank the top 2 of that order: b (5.0) before c (1.0); a stays third.
    let (ranked, cost) = replay(&t, Policy::RescoreMax { dense_weight: 0.9, rerank_k: 2 }, 10);
    assert_eq!(ranked, vec!["b", "c", "a"]);
    assert!(cost >= 20.0);
}

#[test]
fn replay_drops_the_query_id_like_beir() {
    let mut t = trace();
    t.query_id = "a".into();
    assert_eq!(replay(&t, Policy::Lexical, 10).0, vec!["b", "c"]);
}

#[test]
fn policies_parse() {
    assert_eq!(Policy::parse("rescoremax:0.45:8"), Some(Policy::RescoreMax { dense_weight: 0.45, rerank_k: 8 }));
    assert_eq!(Policy::parse("rescore:0.3:16"), Some(Policy::Rescore { dense_weight: 0.3, rerank_k: 16 }));
    assert_eq!(Policy::parse("rescore:0.3"), Some(Policy::Rescore { dense_weight: 0.3, rerank_k: 0 }));
    assert_eq!(Policy::parse("lexical"), Some(Policy::Lexical));
    assert_eq!(Policy::parse("nope"), None);
}

#[test]
fn bootstrap_separates_real_differences_from_noise() {
    let a: Vec<f64> = (0..200).map(|i| if i % 2 == 0 { 1.0 } else { 0.5 }).collect();
    let same = paired_bootstrap(&a, &a, 2000, 1);
    assert_eq!(same, (0.0, 0.0, 0.0));
    let better: Vec<f64> = a.iter().map(|x| x + 0.1).collect();
    let (mean, low, high) = paired_bootstrap(&better, &a, 2000, 1);
    assert!((mean - 0.1).abs() < 1e-9 && low > 0.0 && high > 0.0);
    let noisy: Vec<f64> = a.iter().enumerate().map(|(i, x)| if i % 4 < 2 { x + 0.2 } else { x - 0.2 }).collect();
    let (_, low, high) = paired_bootstrap(&noisy, &a, 2000, 1);
    assert!(low < 0.0 && high > 0.0);
}

#[test]
fn hash_split_is_stable_and_roughly_even() {
    let validation = (0..2000).filter(|i| hash_split(&format!("q{i}")) == "validation").count();
    assert!((900..1100).contains(&validation), "{validation}");
    assert_eq!(hash_split("q1"), hash_split("q1"));
}

#[test]
fn live_trace_replays_to_the_live_ranking() {
    let mut index = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    let docs: Vec<Document> =
        (0..30).map(|i| Document::body(format!("d{i}"), format!("word{} shared text {}", i % 7, i))).collect();
    index.index_sections(&docs).unwrap();
    let reranker = ScriptedReranker::new(vec![("word3".into(), 2.0)]);
    for weight in [0.2f32, 0.6] {
        index.fusion = gyrfalcon::Fusion::Rescore { dense_weight: weight };
        let mut trace = index.trace_query("word3 shared", 100, Some((&reranker, 8))).unwrap();
        trace.query_id = "none".into();
        let live: Vec<String> = index
            .retrieve("word3 shared", RetrieveOptions { rerank: false, limit: 100 })
            .unwrap()
            .hits
            .into_iter()
            .map(|h| h.doc_id)
            .collect();
        assert_eq!(replay(&trace, Policy::Rescore { dense_weight: weight, rerank_k: 0 }, 100).0, live);
        assert!(!trace.rerank.is_empty());
        assert!(trace.features.query_tokens == 2 && trace.features.idf_max > 0.0);
    }
    index.fusion = gyrfalcon::Fusion::RescoreMax { dense_weight: 0.45 };
    let mut trace = index.trace_query("word3 shared", 100, Some((&reranker, 8))).unwrap();
    trace.query_id = "none".into();
    let live: Vec<String> = index
        .retrieve("word3 shared", RetrieveOptions { rerank: false, limit: 100 })
        .unwrap()
        .hits
        .into_iter()
        .map(|h| h.doc_id)
        .collect();
    assert_eq!(replay(&trace, Policy::RescoreMax { dense_weight: 0.45, rerank_k: 0 }, 100).0, live);
}
