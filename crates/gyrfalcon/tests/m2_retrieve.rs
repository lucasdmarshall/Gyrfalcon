use gyrfalcon::testing::CountingEmbedder;
use gyrfalcon::{Document, Error, FallbackStep, Index, IndexConfig, Quantization, RetrieveOptions};

fn unit(dimension: usize, axis: usize, sign: f32) -> Vec<f32> {
    let mut vector = vec![0.0; dimension];
    vector[axis] = sign;
    vector
}

#[test]
fn collection_doc_freq_is_the_sum_and_ignores_nothing_dead() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.index_documents(&[Document::body("a", "rust safety")]).unwrap();
    index.index_documents(&[Document::body("b", "rust cooking")]).unwrap();
    assert_eq!(index.manifest().sealed_segments.len(), 2);
    let before = index.lexical_search(&gyrfalcon::prepare("rust").tokens, 10).unwrap();
    index.testing_tombstone_chunk(0);
    let manifest = index.manifest();
    assert_eq!(manifest.bm25_num_docs, 2);
    assert_eq!(manifest.bm25_generation, 2);
    let hits = index.lexical_search(&gyrfalcon::prepare("rust").tokens, 10).unwrap();
    // The dead chunk never takes a slot; its document frequency stays in the collection sum.
    assert!(!hits.iter().any(|hit| hit.chunk_id == 0));
    assert_eq!(hits.iter().map(|hit| hit.chunk_id).collect::<Vec<_>>(), vec![1]);
    assert_eq!(index.collection_doc_freq("rust"), 2);
    assert_eq!(index.collection_doc_freq("safety"), 1);
    // The score uses the collection statistics, so it does not move when chunk 0 dies.
    let live_before = before.iter().find(|hit| hit.chunk_id == 1).unwrap().score;
    assert!((hits[0].score - live_before).abs() < 1e-6);
}

#[test]
fn collection_scores_differ_from_segment_local_scores() {
    // Segment-local IDF of "rare" would be identical in both one-chunk segments. With
    // collection statistics a common term ("shared") scores lower than a rare one.
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    let docs: Vec<_> = (0..9).map(|i| Document::body(format!("s{i}"), "shared words")).collect();
    index.index_documents(&docs).unwrap();
    index.index_documents(&[Document::body("rare", "rare words")]).unwrap();
    let shared = index.lexical_search(&gyrfalcon::prepare("shared").tokens, 1).unwrap()[0].score;
    let rare = index.lexical_search(&gyrfalcon::prepare("rare").tokens, 1).unwrap()[0].score;
    assert!(rare > shared, "rare {rare} shared {shared}");
}

fn ninety_dead_of_100(config: &IndexConfig) -> (tempfile::TempDir, Index) {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), config).unwrap();
    let mut rows = Vec::new();
    for i in 0..100 {
        let mut vector = vec![0.0; 384];
        vector[0] = if i < 90 { -1.0 } else { 1.0 };
        vector[1 + i] = 0.5;
        rows.push((Document::body(format!("d{i}"), "shared term"), vector));
    }
    index.index_documents_with_vectors(&rows).unwrap();
    for id in 0..90 {
        index.testing_tombstone_chunk(id);
    }
    (dir, index)
}

#[test]
fn filtered_search_returns_ten_live_ids_when_ninety_are_dead() {
    let (_dir, index) = ninety_dead_of_100(&IndexConfig::test_default());
    assert_eq!(index.manifest().sealed_segments.len(), 1);
    let query = unit(384, 0, 1.0);
    let hits = index.vector_search(&query, 10).unwrap();
    assert_eq!(hits.ids.len(), 10);
    assert!(hits.ids.iter().all(|id| *id >= 90));
    assert_eq!(hits.deepest, FallbackStep::E64);

    let lexical = index.lexical_search(&gyrfalcon::prepare("term").tokens, 10).unwrap();
    assert_eq!(lexical.len(), 10);
    assert!(lexical.iter().all(|hit| hit.chunk_id >= 90));
}

#[test]
fn dense_list_is_short_only_when_live_chunks_run_out() {
    let (_dir, index) = ninety_dead_of_100(&IndexConfig::test_default());
    let hits = index.vector_search(&unit(384, 0, -1.0), 50).unwrap();
    assert_eq!(hits.ids.len(), 10);
    assert!(hits.ids.iter().all(|id| *id >= 90));
}

#[test]
fn exact_scan_is_recorded_when_expansion_cannot_fill() {
    let (_dir, mut index) = ninety_dead_of_100(&IndexConfig::test_default());
    index.testing_starve_hnsw = true;
    let hits = index.vector_search(&unit(384, 0, 1.0), 10).unwrap();
    assert_eq!(hits.deepest, FallbackStep::Exact);
    assert_eq!(hits.ids.len(), 10);
    assert!(hits.ids.iter().all(|id| *id >= 90));
    let retrieved = index.retrieve("term", RetrieveOptions { rerank: false, limit: 5 }).unwrap();
    assert_eq!(retrieved.fallback, FallbackStep::Exact);
}

#[test]
fn binary_mode_maps_vectors_bin_for_rescore() {
    let mut config = IndexConfig::test_default();
    config.quantization = Quantization::B1;
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &config).unwrap();
    let vector = unit(384, 0, 1.0);
    index.index_documents_with_vectors(&[(Document::body("a", "rust"), vector.clone())]).unwrap();
    let mapped = index.vector_search_mapped_paths();
    assert!(mapped.iter().any(|path| path.ends_with("vectors.bin")));
    let hits = index.vector_search(&vector, 1).unwrap();
    assert_eq!(hits.ids, vec![0]);
}

#[test]
fn binary_mode_rescore_orders_by_cosine() {
    let mut config = IndexConfig::test_default();
    config.quantization = Quantization::B1;
    let (_dir, index) = ninety_dead_of_100(&config);
    let hits = index.vector_search(&unit(384, 0, 1.0), 10).unwrap();
    assert_eq!(hits.ids.len(), 10);
    assert!(hits.ids.iter().all(|id| *id >= 90));
    assert!(hits.distances.windows(2).all(|w| w[0] <= w[1]));
}

#[test]
fn int8_search_does_not_map_vectors_bin() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    let vector = unit(384, 0, 1.0);
    index.index_documents_with_vectors(&[(Document::body("a", "rust"), vector.clone())]).unwrap();
    let hits = index.vector_search(&vector, 1).unwrap();
    assert_eq!(hits.ids, vec![0]);
    assert!(index.vector_search_mapped_paths().iter().all(|path| !path.ends_with("vectors.bin")));
}

#[test]
fn rrf_uses_constant_60() {
    let fused = gyrfalcon::rrf(&[10, 11], &[11, 10]);
    let score = |rank: u32| 1.0 / (60.0 + rank as f32);
    assert!((fused[0].1 - (score(1) + score(2))).abs() < 1e-6);
    assert_eq!(fused[0].0, 10);
    assert_eq!(fused[1].0, 11);
    let one_leg = gyrfalcon::rrf(&[7], &[]);
    assert!((one_leg[0].1 - score(1)).abs() < 1e-6);
}

#[test]
fn rrf_weights_scale_each_leg() {
    let fused = gyrfalcon::rrf_weighted(&[10, 11], &[11, 10], [2.0, 0.0]);
    assert_eq!(fused[0].0, 10);
    assert!((fused[0].1 - 2.0 / 61.0).abs() < 1e-6);
    assert!((fused[1].1 - 2.0 / 62.0).abs() < 1e-6);
}

#[test]
fn retrieve_prepares_the_query_once() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    let embedder = CountingEmbedder::new(384);
    index.set_embedder(Box::new(embedder.clone()));
    index.index_documents_with_vectors(&[(Document::body("a", "rust safety"), vec![1.0; 384])]).unwrap();
    assert_eq!(embedder.calls(), 0);
    let hits = index.retrieve("rust", RetrieveOptions { rerank: false, limit: 5 }).unwrap();
    assert_eq!(hits.embed_calls, 1);
    assert_eq!(embedder.calls(), 1);
    assert_eq!(hits.chunk_ids, vec![0]);
    assert_eq!(hits.hits[0].doc_id, "a");
    assert_eq!(hits.hits[0].body, "rust safety");
}

#[test]
fn retrieve_fuses_both_legs_across_segments() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.index_documents(&[Document::body("a", "memory safety in rust")]).unwrap();
    index.index_documents(&[Document::body("b", "baking bread at home")]).unwrap();
    index.index_documents(&[Document::body("c", "rust borrow checker")]).unwrap();
    let hits = index.retrieve("Memory safety in Rust", RetrieveOptions { rerank: false, limit: 3 }).unwrap();
    assert_eq!(hits.chunk_ids[0], 0);
    assert_eq!(hits.chunk_ids.len(), 3);
    let rerank_on = index.retrieve("Memory safety in Rust", RetrieveOptions { rerank: true, limit: 3 }).unwrap();
    assert_eq!(rerank_on.chunk_ids, hits.chunk_ids);
}

#[test]
fn empty_query_is_rejected_before_either_leg() {
    let index = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    let err = index.retrieve("  \n ", RetrieveOptions { rerank: false, limit: 5 }).unwrap_err();
    assert!(matches!(err, Error::EmptyQuery));
}

#[test]
fn empty_collection_returns_no_hits() {
    let index = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    let hits = index.retrieve("rust", RetrieveOptions { rerank: false, limit: 5 }).unwrap();
    assert!(hits.chunk_ids.is_empty());
}

#[test]
fn reopen_searches_the_sealed_engines() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.index_documents(&[Document::body("a", "rust safety"), Document::body("b", "bread")]).unwrap();
    drop(index);
    let index = Index::open(dir.path(), &IndexConfig::test_default()).unwrap();
    let hits = index.retrieve("bread", RetrieveOptions { rerank: false, limit: 1 }).unwrap();
    assert_eq!(hits.hits[0].doc_id, "b");
}

#[test]
fn score_fusion_normalizes_each_leg() {
    // Lexical: 10 is best. Dense: 11 is best. With dense weight 0.2 lexical wins.
    let fused = gyrfalcon::score_fusion(&[(10, 8.0), (11, 4.0)], &[(11, 0.9), (10, 0.5)], 0.2);
    assert_eq!(fused[0].0, 10);
    assert!((fused[0].1 - 0.8).abs() < 1e-6);
    assert!((fused[1].1 - 0.2).abs() < 1e-6);
    let dense_heavy = gyrfalcon::score_fusion(&[(10, 8.0), (11, 4.0)], &[(11, 0.9), (10, 0.5)], 0.8);
    assert_eq!(dense_heavy[0].0, 11);
}

#[test]
fn rescore_gives_lexical_only_hits_their_exact_similarity() {
    let (_dir, mut index) = ninety_dead_of_100(&IndexConfig::test_default());
    index.fusion = gyrfalcon::Fusion::Rescore { dense_weight: 0.3 };
    let hits = index.retrieve("term", RetrieveOptions { rerank: false, limit: 10 }).unwrap();
    assert_eq!(hits.chunk_ids.len(), 10);
    assert!(hits.chunk_ids.iter().all(|id| *id >= 90));
    // Rescoring reads stored vectors through USearch; int8 still leaves vectors.bin unmapped.
    assert!(index.vector_search_mapped_paths().iter().all(|path| !path.ends_with("vectors.bin")));
    index.fusion = gyrfalcon::Fusion::Score { dense_weight: 0.3 };
    assert_eq!(index.retrieve("term", RetrieveOptions { rerank: false, limit: 10 }).unwrap().chunk_ids.len(), 10);
}

#[test]
fn many_concurrent_searches_wait_for_a_slot_instead_of_failing() {
    let (_dir, index) = ninety_dead_of_100(&IndexConfig::test_default());
    std::thread::scope(|scope| {
        for _ in 0..96 {
            scope.spawn(|| {
                for _ in 0..20 {
                    assert_eq!(index.vector_search(&unit(384, 0, 1.0), 10).unwrap().ids.len(), 10);
                }
            });
        }
    });
}
