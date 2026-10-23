use gyrfalcon::testing::RecordingEmbedder;
use gyrfalcon::{
    Document, Error, HttpEmbedder, IncompatibleField, Index, IndexConfig, RetrieveOptions, ScriptedReranker, ScriptedSession,
};

fn sealed_one_doc() -> (Index, ScriptedReranker) {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.index_documents(&[Document::body("a", "rust safety")]).unwrap();
    let reranker = ScriptedReranker::new(vec![]);
    index.set_reranker(Box::new(reranker.clone()));
    index.leak_tempdir(dir);
    (index, reranker)
}

#[test]
fn onnx_width_mismatch_is_dimension() {
    let dir = tempfile::tempdir().unwrap();
    Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    let err = Index::open_with_session(dir.path(), &IndexConfig::test_default(), ScriptedSession { width: 768 }).err().unwrap();
    match err {
        Error::IncompatibleIndex { field, manifest_value, requested_value } => {
            assert_eq!(field, IncompatibleField::Dimension);
            assert_eq!(manifest_value, "384");
            assert_eq!(requested_value, "768");
        }
        other => panic!("{other:?}"),
    }
    // No handle was returned, so the writer lock is free.
    Index::open(dir.path(), &IndexConfig::test_default()).unwrap();
}

#[test]
fn matching_session_width_opens_and_embeds() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.index_documents(&[Document::body("a", "rust safety")]).unwrap();
    drop(index);
    let index = Index::open_with_session(dir.path(), &IndexConfig::test_default(), ScriptedSession { width: 384 }).unwrap();
    let hits = index.retrieve("rust", RetrieveOptions { rerank: false, limit: 5 }).unwrap();
    assert_eq!(hits.chunk_ids, vec![0]);
}

#[test]
fn missing_onnx_file_is_model_unavailable() {
    let dir = tempfile::tempdir().unwrap();
    Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    let models = tempfile::tempdir().unwrap();
    let err = Index::open_with_model(dir.path(), &IndexConfig::test_default(), models.path()).err().unwrap();
    assert!(matches!(err, Error::ModelUnavailable(_)), "{err:?}");
}

#[test]
fn rerank_off_does_not_call_the_session() {
    let (index, session) = sealed_one_doc();
    let _ = index.retrieve("rust", RetrieveOptions { rerank: false, limit: 5 }).unwrap();
    assert_eq!(session.rerank_calls(), 0);
    let _ = index.retrieve("rust", RetrieveOptions { rerank: true, limit: 5 }).unwrap();
    assert_eq!(session.rerank_calls(), 1);
}

#[test]
fn reranker_orders_a_scripted_high_score_first() {
    let mut index = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    index
        .index_documents(&[
            Document::body("a", "rust rust rust safety"),
            Document::body("b", "rust cooking"),
            Document::body("c", "rust gardening"),
        ])
        .unwrap();
    let off = index.retrieve("rust", RetrieveOptions { rerank: false, limit: 3 }).unwrap();
    assert_eq!(off.chunk_ids.len(), 3);
    assert_ne!(off.chunk_ids[0], 2);
    index.set_reranker(Box::new(ScriptedReranker::new(vec![("gardening".into(), 9.0), ("cooking".into(), 1.0)])));
    let on = index.retrieve("rust", RetrieveOptions { rerank: true, limit: 3 }).unwrap();
    assert_eq!(on.chunk_ids, vec![2, 1, 0]);
    assert_eq!(on.hits[0].score, 9.0);
}

#[test]
fn embed_receives_embed_text_unchanged() {
    let mut index = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    index.set_embedder(Box::new(RecordingEmbedder::new(384)));
    index.index_documents(&[Document::body("a", "  Memory Safety in Rust\n")]).unwrap();
    assert_eq!(index.embedder_texts(), vec!["Memory Safety in Rust".to_string()]);
    index.retrieve("  Cafe\u{301} QUERY ", RetrieveOptions { rerank: false, limit: 1 }).unwrap();
    assert_eq!(index.embedder_texts()[1], "Café QUERY");
}

#[test]
fn http_embedder_rejects_the_wrong_width() {
    let embedder = HttpEmbedder::new("http://127.0.0.1:9/v1/embeddings", 384);
    let err = embedder.check_width(768).unwrap_err();
    match err {
        Error::IncompatibleIndex { field, .. } => assert_eq!(field, IncompatibleField::Dimension),
        other => panic!("{other:?}"),
    }
    embedder.check_width(384).unwrap();
}

#[test]
fn ingest_rejects_an_embedder_of_the_wrong_width() {
    let mut index = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    index.set_embedder(Box::new(RecordingEmbedder::new(768)));
    let err = index.index_documents(&[Document::body("a", "rust")]).unwrap_err();
    assert!(matches!(err, Error::IncompatibleIndex { field: IncompatibleField::Dimension, .. }), "{err:?}");
    assert!(index.manifest().sealed_segments.is_empty());
}

/// Runs only when `GYR_ORT_FIXTURE` names a directory with `embed/` and `rerank/` model
/// directories (`model.onnx` and `tokenizer.json`) and ONNX Runtime can be loaded.
#[test]
fn ort_fixture_round_trip() {
    let Ok(fixture) = std::env::var("GYR_ORT_FIXTURE") else {
        eprintln!("GYR_ORT_FIXTURE is not set; skipping");
        return;
    };
    let fixture = std::path::Path::new(&fixture);
    let embedder = gyrfalcon::OrtEmbedder::from_dir(&fixture.join("embed"), gyrfalcon::DEFAULT_MODEL_ID).unwrap();
    use gyrfalcon::Embedder;
    assert_eq!(embedder.dimension(), 384);
    let vectors = embedder.embed(&["memory safety".to_string(), "bread".to_string()]).unwrap();
    assert_eq!(vectors.len(), 2);
    let norm: f32 = vectors[0].iter().map(|v| v * v).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4);

    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.set_embedder(Box::new(embedder));
    index.index_documents(&[Document::body("a", "memory safety in rust"), Document::body("b", "bread cooking")]).unwrap();
    drop(index);
    let mut index = Index::open_with_model(dir.path(), &IndexConfig::test_default(), &fixture.join("embed")).unwrap();
    index.set_reranker(Box::new(gyrfalcon::OrtReranker::from_dir(&fixture.join("rerank")).unwrap()));
    let hits = index.retrieve("bread", RetrieveOptions { rerank: true, limit: 2 }).unwrap();
    assert_eq!(hits.chunk_ids.len(), 2);
    assert!(hits.hits.iter().all(|hit| hit.score.is_finite()));
}
