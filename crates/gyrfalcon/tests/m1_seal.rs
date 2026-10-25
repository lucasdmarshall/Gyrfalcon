use gyrfalcon::{Document, Error, IncompatibleField, Index, IndexConfig, Manifest, Quantization};

fn assert_mismatch(other: IndexConfig, field: IncompatibleField, manifest_value: &str, requested_value: &str) {
    let dir = tempfile::tempdir().unwrap();
    Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    let mut opens = 0u32;
    let err = Index::open_counting(dir.path(), &other, &mut opens).unwrap_err();
    match err {
        Error::IncompatibleIndex { field: got, manifest_value: m, requested_value: r } => {
            assert_eq!(got, field);
            assert_eq!(m, manifest_value);
            assert_eq!(r, requested_value);
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(opens, 0);
}

#[test]
fn open_rejects_quantization_mismatch_without_mapping() {
    let dir = tempfile::tempdir().unwrap();
    let config = IndexConfig::test_default();
    Index::create(dir.path(), &config).unwrap();
    let mut other = config.clone();
    other.quantization = Quantization::F32;
    let mut opens = 0u32;
    let err = Index::open_counting(dir.path(), &other, &mut opens).unwrap_err();
    match err {
        Error::IncompatibleIndex { field, manifest_value, requested_value } => {
            assert_eq!(field, IncompatibleField::Quantization);
            assert_eq!(manifest_value, "i8");
            assert_eq!(requested_value, "f32");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(opens, 0);
}

#[test]
fn open_rejects_model_id_mismatch_without_mapping() {
    let mut other = IndexConfig::test_default();
    other.model_id = "intfloat/e5-small-v2".into();
    assert_mismatch(other, IncompatibleField::ModelId, "BAAI/bge-small-en-v1.5", "intfloat/e5-small-v2");
}

#[test]
fn open_rejects_dimension_mismatch_without_mapping() {
    let mut other = IndexConfig::test_default();
    other.dimension = 768;
    assert_mismatch(other, IncompatibleField::Dimension, "384", "768");
}

#[test]
fn open_rejects_analyzer_mismatch_without_mapping() {
    let mut other = IndexConfig::test_default();
    other.analyzer_id = "gyr-text-v2".into();
    assert_mismatch(other, IncompatibleField::AnalyzerId, "gyr-text-v1", "gyr-text-v2");
}

#[test]
fn matching_open_counts_one_segment_open() {
    let dir = tempfile::tempdir().unwrap();
    Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    let mut opens = 0u32;
    Index::open_counting(dir.path(), &IndexConfig::test_default(), &mut opens).unwrap();
    assert_eq!(opens, 1);
}

#[test]
fn missing_directory_is_index_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let mut opens = 0u32;
    let err = Index::open_counting(&dir.path().join("absent"), &IndexConfig::test_default(), &mut opens).unwrap_err();
    assert!(matches!(err, Error::IndexNotFound(_)), "{err:?}");
}

#[test]
fn create_writes_epoch_1_and_empty_generation_0_files() {
    let dir = tempfile::tempdir().unwrap();
    Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    let manifest = Manifest::read(&dir.path().join("manifest.json")).unwrap();
    assert_eq!(manifest.epoch, 1);
    assert_eq!(manifest.format_version, 3);
    assert_eq!((manifest.tombstone_generation, manifest.catalog_generation, manifest.bm25_generation), (0, 0, 0));
    assert_eq!((manifest.bm25_num_docs, manifest.bm25_num_tokens), (0, 0));
    assert_eq!(manifest.segments_per_tier, 8);
    assert_eq!(manifest.max_merge_chunks, 100_000);
    assert_eq!(manifest.dead_ratio_limit, 0.20);
    assert!(manifest.sealed_segments.is_empty());
    for file in ["tombstones/0.bin", "catalog/0.bin", "bm25/0.bin"] {
        assert!(dir.path().join(file).is_file(), "{file}");
    }
    assert_eq!(std::fs::read(dir.path().join("tombstones/0.bin")).unwrap().len(), 0);
}

#[test]
fn prepare_lowercases_words_and_keeps_embed_case() {
    let prepared = gyrfalcon::prepare("Memory safety in Rust မြန်မာ E0308");
    assert_eq!(prepared.embed_text, "Memory safety in Rust မြန်မာ E0308");
    assert_eq!(
        prepared.tokens.iter().map(|t| t.text.as_str()).collect::<Vec<_>>(),
        vec!["memory", "safety", "in", "rust", "မြန်မာ", "e0308"]
    );
    assert_eq!(prepared.tokens.iter().map(|t| t.position).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4, 5]);
    let only_emoji = gyrfalcon::prepare("😀");
    assert!(only_emoji.tokens.is_empty());
}

#[test]
fn prepare_normalizes_trims_and_drops_long_pieces() {
    let prepared = gyrfalcon::prepare("  Cafe\u{301} \t error E0308  ");
    assert_eq!(prepared.embed_text, "Café \t error E0308");
    let long = "x".repeat(41);
    let kept = "y".repeat(40);
    let prepared = gyrfalcon::prepare(&format!("{long} {kept}"));
    assert_eq!(prepared.tokens.len(), 1);
    assert_eq!(prepared.tokens[0].text, kept);
    assert_eq!(prepared.tokens[0].position, 0);
    assert!(gyrfalcon::prepare("   \n\t ").tokens.is_empty());
}

#[test]
fn chunk_covers_bytes_without_splitting_utf8() {
    let text = "aaaa\n\nမြန်မာစာ";
    let spans = gyrfalcon::chunk_with(text, 8, 2);
    let mut covered = vec![false; text.len()];
    for span in &spans {
        assert!(text.is_char_boundary(span.start) && text.is_char_boundary(span.end));
        covered[span.start..span.end].fill(true);
    }
    assert!(covered.iter().all(|b| *b));
}

#[test]
fn chunk_prefers_paragraph_breaks_and_overlaps() {
    let text = format!("{}\n\n{}", "a".repeat(1500), "b".repeat(1500));
    let spans = gyrfalcon::chunk(&text);
    assert_eq!(spans.len(), 2);
    assert_eq!(spans[0].start, 0);
    assert_eq!(spans[0].end, 1502);
    assert_eq!(spans[1].start, 1502 - 256);
    assert_eq!(spans[1].end, text.len());
    let myanmar = "မြန်မာစာ။".repeat(200);
    for span in gyrfalcon::chunk(&myanmar) {
        assert!(span.end - span.start <= 2048);
        assert!(myanmar.is_char_boundary(span.start) && myanmar.is_char_boundary(span.end));
    }
}

#[test]
fn index_documents_publishes_chunk_ids_and_epoch_2() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    let report = index
        .index_documents(&[Document { doc_id: "a".into(), text: "alpha beta".into(), contextual_prefix: false }])
        .unwrap();
    assert_eq!(report.chunk_ids.len(), 1);
    let manifest = Manifest::read(&dir.path().join("manifest.json")).unwrap();
    assert_eq!(manifest.epoch, 2);
    assert_eq!(manifest.sealed_segments.len(), 1);
    assert_eq!(manifest.bm25_num_docs, 1);
    assert_eq!(manifest.bm25_num_tokens, 2);
    assert_eq!(manifest.bm25_generation, 1);
    assert_eq!(manifest.catalog_generation, 1);
    assert_eq!(manifest.tombstone_generation, 0);
    assert!(dir.path().join(format!("segments/{}/terms.bin", manifest.sealed_segments[0])).exists());
    assert!(dir.path().join(format!("segments/{}/vectors.bin", manifest.sealed_segments[0])).exists());
    let segment_json: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.path().join(format!("segments/{}/segment.json", manifest.sealed_segments[0]))).unwrap(),
    )
    .unwrap();
    assert_eq!(segment_json["chunk_count"], 1);
    assert_eq!(segment_json["num_docs"], 1);
    assert_eq!(segment_json["num_tokens"], 2);
    assert_eq!(segment_json["quantization"], "i8");
}

#[test]
fn crash_before_rename_keeps_the_previous_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.fail_before_rename = true;
    assert!(index
        .index_documents(&[Document { doc_id: "a".into(), text: "alpha".into(), contextual_prefix: false }])
        .is_err());
    let manifest = Manifest::read(&dir.path().join("manifest.json")).unwrap();
    assert!(manifest.sealed_segments.is_empty());
    assert_eq!(manifest.epoch, 1);

    // The sealed but unpublished directory is removed by the next open.
    assert_eq!(std::fs::read_dir(dir.path().join("segments")).unwrap().count(), 1);
    drop(index);
    let reopened = Index::open(dir.path(), &IndexConfig::test_default()).unwrap();
    assert_eq!(std::fs::read_dir(dir.path().join("segments")).unwrap().count(), 0);
    assert_eq!(reopened.manifest().epoch, 1);
}

#[test]
fn seal_splits_when_the_threshold_is_three() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.seal_every = 3;
    let docs: Vec<_> = (0..4).map(|i| Document::body(format!("d{i}"), format!("word {i}"))).collect();
    let report = index.index_documents(&docs).unwrap();
    assert_eq!(index.manifest().sealed_segments.len(), 2);
    assert_eq!(report.chunk_ids, vec![0, 1, 2, 3]);
}

#[test]
fn chunk_ids_survive_reopen_and_are_not_reused() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.index_documents(&[Document::body("a", "alpha")]).unwrap();
    drop(index);
    let mut index = Index::open(dir.path(), &IndexConfig::test_default()).unwrap();
    let report = index.index_documents(&[Document::body("b", "beta")]).unwrap();
    assert_eq!(report.chunk_ids, vec![1]);
    assert_eq!(index.manifest().sealed_segments.len(), 2);
}

#[test]
fn second_writer_is_busy_and_a_dead_pid_is_reclaimed() {
    let dir = tempfile::tempdir().unwrap();
    let index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    let err = Index::open(dir.path(), &IndexConfig::test_default()).err().unwrap();
    assert!(matches!(err, Error::WriterBusy { .. }), "{err:?}");
    drop(index);
    // A pid that cannot be alive.
    std::fs::write(dir.path().join("writer.lock"), "4194305").unwrap();
    Index::open(dir.path(), &IndexConfig::test_default()).unwrap();
}
