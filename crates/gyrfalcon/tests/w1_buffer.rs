//! The write buffer and its WAL: small writes are durable and searchable without a segment.

use gyrfalcon::{Document, Index, IndexConfig, RetrieveOptions};

fn docs(prefix: &str, n: usize) -> Vec<Document> {
    (0..n).map(|i| Document::body(format!("{prefix}{i}"), format!("rust memory safety note {i} {prefix}{i}"))).collect()
}

fn wal_files(dir: &std::path::Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir.join("wal"))
        .map(|entries| entries.flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect())
        .unwrap_or_default();
    names.sort();
    names
}

#[test]
fn small_writes_are_searchable_without_a_segment_or_a_manifest_write() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    let epoch = index.manifest().epoch;
    for doc in docs("d", 5) {
        index.index_documents(&[doc]).unwrap();
    }
    assert!(index.manifest().sealed_segments.is_empty());
    assert_eq!(index.manifest().epoch, epoch);
    assert_eq!(index.collection_doc_freq("safety"), 5);
    let hits = index.retrieve("note 3 d3", RetrieveOptions { rerank: false, limit: 3 }).unwrap();
    assert_eq!(hits.hits[0].doc_id, "d3");
    assert_eq!(hits.hits[0].body, "rust memory safety note 3 d3");
}

#[test]
fn the_buffer_survives_a_reopen_and_chunk_ids_are_not_reused() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    let first = index.index_documents(&docs("a", 3)).unwrap();
    index.delete_document("a1").unwrap();
    drop(index);

    let mut index = Index::open(dir.path(), &IndexConfig::test_default()).unwrap();
    assert!(index.manifest().sealed_segments.is_empty());
    let hits = index.lexical_search(&gyrfalcon::prepare("safety").tokens, 10).unwrap();
    let mut ids: Vec<u64> = hits.iter().map(|h| h.chunk_id).collect();
    ids.sort();
    assert_eq!(ids, vec![first.chunk_ids[0], first.chunk_ids[2]]);
    let next = index.index_documents(&docs("b", 1)).unwrap();
    assert!(next.chunk_ids[0] > *first.chunk_ids.iter().max().unwrap());
}

#[test]
fn an_update_of_a_buffered_document_is_atomic_and_durable() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.index_documents(&[Document::body("x", "old words about cooking")]).unwrap();
    index.update_document(Document::body("x", "new words about rust")).unwrap();
    let check = |index: &Index| {
        assert_eq!(index.lexical_search(&gyrfalcon::prepare("cooking").tokens, 10).unwrap().len(), 0);
        let hits = index.retrieve("rust", RetrieveOptions { rerank: false, limit: 5 }).unwrap();
        assert_eq!(hits.hits.len(), 1);
        assert_eq!(hits.hits[0].body, "new words about rust");
    };
    check(&index);
    drop(index);
    check(&Index::open(dir.path(), &IndexConfig::test_default()).unwrap());
}

#[test]
fn a_full_buffer_seals_one_segment_and_starts_a_new_log() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.write_buffer = 4;
    index.auto_merge = false;
    for doc in docs("d", 3) {
        index.index_documents(&[doc]).unwrap();
    }
    assert!(index.manifest().sealed_segments.is_empty());
    assert_eq!(wal_files(dir.path()), vec!["0.log"]);
    index.index_documents(&docs("e", 1)).unwrap();
    let manifest = index.manifest();
    assert_eq!(manifest.sealed_segments.len(), 1);
    assert_eq!(manifest.wal_generation, 1);
    assert_eq!(wal_files(dir.path()), vec!["1.log"]);
    assert_eq!(index.segment_chunk_counts(), vec![4]);
    assert_eq!(index.collection_doc_freq("safety"), 4);
}

#[test]
fn sealing_does_not_change_scores() {
    let mut index = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    index.auto_merge = false;
    index.index_documents(&docs("s", 40)).unwrap();
    index.write_buffer = 1000;
    for doc in docs("t", 30) {
        index.index_documents(&[doc]).unwrap();
    }
    let tokens = gyrfalcon::prepare("memory note t7 s3").tokens;
    let query: Vec<f32> = (0..384).map(|i| ((i * 31 % 17) as f32 - 8.0) / 8.0).collect();
    let lexical_before = index.lexical_search(&tokens, 20).unwrap();
    let dense_before = index.vector_search(&query, 20).unwrap();
    assert!(!index.flush().unwrap().sealed.is_empty());
    assert_eq!(index.lexical_search(&tokens, 20).unwrap(), lexical_before);
    let dense_after = index.vector_search(&query, 20).unwrap();
    assert_eq!(dense_after.ids, dense_before.ids);
    assert_eq!(dense_after.distances, dense_before.distances);
}

#[test]
fn a_torn_log_tail_loses_only_the_unfinished_write() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.index_documents(&docs("a", 2)).unwrap();
    drop(index);
    let log = dir.path().join("wal").join("0.log");
    let mut bytes = std::fs::read(&log).unwrap();
    bytes.extend_from_slice(&[200, 0, 0, 0, 1, 2, 3]);
    std::fs::write(&log, &bytes).unwrap();
    let mut index = Index::open(dir.path(), &IndexConfig::test_default()).unwrap();
    assert_eq!(index.collection_doc_freq("safety"), 2);
    index.index_documents(&docs("b", 1)).unwrap();
    drop(index);
    let index = Index::open(dir.path(), &IndexConfig::test_default()).unwrap();
    assert_eq!(index.collection_doc_freq("safety"), 3);
}

#[test]
fn a_bulk_write_seals_directly_beside_the_buffer() {
    let mut index = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    index.auto_merge = false;
    index.write_buffer = 10;
    index.index_documents(&docs("small", 2)).unwrap();
    index.index_documents(&docs("bulk", 12)).unwrap();
    assert_eq!(index.segment_chunk_counts(), vec![12]);
    assert_eq!(index.collection_doc_freq("safety"), 14);
    let hits = index.retrieve("small1", RetrieveOptions { rerank: false, limit: 1 }).unwrap();
    assert_eq!(hits.hits[0].doc_id, "small1");
}
