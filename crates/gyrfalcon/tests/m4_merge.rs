use gyrfalcon::{Document, Index, IndexConfig, RetrieveOptions};

fn two_rust_docs() -> Index {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.index_documents(&[Document::body("a", "rust safety"), Document::body("b", "rust cooking")]).unwrap();
    index.leak_tempdir(dir);
    index
}

fn eight_small_segments() -> Index {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.pause_merge_after_copy = true;
    for i in 0..8 {
        let doc_id = match i {
            0 => "keep".to_string(),
            1 => "doomed".to_string(),
            _ => format!("d{i}"),
        };
        let text = "word ".repeat(10);
        index.index_documents(&[Document::body(doc_id, text)]).unwrap();
    }
    index.leak_tempdir(dir);
    index
}

fn eight_tens_and_one_thousand() -> Index {
    let mut index = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    index.seal_n_chunks(1_000, "word").unwrap();
    for _ in 0..8 {
        index.seal_n_chunks(10, "word").unwrap();
    }
    index
}

fn one_segment_with_stored_count(n: usize) -> Index {
    let mut index = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    index.seal_n_chunks(n, "shared").unwrap();
    index
}

#[test]
fn delete_hides_the_chunk_and_keeps_doc_freq() {
    let mut index = two_rust_docs();
    let before = index.collection_doc_freq("rust");
    let report = index.delete_document("a").unwrap();
    assert_eq!(report.deleted_chunks, 1);
    assert_eq!(index.collection_doc_freq("rust"), before);
    assert_eq!(index.collection_doc_freq("safety"), 1);
    assert_eq!(index.manifest().bm25_num_docs, 2);
    let hits = index.retrieve("rust", RetrieveOptions { rerank: false, limit: 5 }).unwrap();
    assert!(!hits.chunk_ids.contains(&0));
    assert_eq!(hits.chunk_ids, vec![1]);
    assert!(index.tombstone_contains(0));
}

#[test]
fn unknown_delete_publishes_nothing() {
    let mut index = two_rust_docs();
    let epoch = index.manifest().epoch;
    let report = index.delete_document("missing").unwrap();
    assert_eq!(report.deleted_chunks, 0);
    assert_eq!(index.manifest().epoch, epoch);
}

#[test]
fn update_tombstones_the_old_chunks_in_the_same_rename() {
    let mut index = two_rust_docs();
    let epoch = index.manifest().epoch;
    let report = index.update_document(Document::body("a", "rust lifetimes")).unwrap();
    assert_eq!(report.chunk_ids, vec![2]);
    assert_eq!(index.manifest().epoch, epoch + 1);
    assert!(index.tombstone_contains(0));
    assert_eq!(index.collection_doc_freq("rust"), 3);
    let hits = index.retrieve("lifetimes", RetrieveOptions { rerank: false, limit: 5 }).unwrap();
    assert_eq!(hits.chunk_ids[0], 2);
    assert!(!hits.chunk_ids.contains(&0));
    let inserted = index.update_document(Document::body("new", "fresh text")).unwrap();
    assert_eq!(inserted.chunk_ids, vec![3]);
    assert_eq!(index.chunk_id_for_doc("new"), 3);
}

#[test]
fn crashed_update_keeps_the_old_document_retrievable() {
    let mut index = two_rust_docs();
    index.fail_before_rename = true;
    assert!(index.update_document(Document::body("a", "rust lifetimes")).is_err());
    let hits = index.retrieve("safety", RetrieveOptions { rerank: false, limit: 5 }).unwrap();
    assert_eq!(hits.chunk_ids[0], 0);
    assert!(!index.tombstone_contains(0));
}

#[test]
fn merge_of_the_dead_chunk_segment_drops_doc_freq_by_one() {
    let mut index = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    for i in 0..7 {
        index.index_documents(&[Document::body(format!("d{i}"), "common words")]).unwrap();
    }
    // Drain the merge request the last call queued, so no pass sees the eighth segment before
    // the delete below.
    index.wait_until_merge_idle();
    index.auto_merge = false;
    index.index_documents(&[Document::body("rare", "common unicorn")]).unwrap();
    index.delete_document("rare").unwrap();
    assert_eq!(index.collection_doc_freq("unicorn"), 1);
    assert_eq!(index.collection_doc_freq("common"), 8);
    index.auto_merge = true;
    index.wait_until_merge_idle();
    assert_eq!(index.segment_chunk_counts(), vec![7]);
    assert_eq!(index.collection_doc_freq("unicorn"), 0);
    assert_eq!(index.collection_doc_freq("common"), 7);
    assert_eq!(index.manifest().bm25_num_docs, 7);
    // A tier merge leaves the bitmap alone.
    assert!(index.tombstone_contains(7));
    assert!(index.merge_error().is_none());
}

#[test]
fn index_documents_returns_before_the_paused_merge_swaps() {
    let mut index = eight_small_segments();
    index.wait_until_merge_copied();
    assert_eq!(index.segment_chunk_counts().len(), 8);
    index.wait_until_merge_idle();
    assert_eq!(index.segment_chunk_counts(), vec![8]);
}

#[test]
fn delete_during_merge_stays_dead_and_survivor_ids_do_not_change() {
    let mut index = eight_small_segments();
    let survivor = index.chunk_id_for_doc("keep");
    index.pause_merge_after_copy = true;
    index.index_documents(&[Document::body("ninth", "word ".repeat(10))]).unwrap();
    index.wait_until_merge_copied();
    let doomed = index.chunk_id_for_doc("doomed");
    index.delete_document("doomed").unwrap();
    index.pause_merge_after_copy = false;
    index.wait_until_merge_idle();
    assert_eq!(index.segment_chunk_counts().iter().sum::<u64>(), 9);
    assert_eq!(index.segment_chunk_counts().len(), 2);
    let hits = index.retrieve("word", RetrieveOptions { rerank: false, limit: 50 }).unwrap();
    assert!(!hits.chunk_ids.contains(&doomed));
    assert_eq!(hits.chunk_ids.len(), 8);
    assert!(index.tombstone_contains(doomed));
    assert_eq!(index.chunk_id_for_doc("keep"), survivor);
}

#[test]
fn tier_merge_leaves_the_thousand_chunk_segment() {
    let mut index = eight_tens_and_one_thousand();
    index.wait_until_merge_idle();
    let sizes = index.segment_chunk_counts();
    assert!(sizes.contains(&1_000));
    assert_eq!(sizes.iter().filter(|n| **n < 1_000).count(), 1);
    assert!(sizes.contains(&80));
}

#[test]
fn select_tier_never_takes_a_full_segment() {
    let full: Vec<(String, u64)> = (0..8).map(|i| (format!("{i}"), 100_000)).collect();
    assert_eq!(gyrfalcon::select_tier(&full), None);
    let mixed: Vec<(String, u64)> = (0..8).map(|i| (format!("{i}"), 12_500 + i)).collect();
    assert_eq!(gyrfalcon::select_tier(&mixed), None, "eight in tier 13 sum above 100,000");
    let small: Vec<(String, u64)> = (0..9).map(|i| (format!("{i}"), 1)).collect();
    assert_eq!(gyrfalcon::select_tier(&small).unwrap().len(), 8);
}

#[test]
fn dead_ratio_above_20_percent_rewrites_one_large_segment() {
    let mut index = one_segment_with_stored_count(100_000);
    index.testing_tombstone_first_n(21_000);
    let before = index.collection_doc_freq("shared");
    assert_eq!(before, 100_000);
    let survivor = index.chunk_id_for_doc("shared-50000");
    index.wait_until_merge_idle();
    assert_eq!(index.segment_chunk_counts(), vec![79_000]);
    assert_eq!(index.collection_doc_freq("shared"), before - 21_000);
    assert_eq!(index.manifest().bm25_num_docs, 79_000);
    assert_eq!(index.tombstone_len(), 0);
    assert_eq!(index.chunk_id_for_doc("shared-50000"), survivor);
    let mut exact = one_segment_with_stored_count(100_000);
    exact.testing_tombstone_first_n(20_000);
    exact.wait_until_merge_idle();
    assert_eq!(exact.segment_chunk_counts(), vec![100_000]);
    assert_eq!(gyrfalcon::dead_ratio_target(&[("a".into(), 100_000, 20_000)]), None);
    assert_eq!(gyrfalcon::dead_ratio_target(&[("a".into(), 99_999, 90_000)]), None);
}

#[test]
fn old_segment_remains_until_the_reader_releases() {
    let mut index = eight_small_segments();
    let guard = index.acquire();
    let old = index.segment_dirs();
    index.wait_until_merge_idle();
    assert_eq!(index.segment_chunk_counts(), vec![8]);
    assert!(old[0].exists());
    index.reap_now();
    assert!(old[0].exists());
    drop(guard);
    index.reap_now();
    assert!(!old[0].exists());
}

#[test]
fn busy_delete_is_retried() {
    let mut index = eight_small_segments();
    index.deleter_fail_times = 1;
    index.wait_until_merge_idle();
    index.reap_now();
    let old = index.retired_dirs();
    assert_eq!(old.len(), 1);
    assert!(old[0].exists());
    index.reap_now();
    assert!(!old[0].exists());
    assert!(index.retired_dirs().is_empty());
}

#[test]
fn compact_all_rewrites_everything_and_clears_tombstones() {
    let mut index = two_rust_docs();
    index.auto_merge = false;
    index.index_documents(&[Document::body("c", "rust gardening")]).unwrap();
    index.delete_document("a").unwrap();
    let ids = (index.chunk_id_for_doc("b"), index.chunk_id_for_doc("c"));
    assert!(index.compact_all().unwrap());
    assert_eq!(index.segment_chunk_counts(), vec![2]);
    assert_eq!(index.tombstone_len(), 0);
    assert_eq!(index.collection_doc_freq("rust"), 2);
    assert_eq!((index.chunk_id_for_doc("b"), index.chunk_id_for_doc("c")), ids);
    let hits = index.retrieve("gardening", RetrieveOptions { rerank: false, limit: 1 }).unwrap();
    assert_eq!(hits.hits[0].doc_id, "c");
}

#[test]
fn thirty_three_segments_warn() {
    let mut index = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    index.auto_merge = false;
    for i in 0..33 {
        index.index_documents(&[Document::body(format!("d{i}"), "word")]).unwrap();
    }
    assert_eq!(index.last_report().warning.as_deref(), Some("segment_count"));
    let hits = index.retrieve("word", RetrieveOptions { rerank: false, limit: 5 }).unwrap();
    assert_eq!(hits.warning.as_deref(), Some("segment_count"));
}
