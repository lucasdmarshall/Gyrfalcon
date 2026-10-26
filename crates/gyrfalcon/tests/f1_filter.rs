//! Metadata filters: documents carry `(key, value)` pairs, and a filtered search returns only
//! chunks of matching documents, from both legs, buffered or sealed, across reopen and merge.

use gyrfalcon::{Document, Filter, Index, IndexConfig, RetrieveOptions};

fn doc(id: &str, text: &str, metadata: &[(&str, &str)]) -> Document {
    Document::body(id, text).with_metadata(metadata.iter().map(|(k, v)| (k.to_string(), v.to_string())))
}

fn ids(index: &Index, query: &str, filter: &Filter) -> Vec<String> {
    let hits = index.retrieve_filtered(query, RetrieveOptions { rerank: false, limit: 50 }, Some(filter)).unwrap();
    let mut ids: Vec<String> = hits.hits.iter().map(|h| h.doc_id.clone()).collect();
    ids.sort();
    ids.dedup();
    ids
}

fn corpus() -> Vec<Document> {
    vec![
        doc("a", "falcons hunt over the tundra", &[("lang", "en"), ("tag", "birds")]),
        doc("b", "falcons nest on cliffs", &[("lang", "en"), ("tag", "birds"), ("tag", "nesting")]),
        doc("c", "falcons sont des rapaces", &[("lang", "fr"), ("tag", "birds")]),
        doc("d", "falcon heavy rocket launch", &[("lang", "en"), ("tag", "space")]),
        doc("e", "falcons without metadata", &[]),
    ]
}

#[test]
fn filters_combine_clauses_with_and_values_with_or() {
    for write_buffer in [0, 4096] {
        let mut index = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
        index.write_buffer = write_buffer;
        index.index_documents(&corpus()).unwrap();
        let everything = ids(&index, "falcons", &Filter::new());
        assert_eq!(everything, ["a", "b", "c", "d", "e"], "buffer {write_buffer}");
        assert_eq!(ids(&index, "falcons", &Filter::new().eq("lang", "en")), ["a", "b", "d"]);
        assert_eq!(ids(&index, "falcons", &Filter::new().eq("lang", "en").eq("tag", "birds")), ["a", "b"]);
        assert_eq!(ids(&index, "falcons", &Filter::new().any("tag", ["nesting", "space"])), ["b", "d"]);
        assert_eq!(ids(&index, "falcons", &Filter::new().eq("lang", "de")), Vec::<String>::new());
        assert_eq!(ids(&index, "falcons", &Filter::new().eq("tag", "Birds")), Vec::<String>::new(), "values are exact");
        let hit = &index
            .retrieve_filtered(
                "cliffs",
                RetrieveOptions { rerank: false, limit: 1 },
                Some(&Filter::new().eq("tag", "nesting")),
            )
            .unwrap()
            .hits[0];
        assert_eq!(hit.doc_id, "b");
        let mut metadata = hit.metadata.clone();
        metadata.sort();
        assert_eq!(
            metadata,
            [("lang".into(), "en".into()), ("tag".into(), "birds".into()), ("tag".into(), "nesting".into())]
        );
    }
}

#[test]
fn metadata_survives_reopen_update_delete_and_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.index_documents(&corpus()).unwrap();
    drop(index);
    // The buffer was replayed from the log.
    let mut index = Index::open(dir.path(), &IndexConfig::test_default()).unwrap();
    assert_eq!(ids(&index, "falcons", &Filter::new().eq("lang", "fr")), ["c"]);
    index.update_document(doc("c", "falcons en français", &[("lang", "en")])).unwrap();
    index.delete_document("a").unwrap();
    assert_eq!(ids(&index, "falcons", &Filter::new().eq("lang", "fr")), Vec::<String>::new());
    assert_eq!(ids(&index, "falcons", &Filter::new().eq("lang", "en")), ["b", "c", "d"]);
    index.flush().unwrap();
    index.compact_all().unwrap();
    assert_eq!(ids(&index, "falcons", &Filter::new().eq("lang", "en")), ["b", "c", "d"]);
    assert_eq!(ids(&index, "falcons", &Filter::new().eq("tag", "birds")), ["b"]);
}

#[test]
fn metadata_does_not_change_scores() {
    let mut plain = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    let mut tagged = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    let docs = corpus();
    plain.index_documents(&docs.iter().map(|d| Document::body(&d.doc_id, &d.text)).collect::<Vec<_>>()).unwrap();
    tagged.index_documents(&docs).unwrap();
    let tokens = gyrfalcon::prepare("falcons nest tundra").tokens;
    assert_eq!(plain.lexical_search(&tokens, 10).unwrap(), tagged.lexical_search(&tokens, 10).unwrap());
}

#[test]
fn a_selective_filter_finds_every_match_through_both_legs() {
    let mut index = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    index.write_buffer = 0;
    let docs: Vec<Document> = (0..3000)
        .map(|i| {
            let shard = if i % 300 == 7 { "rare" } else { "common" };
            doc(&format!("d{i}"), &format!("record {i} about topic {} and more", i % 17), &[("shard", shard)])
        })
        .collect();
    index.index_documents(&docs).unwrap();
    let rare: Vec<String> = {
        let mut rare: Vec<String> = (0..3000).filter(|i| i % 300 == 7).map(|i| format!("d{i}")).collect();
        rare.sort();
        rare
    };
    // A query no rare document shares a word with: only the dense leg can find them.
    assert_eq!(ids(&index, "unrelated words entirely", &Filter::new().eq("shard", "rare")), rare);
    assert_eq!(ids(&index, "record topic", &Filter::new().eq("shard", "rare")), rare);
}
