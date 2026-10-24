//! Tantivy, the previous lexical engine, kept as the reference the owned index must match.
//! Test-only: Tantivy is a dev-dependency.

use std::path::Path;

use tantivy::collector::TopDocs;
use tantivy::query::{Bm25StatisticsProvider, BooleanQuery, Occur, Query, TermQuery};
use tantivy::schema::{Field, IndexRecordOption, NumericOptions, Schema, TextFieldIndexing, TextOptions, Value};
use tantivy::tokenizer::{PreTokenizedString, Token as TantivyToken};
use tantivy::{IndexReader, TantivyDocument, Term};

use super::*;
use crate::prepare::{prepare, prepare_with_prefix, PreparedText};
use crate::segment::ChunkRecord;

struct Stats<'a>(&'a CollectionStats<'a>);

impl Bm25StatisticsProvider for Stats<'_> {
    fn total_num_tokens(&self, _field: Field) -> tantivy::Result<u64> {
        Ok(self.0.num_tokens)
    }

    fn total_num_docs(&self) -> tantivy::Result<u64> {
        Ok(self.0.num_docs)
    }

    fn doc_freq(&self, term: &Term) -> tantivy::Result<u64> {
        Ok(term.value().as_str().map(|text| self.0.terms.get(text)).unwrap_or(0))
    }
}

/// One Tantivy index over `chunks`, built the way v1 built it.
struct Tantivy {
    reader: IndexReader,
    text: Field,
    chunk_id: Field,
}

impl Tantivy {
    fn build(dir: &Path, chunks: &[ChunkRecord]) -> Tantivy {
        let mut builder = Schema::builder();
        let indexing = TextFieldIndexing::default()
            .set_tokenizer("gyr-text-v1")
            .set_index_option(IndexRecordOption::WithFreqsAndPositions);
        let text = builder.add_text_field("text", TextOptions::default().set_indexing_options(indexing));
        let chunk_id = builder.add_u64_field("chunk_id", NumericOptions::default().set_fast().set_stored());
        std::fs::create_dir_all(dir).unwrap();
        let index = tantivy::Index::create_in_dir(dir, builder.build()).unwrap();
        // Pre-tokenized text bypasses the tokenizer, but the field must name a registered one.
        index.tokenizers().register("gyr-text-v1", tantivy::tokenizer::WhitespaceTokenizer::default());
        let mut writer: tantivy::IndexWriter = index.writer_with_num_threads(1, 64 << 20).unwrap();
        for chunk in chunks {
            let tokens = chunk
                .prepared
                .tokens
                .iter()
                .map(|t| TantivyToken {
                    offset_from: t.offset_from,
                    offset_to: t.offset_to,
                    position: t.position,
                    text: t.text.clone(),
                    position_length: 1,
                })
                .collect();
            let mut doc = TantivyDocument::default();
            doc.add_pre_tokenized_text(text, PreTokenizedString { text: chunk.prepared.embed_text.clone(), tokens });
            doc.add_u64(chunk_id, chunk.chunk_id);
            writer.add_document(doc).unwrap();
        }
        writer.commit().unwrap();
        writer.wait_merging_threads().unwrap();
        Tantivy { reader: index.reader().unwrap(), text, chunk_id }
    }

    /// Every matching live chunk, best first, ties by chunk id.
    fn search(&self, tokens: &[Token], limit: usize, stats: &CollectionStats, dead: &Tombstones) -> Vec<ScoredId> {
        let mut terms: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        terms.sort_unstable();
        terms.dedup();
        let clauses: Vec<(Occur, Box<dyn Query>)> = terms
            .into_iter()
            .map(|t| {
                let term = Term::from_field_text(self.text, t);
                (Occur::Should, Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs)) as Box<dyn Query>)
            })
            .collect();
        let searcher = self.reader.searcher();
        let everything = TopDocs::with_limit(searcher.num_docs().max(1) as usize).order_by_score();
        let found =
            searcher.search_with_statistics_provider(&BooleanQuery::new(clauses), &everything, &Stats(stats)).unwrap();
        let mut hits: Vec<ScoredId> = found
            .into_iter()
            .map(|(score, address)| {
                let doc: TantivyDocument = searcher.doc(address).unwrap();
                ScoredId { chunk_id: doc.get_first(self.chunk_id).and_then(|v| v.as_u64()).unwrap(), score }
            })
            .filter(|hit| !dead.contains(hit.chunk_id))
            .collect();
        hits.sort_by_key(|hit| Reverse(Ranked(*hit)));
        hits.truncate(limit);
        hits
    }
}

fn record(chunk_id: u64, prepared: PreparedText, body: &str, prefix: Option<&str>) -> ChunkRecord {
    ChunkRecord {
        chunk_id,
        doc_id: format!("doc-{chunk_id}"),
        byte_start: chunk_id * 3,
        byte_end: chunk_id * 3 + body.len() as u64,
        body: body.to_string(),
        prefix: prefix.map(str::to_string),
        prepared,
        vector: Vec::new(),
    }
}

fn plain(chunk_id: u64, text: &str) -> ChunkRecord {
    record(chunk_id, prepare(text), text, None)
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }

    /// A Zipf-like word: low ranks are far more common.
    fn word(&mut self, vocabulary: u64) -> String {
        let rank = (self.below(vocabulary) * self.below(vocabulary)) / vocabulary;
        format!("w{rank}")
    }
}

fn assert_same_ranking(ours: &[ScoredId], theirs: &[ScoredId], context: &str) {
    assert_eq!(ours.len(), theirs.len(), "{context}: hit count");
    for (rank, (a, b)) in ours.iter().zip(theirs).enumerate() {
        let tolerance = 1e-6 * a.score.abs().max(1.0);
        assert!((a.score - b.score).abs() <= tolerance, "{context}: rank {rank} score {} vs {}", a.score, b.score);
        if a.chunk_id != b.chunk_id {
            // Only a float-rounding tie may swap two hits.
            let other = theirs.iter().find(|h| h.chunk_id == a.chunk_id).expect("hit missing from Tantivy");
            assert!((other.score - a.score).abs() <= tolerance, "{context}: rank {rank} differs");
        }
    }
}

#[test]
fn norm_table_is_tantivys() {
    use tantivy::fieldnorm::FieldNormReader;
    for id in 0..=255u8 {
        assert_eq!(NORM_TABLE[id as usize], FieldNormReader::id_to_fieldnorm(id));
    }
    for length in (0..5000).chain([u32::MAX - 1, u32::MAX]) {
        assert_eq!(fieldnorm_to_id(length), FieldNormReader::fieldnorm_to_id(length), "{length}");
    }
}

#[test]
fn scores_match_tantivy_on_random_collections() {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for round in 0..6 {
        let num_docs = [1, 7, 130, 600, 2000, 300][round];
        let vocabulary = [3, 20, 200, 1000, 5000, 40][round];
        let chunks: Vec<ChunkRecord> = (0..num_docs)
            .map(|i| {
                // Lengths cross the exact-norm limit (40) and reach thousands of tokens.
                let len = match rng.below(10) {
                    0 => 0,
                    1..=6 => 1 + rng.below(60),
                    7 | 8 => 60 + rng.below(400),
                    _ => 400 + rng.below(3000),
                };
                let text: Vec<String> = (0..len).map(|_| rng.word(vocabulary)).collect();
                plain(i * 2 + round as u64, &text.join(" "))
            })
            .collect();
        let dir = tempfile::tempdir().unwrap();
        let built = write(&dir.path().join("lexical"), &chunks).unwrap();
        let ours = LexicalSegment::open(&dir.path().join("lexical")).unwrap();
        let theirs = Tantivy::build(&dir.path().join("tantivy"), &chunks);
        let ids: Vec<u64> = chunks.iter().map(|c| c.chunk_id).collect();

        // Collection statistics both as this segment alone and as part of a larger collection.
        let mut wider = built.terms.clone();
        for value in wider.0.values_mut() {
            *value = *value * 3 + 1;
        }
        let collections =
            [(num_docs, built.num_tokens, &built.terms), (num_docs * 3 + 5, built.num_tokens * 2 + 17, &wider)];
        for (collection, (num_docs, num_tokens, terms)) in collections.into_iter().enumerate() {
            let stats = CollectionStats { num_docs, num_tokens: num_tokens.max(1), terms };
            for query in 0..40 {
                let len = 1 + rng.below(if query % 5 == 0 { 40 } else { 6 });
                let mut text: Vec<String> = (0..len).map(|_| rng.word(vocabulary + 5)).collect();
                text.push(text[0].clone());
                let tokens = prepare(&text.join(" ")).tokens;
                let dead_every = [0, 2, 3, 7][query % 4];
                let dead = Tombstones::from_ids(
                    ids.iter().copied().filter(|id| dead_every > 0 && id % dead_every == 0).collect(),
                );
                for limit in [1, 10, 100] {
                    let context = format!("round {round} collection {collection} query {query} limit {limit}");
                    let got = ours.search(&tokens, limit, &stats, &dead, &ids).unwrap();
                    assert_same_ranking(&got, &theirs.search(&tokens, limit, &stats, &dead), &context);
                }
            }
        }
    }
}

#[test]
fn postings_follow_the_prepared_tokens() {
    let fixtures = ["Memory safety in Rust", "မြန်မာ စာ ကို ဖတ်ပါ", "error E0308", "   \t\n ", "😀", "rust rust RUST"];
    let chunks: Vec<ChunkRecord> = fixtures.iter().enumerate().map(|(i, t)| plain(i as u64 * 10, t)).collect();
    let dir = tempfile::tempdir().unwrap();
    let built = write(dir.path(), &chunks).unwrap();
    let segment = LexicalSegment::open(dir.path()).unwrap();
    assert_eq!(segment.postings("memory"), vec![(0, 1)]);
    assert_eq!(segment.postings("e0308"), vec![(2, 1)]);
    assert_eq!(segment.postings("rust"), vec![(0, 1), (5, 3)]);
    assert_eq!(segment.postings("မြန်မာ"), vec![(1, 1)]);
    for absent in ["Memory", "E0308", "😀", " ", "", "zzz"] {
        assert!(segment.postings(absent).is_empty(), "{absent:?}");
    }
    assert_eq!(built.terms.get("rust"), 2);
    assert_eq!(built.num_tokens, 4 + 4 + 2 + 3);
    assert_eq!(&segment.norms[8..], &[4, 4, 2, 0, 0, 3]);
}

#[test]
fn stored_chunks_roundtrip_across_blocks() {
    let long = "ü".repeat(20_000);
    let bodies = ["short body", "", long.as_str(), "with prefix", "last"];
    let chunks: Vec<ChunkRecord> = (0..300)
        .map(|i| {
            let body = bodies[i % bodies.len()];
            let prefix = (i % 5 == 3).then_some("Context: a prefix.");
            let prepared = match prefix {
                Some(p) => prepare_with_prefix(p, body),
                None => prepare(body),
            };
            record(1000 + i as u64, prepared, body, prefix)
        })
        .collect();
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), &chunks).unwrap();
    let segment = LexicalSegment::open(dir.path()).unwrap();
    let expected: Vec<StoredChunk> = chunks
        .iter()
        .map(|c| StoredChunk {
            chunk_id: c.chunk_id,
            doc_key: c.doc_id.clone(),
            body: c.body.clone(),
            prefix: c.prefix.clone(),
            byte_start: c.byte_start,
            byte_end: c.byte_end,
        })
        .collect();
    assert_eq!(segment.all_stored().unwrap(), expected);
    for (doc, chunk) in expected.iter().enumerate() {
        assert_eq!(&segment.stored(doc as u32).unwrap(), chunk);
    }
    assert!(segment.stored(300).is_err());
}

#[test]
fn an_empty_segment_opens_and_finds_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let built = write(dir.path(), &[]).unwrap();
    assert!(built.terms.0.is_empty());
    let segment = LexicalSegment::open(dir.path()).unwrap();
    let terms = TermMap::default();
    let stats = CollectionStats { num_docs: 5, num_tokens: 9, terms: &terms };
    let hits = segment.search(&prepare("anything").tokens, 10, &stats, &Tombstones::from_ids(vec![]), &[]).unwrap();
    assert!(hits.is_empty());
    assert!(segment.all_stored().unwrap().is_empty());
}

#[test]
fn damaged_files_are_errors_not_panics() {
    let chunks: Vec<ChunkRecord> = (0..400)
        .map(|i| plain(i, &format!("alpha beta gamma{} delta {}", i % 7, "x ".repeat(i as usize % 50))))
        .collect();
    let dir = tempfile::tempdir().unwrap();
    let built = write(dir.path(), &chunks).unwrap();
    let ids: Vec<u64> = (0..400).collect();
    let stats = CollectionStats { num_docs: 400, num_tokens: built.num_tokens, terms: &built.terms };
    let dead = Tombstones::from_ids(vec![]);
    let query = prepare("alpha gamma3 x delta").tokens;
    for file in ["terms.dict", "postings.bin", "norms.bin", "docs.bin"] {
        let path = dir.path().join(file);
        let original = std::fs::read(&path).unwrap();
        let mut rng = Rng(original.len() as u64 | 1);
        for trial in 0..60 {
            let mut damaged = original.clone();
            if trial % 3 == 0 {
                damaged.truncate(rng.below(original.len() as u64) as usize);
            } else {
                for _ in 0..1 + trial % 4 {
                    let at = 8 + rng.below(original.len() as u64 - 8) as usize;
                    damaged[at] ^= 1 << rng.below(8);
                }
            }
            std::fs::write(&path, &damaged).unwrap();
            // Opening, searching, and reading back may fail, but must not panic.
            if let Ok(segment) = LexicalSegment::open(dir.path()) {
                let _ = segment.search(&query, 10, &stats, &dead, &ids);
                let _ = segment.all_stored();
                let _ = segment.stored(399);
            }
        }
        std::fs::write(&path, &original).unwrap();
    }
}
