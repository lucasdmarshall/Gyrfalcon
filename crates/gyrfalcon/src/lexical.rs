//! The lexical leg: one Tantivy index per segment, scored with collection BM25 statistics.

use std::path::Path;
use std::sync::Arc;

use tantivy::collector::{Collector, SegmentCollector};
use tantivy::columnar::Column;
use tantivy::query::{Bm25StatisticsProvider, BooleanQuery, Occur, Query, TermQuery};
use tantivy::schema::{
    Field, IndexRecordOption, NumericOptions, Schema, TextFieldIndexing, TextOptions, Value, STORED, STRING,
};
use tantivy::tokenizer::{PreTokenizedString, Token as TantivyToken, TokenStream, Tokenizer};
use tantivy::{
    DocAddress, DocId, IndexReader, IndexWriter, ReloadPolicy, Score, SegmentOrdinal, SegmentReader, TantivyDocument,
    Term,
};

use crate::error::{Error, Result};
use crate::manifest::ANALYZER_ID;
use crate::prepare::{prepare, PreparedText, Token};
use crate::segment::ChunkRecord;
use crate::store::{TermMap, Tombstones};

const WRITER_HEAP_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredId {
    pub chunk_id: u64,
    pub score: f32,
}

/// The fields a stored chunk keeps in Tantivy. `body` and `prefix` let a merge rebuild the
/// chunk without the original document.
#[derive(Debug, Clone)]
pub struct StoredChunk {
    pub chunk_id: u64,
    pub doc_key: String,
    pub body: String,
    pub prefix: Option<String>,
    pub byte_start: u64,
    pub byte_end: u64,
}

fn lexical_error(err: impl std::fmt::Display) -> Error {
    Error::Lexical(err.to_string())
}

/// The registered `gyr-text-v1` tokenizer. It runs `prepare`, so plain-text writes and
/// `PreTokenizedString` writes produce the same postings.
#[derive(Clone, Default)]
pub struct GyrTextTokenizer;

pub struct GyrTokenStream {
    tokens: Vec<TantivyToken>,
    at: usize,
}

impl Tokenizer for GyrTextTokenizer {
    type TokenStream<'a> = GyrTokenStream;

    fn token_stream<'a>(&'a mut self, text: &'a str) -> GyrTokenStream {
        GyrTokenStream { tokens: to_tantivy_tokens(&prepare(text).tokens), at: 0 }
    }
}

impl TokenStream for GyrTokenStream {
    fn advance(&mut self) -> bool {
        self.at += 1;
        self.at <= self.tokens.len()
    }

    fn token(&self) -> &TantivyToken {
        &self.tokens[self.at - 1]
    }

    fn token_mut(&mut self) -> &mut TantivyToken {
        &mut self.tokens[self.at - 1]
    }
}

fn to_tantivy_tokens(tokens: &[Token]) -> Vec<TantivyToken> {
    tokens
        .iter()
        .map(|token| TantivyToken {
            offset_from: token.offset_from,
            offset_to: token.offset_to,
            position: token.position,
            text: token.text.clone(),
            position_length: 1,
        })
        .collect()
}

#[derive(Clone, Copy)]
struct Fields {
    text: Field,
    chunk_id: Field,
    doc_key: Field,
    body: Field,
    prefix: Field,
    byte_start: Field,
    byte_end: Field,
}

fn schema() -> (Schema, Fields) {
    let mut builder = Schema::builder();
    let indexing = TextFieldIndexing::default()
        .set_tokenizer(ANALYZER_ID)
        .set_index_option(IndexRecordOption::WithFreqsAndPositions);
    let text = builder.add_text_field("text", TextOptions::default().set_indexing_options(indexing));
    let chunk_id = builder.add_u64_field("chunk_id", NumericOptions::default().set_fast().set_stored().set_indexed());
    let doc_key = builder.add_text_field("doc_key", STRING | STORED);
    let body = builder.add_text_field("body", STORED);
    let prefix = builder.add_text_field("prefix", STORED);
    let byte_start = builder.add_u64_field("byte_start", STORED);
    let byte_end = builder.add_u64_field("byte_end", STORED);
    (builder.build(), Fields { text, chunk_id, doc_key, body, prefix, byte_start, byte_end })
}

fn register(index: &tantivy::Index) {
    index.tokenizers().register(ANALYZER_ID, GyrTextTokenizer);
}

/// Collection-wide BM25 statistics read from the published manifest and term map. Dead
/// chunks are included until a merge drops them. No segment dictionary is opened.
pub struct CollectionStats<'a> {
    pub num_docs: u64,
    pub num_tokens: u64,
    pub terms: &'a TermMap,
}

impl Bm25StatisticsProvider for CollectionStats<'_> {
    fn total_num_tokens(&self, _field: Field) -> tantivy::Result<u64> {
        Ok(self.num_tokens)
    }

    fn total_num_docs(&self) -> tantivy::Result<u64> {
        Ok(self.num_docs)
    }

    fn doc_freq(&self, term: &Term) -> tantivy::Result<u64> {
        Ok(term.value().as_str().map(|text| self.terms.get(text)).unwrap_or(0))
    }
}

/// Lexical operations the retrieve path needs. `TantivyLexical` is the v1 implementation.
pub trait LexicalIndex {
    fn add(&mut self, chunk: &ChunkRecord) -> Result<()>;
    fn search(
        &self,
        tokens: &[Token],
        limit: usize,
        stats: &CollectionStats,
        dead: &Arc<Tombstones>,
    ) -> Result<Vec<ScoredId>>;
    fn seal(&mut self) -> Result<()>;
}

pub struct TantivyLexical {
    index: tantivy::Index,
    fields: Fields,
    writer: Option<IndexWriter>,
    reader: Option<IndexReader>,
    /// `(chunk_id, address)` sorted by chunk id, filled when sealed.
    addresses: Vec<(u64, DocAddress)>,
}

impl TantivyLexical {
    pub fn create(dir: &Path) -> Result<TantivyLexical> {
        std::fs::create_dir_all(dir)?;
        let (schema, fields) = schema();
        let index = tantivy::Index::create_in_dir(dir, schema).map_err(lexical_error)?;
        register(&index);
        let writer = index.writer_with_num_threads(1, WRITER_HEAP_BYTES).map_err(lexical_error)?;
        Ok(TantivyLexical { index, fields, writer: Some(writer), reader: None, addresses: Vec::new() })
    }

    pub fn open(dir: &Path) -> Result<TantivyLexical> {
        let index = tantivy::Index::open_in_dir(dir).map_err(lexical_error)?;
        register(&index);
        let (_, fields) = schema();
        let mut lexical = TantivyLexical { index, fields, writer: None, reader: None, addresses: Vec::new() };
        lexical.open_reader()?;
        Ok(lexical)
    }

    fn open_reader(&mut self) -> Result<()> {
        let reader =
            self.index.reader_builder().reload_policy(ReloadPolicy::Manual).try_into().map_err(lexical_error)?;
        let searcher = reader.searcher();
        let mut addresses = Vec::new();
        for (ordinal, segment) in searcher.segment_readers().iter().enumerate() {
            let column = segment.fast_fields().u64("chunk_id").map_err(lexical_error)?;
            for doc in segment.doc_ids_alive() {
                if let Some(chunk_id) = column.first(doc) {
                    addresses.push((chunk_id, DocAddress::new(ordinal as SegmentOrdinal, doc)));
                }
            }
        }
        addresses.sort_unstable_by_key(|(id, _)| *id);
        self.addresses = addresses;
        self.reader = Some(reader);
        Ok(())
    }

    fn searcher(&self) -> Result<tantivy::Searcher> {
        self.reader.as_ref().map(|r| r.searcher()).ok_or_else(|| lexical_error("segment is not sealed"))
    }

    pub fn stored(&self, chunk_id: u64) -> Result<Option<StoredChunk>> {
        let Ok(at) = self.addresses.binary_search_by_key(&chunk_id, |(id, _)| *id) else {
            return Ok(None);
        };
        let searcher = self.searcher()?;
        let doc: TantivyDocument = searcher.doc(self.addresses[at].1).map_err(lexical_error)?;
        Ok(Some(self.decode(&doc)))
    }

    /// Every stored chunk, ascending by chunk id. A merge reads its inputs this way.
    pub fn all_stored(&self) -> Result<Vec<StoredChunk>> {
        let searcher = self.searcher()?;
        let mut out = Vec::with_capacity(self.addresses.len());
        for (_, address) in &self.addresses {
            let doc: TantivyDocument = searcher.doc(*address).map_err(lexical_error)?;
            out.push(self.decode(&doc));
        }
        Ok(out)
    }

    fn decode(&self, doc: &TantivyDocument) -> StoredChunk {
        let text = |field| doc.get_first(field).and_then(|v| v.as_str()).map(str::to_string);
        let number = |field| doc.get_first(field).and_then(|v| v.as_u64()).unwrap_or(0);
        StoredChunk {
            chunk_id: number(self.fields.chunk_id),
            doc_key: text(self.fields.doc_key).unwrap_or_default(),
            body: text(self.fields.body).unwrap_or_default(),
            prefix: text(self.fields.prefix),
            byte_start: number(self.fields.byte_start),
            byte_end: number(self.fields.byte_end),
        }
    }

    /// Writes a chunk as plain text through the registered tokenizer. Used by the tokenizer
    /// equivalence test only.
    #[doc(hidden)]
    pub fn add_plain_text(&mut self, chunk_id: u64, text: &str) -> Result<()> {
        let writer = self.writer.as_mut().ok_or_else(|| lexical_error("segment is sealed"))?;
        let mut doc = TantivyDocument::default();
        doc.add_text(self.fields.text, text);
        doc.add_u64(self.fields.chunk_id, chunk_id);
        writer.add_document(doc).map_err(lexical_error)?;
        Ok(())
    }

    /// Writes a chunk from an already prepared text as a `PreTokenizedString`.
    #[doc(hidden)]
    pub fn add_prepared(&mut self, chunk_id: u64, prepared: &PreparedText) -> Result<()> {
        let writer = self.writer.as_mut().ok_or_else(|| lexical_error("segment is sealed"))?;
        let mut doc = TantivyDocument::default();
        doc.add_pre_tokenized_text(
            self.fields.text,
            PreTokenizedString { text: prepared.embed_text.clone(), tokens: to_tantivy_tokens(&prepared.tokens) },
        );
        doc.add_u64(self.fields.chunk_id, chunk_id);
        writer.add_document(doc).map_err(lexical_error)?;
        Ok(())
    }

    /// The postings of `term`: `(chunk_id, term_freq, positions)` per document.
    #[doc(hidden)]
    pub fn postings(&self, term: &str) -> Result<Vec<(u64, u32, Vec<u32>)>> {
        use tantivy::postings::Postings;
        use tantivy::DocSet;
        let searcher = self.searcher()?;
        let mut out = Vec::new();
        for segment in searcher.segment_readers() {
            let column = segment.fast_fields().u64("chunk_id").map_err(lexical_error)?;
            let inverted = segment.inverted_index(self.fields.text).map_err(lexical_error)?;
            let term = Term::from_field_text(self.fields.text, term);
            if let Some(mut postings) =
                inverted.read_postings(&term, IndexRecordOption::WithFreqsAndPositions).map_err(lexical_error)?
            {
                while postings.doc() != tantivy::TERMINATED {
                    let mut positions = Vec::new();
                    postings.positions(&mut positions);
                    out.push((column.first(postings.doc()).unwrap_or(0), postings.term_freq(), positions));
                    postings.advance();
                }
            }
        }
        out.sort();
        Ok(out)
    }
}

impl LexicalIndex for TantivyLexical {
    fn add(&mut self, chunk: &ChunkRecord) -> Result<()> {
        let writer = self.writer.as_mut().ok_or_else(|| lexical_error("segment is sealed"))?;
        let mut doc = TantivyDocument::default();
        doc.add_pre_tokenized_text(
            self.fields.text,
            PreTokenizedString {
                text: chunk.prepared.embed_text.clone(),
                tokens: to_tantivy_tokens(&chunk.prepared.tokens),
            },
        );
        doc.add_u64(self.fields.chunk_id, chunk.chunk_id);
        doc.add_text(self.fields.doc_key, &chunk.doc_id);
        doc.add_text(self.fields.body, &chunk.body);
        if let Some(prefix) = &chunk.prefix {
            doc.add_text(self.fields.prefix, prefix);
        }
        doc.add_u64(self.fields.byte_start, chunk.byte_start);
        doc.add_u64(self.fields.byte_end, chunk.byte_end);
        writer.add_document(doc).map_err(lexical_error)?;
        Ok(())
    }

    fn search(
        &self,
        tokens: &[Token],
        limit: usize,
        stats: &CollectionStats,
        dead: &Arc<Tombstones>,
    ) -> Result<Vec<ScoredId>> {
        if tokens.is_empty() || limit == 0 || stats.num_docs == 0 {
            return Ok(Vec::new());
        }
        let mut terms: Vec<&str> = tokens.iter().map(|t| t.text.as_str()).collect();
        terms.sort_unstable();
        terms.dedup();
        let clauses: Vec<(Occur, Box<dyn Query>)> = terms
            .into_iter()
            .map(|text| {
                let term = Term::from_field_text(self.fields.text, text);
                (Occur::Should, Box::new(TermQuery::new(term, IndexRecordOption::WithFreqs)) as Box<dyn Query>)
            })
            .collect();
        let query = BooleanQuery::new(clauses);
        let collector = LiveTopK { limit, dead: dead.clone() };
        self.searcher()?.search_with_statistics_provider(&query, &collector, stats).map_err(lexical_error)
    }

    fn seal(&mut self) -> Result<()> {
        if let Some(mut writer) = self.writer.take() {
            writer.commit().map_err(lexical_error)?;
            writer.wait_merging_threads().map_err(lexical_error)?;
        }
        self.open_reader()
    }
}

/// Top-k collector over the `chunk_id` fast field. A tombstoned id never enters the heap,
/// so the result holds the best live hits.
struct LiveTopK {
    limit: usize,
    dead: Arc<Tombstones>,
}

struct LiveTopKSegment {
    limit: usize,
    dead: Arc<Tombstones>,
    column: Column<u64>,
    hits: Vec<ScoredId>,
}

fn by_score(a: &ScoredId, b: &ScoredId) -> std::cmp::Ordering {
    b.score.total_cmp(&a.score).then(a.chunk_id.cmp(&b.chunk_id))
}

fn top(mut hits: Vec<ScoredId>, limit: usize) -> Vec<ScoredId> {
    hits.sort_unstable_by(by_score);
    hits.truncate(limit);
    hits
}

impl Collector for LiveTopK {
    type Fruit = Vec<ScoredId>;
    type Child = LiveTopKSegment;

    fn for_segment(&self, _ordinal: SegmentOrdinal, segment: &SegmentReader) -> tantivy::Result<LiveTopKSegment> {
        Ok(LiveTopKSegment {
            limit: self.limit,
            dead: self.dead.clone(),
            column: segment.fast_fields().u64("chunk_id")?,
            hits: Vec::new(),
        })
    }

    fn requires_scoring(&self) -> bool {
        true
    }

    fn merge_fruits(&self, fruits: Vec<Vec<ScoredId>>) -> tantivy::Result<Vec<ScoredId>> {
        Ok(top(fruits.into_iter().flatten().collect(), self.limit))
    }
}

impl SegmentCollector for LiveTopKSegment {
    type Fruit = Vec<ScoredId>;

    fn collect(&mut self, doc: DocId, score: Score) {
        let Some(chunk_id) = self.column.first(doc) else { return };
        if self.dead.contains(chunk_id) {
            return;
        }
        self.hits.push(ScoredId { chunk_id, score });
        // Amortized pruning keeps the buffer at most twice the limit.
        if self.hits.len() >= self.limit.max(1) * 2 {
            let hits = std::mem::take(&mut self.hits);
            self.hits = top(hits, self.limit);
        }
    }

    fn harvest(self) -> Vec<ScoredId> {
        top(self.hits, self.limit)
    }
}
