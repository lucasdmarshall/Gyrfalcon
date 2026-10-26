//! Metadata filters.
//!
//! A document's `(key, value)` metadata pairs are indexed with its chunks as filter terms,
//! `\u{1}key\u{1}value`, which no analyzer token can equal (the analyzer drops control
//! characters). A filter is a conjunction of clauses, each matching any of its values; its
//! chunks come from the filter terms' postings in every segment and buffer batch, and a
//! filtered search excludes every other chunk in both legs.

use crate::error::Result;
use crate::index::Snapshot;

/// Every clause must hold; a clause holds when the chunk's document has the key with any of
/// the clause's values.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Filter {
    clauses: Vec<(String, Vec<String>)>,
}

impl Filter {
    pub fn new() -> Filter {
        Filter::default()
    }

    /// Adds the clause `key == value`.
    pub fn eq(self, key: impl Into<String>, value: impl Into<String>) -> Filter {
        self.any(key, [value])
    }

    /// Adds the clause `key` in `values`.
    pub fn any<V: Into<String>>(mut self, key: impl Into<String>, values: impl IntoIterator<Item = V>) -> Filter {
        self.clauses.push((key.into(), values.into_iter().map(Into::into).collect()));
        self
    }

    pub fn clauses(&self) -> &[(String, Vec<String>)] {
        &self.clauses
    }

    pub fn is_empty(&self) -> bool {
        self.clauses.is_empty()
    }
}

/// The indexed term of a metadata pair.
pub(crate) fn filter_term(key: &str, value: &str) -> String {
    format!("\u{1}{key}\u{1}{value}")
}

impl Snapshot {
    /// The chunk ids, ascending, whose documents match `filter`, dead ones included.
    pub(crate) fn matching(&self, filter: &Filter) -> Result<Vec<u64>> {
        let mut result: Option<Vec<u64>> = None;
        for (key, values) in filter.clauses() {
            let mut clause = Vec::new();
            for value in values {
                let term = filter_term(key, value);
                for segment in &self.segments {
                    clause.extend(segment.lexical.docs_with(&term)?.into_iter().map(|doc| segment.ids[doc as usize]));
                }
                for batch in &self.pending.batches {
                    clause.extend(batch.chunks_with(&term));
                }
            }
            clause.sort_unstable();
            clause.dedup();
            result = Some(match result {
                None => clause,
                Some(previous) => previous.into_iter().filter(|id| clause.binary_search(id).is_ok()).collect(),
            });
        }
        Ok(result.unwrap_or_default())
    }
}
