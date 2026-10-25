//! Delete and update. The collection tombstone bitmap is the only delete record; sealed
//! segment files and the BM25 sums are not rewritten.

use std::collections::HashSet;
use std::sync::Arc;

use crate::error::Result;
use crate::index::{Change, Document, Index, IndexReport};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DeleteReport {
    pub deleted_chunks: usize,
    /// Epoch of the published manifest, or the unchanged one when nothing was deleted.
    pub epoch: u64,
}

impl Index {
    /// Tombstones every live chunk of `doc_id`. An unknown id publishes nothing. BM25
    /// generation and totals stay as they are.
    pub fn delete_document(&mut self, doc_id: &str) -> Result<DeleteReport> {
        let ids: Vec<u64> = {
            let snapshot = self.shared.snapshot();
            snapshot.chunks_for_doc(doc_id)
        };
        self.delete_chunks(&ids)
    }

    /// Tombstones single chunks.
    pub fn delete_chunks(&mut self, ids: &[u64]) -> Result<DeleteReport> {
        let report = self.tombstone_chunks(ids)?;
        if report.deleted_chunks > 0 {
            self.after_publish();
            self.schedule_merge();
        }
        Ok(report)
    }

    fn tombstone_chunks(&mut self, ids: &[u64]) -> Result<DeleteReport> {
        let shared = self.shared.clone();
        let mut state = shared.lock();
        // Only live stored chunks are tombstoned; unknown and already dead ids are skipped.
        let mut doomed: Vec<u64> = ids
            .iter()
            .copied()
            .filter(|id| !state.tombstones.contains(*id) && state.segment_of(*id).is_some())
            .collect::<HashSet<u64>>()
            .into_iter()
            .collect();
        doomed.sort_unstable();
        if doomed.is_empty() {
            return Ok(DeleteReport { deleted_chunks: 0, epoch: state.manifest.epoch });
        }
        let change = Change { tombstones: Some(state.tombstones.with(&doomed)), ..Change::default() };
        let next = shared.publish(&state, change, self.fail_before_rename)?;
        *state = Arc::new(next);
        Ok(DeleteReport { deleted_chunks: doomed.len(), epoch: state.manifest.epoch })
    }

    /// Seals the new text, then in one rename adds the segment, tombstones the previous chunk
    /// ids. An unknown id is an insert. Schedules the
    /// merger and returns without waiting for it.
    pub fn update_document(&mut self, doc: Document) -> Result<IndexReport> {
        let mut records = self.chunk_records(std::slice::from_ref(&doc))?;
        self.embed_records(&mut records)?;
        let shared = self.shared.clone();
        let mut state = shared.lock();
        let previous = state.chunks_for_doc(&doc.doc_id);
        let report = self.seal_records_replacing(&mut state, records, &previous)?;
        drop(state);
        self.after_publish();
        self.schedule_merge();
        self.last_report = report.clone();
        Ok(report)
    }

    /// Writes the bitmap only. Leaves `bm25_num_docs` and the merger alone.
    #[doc(hidden)]
    pub fn testing_tombstone_chunk(&mut self, chunk_id: u64) {
        self.tombstone_chunks(&[chunk_id]).expect("tombstone chunk");
    }

    /// Tombstones the `n` lowest stored chunk ids without scheduling the merger.
    #[doc(hidden)]
    pub fn testing_tombstone_first_n(&mut self, n: usize) {
        let snapshot = self.shared.snapshot();
        let mut ids: Vec<u64> = snapshot.segments.iter().flat_map(|s| s.ids.iter().copied()).collect();
        ids.sort_unstable();
        ids.truncate(n);
        self.tombstone_chunks(&ids).expect("tombstone chunks");
    }

    pub fn tombstone_contains(&self, chunk_id: u64) -> bool {
        self.shared.snapshot().tombstones.contains(chunk_id)
    }

    pub fn tombstone_len(&self) -> usize {
        self.shared.snapshot().tombstones.len()
    }

    /// The live chunk id of `doc_id`'s first chunk.
    #[doc(hidden)]
    pub fn chunk_id_for_doc(&self, doc_id: &str) -> u64 {
        self.shared.snapshot().chunks_for_doc(doc_id).first().copied().expect("document is live")
    }
}
