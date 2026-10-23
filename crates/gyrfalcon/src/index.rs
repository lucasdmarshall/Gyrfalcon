//! `Index`: the writer lock, the published snapshot, and `index_documents`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use crate::chunk::chunk;
use crate::embed::{Embedder, HashEmbedder};
use crate::epoch::{ReaderGuard, Reaper};
use crate::error::{mismatch, Error, IncompatibleField, Result};
use crate::manifest::{check_compatible, IndexConfig, Manifest};
use crate::prepare::{prepare, prepare_with_prefix};
use crate::rerank::{NoopReranker, Reranker};
use crate::segment::{self, ChunkRecord, Segment};
use crate::store::{self, Catalog, CatalogRow, TermMap, Tombstones};
use crate::vector::quantize;

/// Texts per `embed` call during ingest.
pub const EMBED_BATCH: usize = 32;

/// The open segment seals when it reaches this many chunks.
pub const SEAL_EVERY: usize = 100_000;
/// Sealed segment count above which reports carry `warning: "segment_count"`.
pub const FAN_OUT_WARNING: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    pub doc_id: String,
    pub text: String,
    /// Ask the external LLM for a contextual prefix per chunk. Off by default.
    pub contextual_prefix: bool,
}

impl Document {
    /// A document without a contextual prefix.
    pub fn body(doc_id: impl Into<String>, text: impl Into<String>) -> Document {
        Document { doc_id: doc_id.into(), text: text.into(), contextual_prefix: false }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexReport {
    /// Chunk ids written by this call, ascending.
    pub chunk_ids: Vec<u64>,
    /// Segments sealed by this call.
    pub sealed: Vec<String>,
    /// Epoch of the last manifest this call published.
    pub epoch: u64,
    pub warning: Option<String>,
}

/// What a reader sees: one published manifest and the files it names.
pub(crate) struct Snapshot {
    pub manifest: Manifest,
    pub catalog: Arc<Catalog>,
    pub tombstones: Arc<Tombstones>,
    pub bm25: Arc<TermMap>,
    pub segments: Vec<Arc<Segment>>,
    /// Live (not tombstoned) chunk count per entry of `segments`.
    pub live: Vec<u64>,
}

impl Snapshot {
    pub fn new(
        manifest: Manifest,
        catalog: Arc<Catalog>,
        tombstones: Arc<Tombstones>,
        bm25: Arc<TermMap>,
        segments: Vec<Arc<Segment>>,
    ) -> Snapshot {
        let live = segments
            .iter()
            .map(|segment| {
                let dead = if tombstones.is_empty() {
                    0
                } else {
                    segment.ids.iter().filter(|id| tombstones.contains(**id)).count() as u64
                };
                segment.meta.chunk_count - dead
            })
            .collect();
        Snapshot { manifest, catalog, tombstones, bm25, segments, live }
    }

    pub fn segment_of(&self, chunk_id: u64) -> Option<&Arc<Segment>> {
        self.segments.iter().find(|segment| segment.contains(chunk_id))
    }

    pub fn warning(&self) -> Option<String> {
        (self.segments.len() > FAN_OUT_WARNING).then(|| "segment_count".to_string())
    }
}

/// One publish: every changed file is written at `generation + 1` before the manifest rename.
#[derive(Default)]
pub(crate) struct Change {
    pub add: Vec<Arc<Segment>>,
    pub remove: Vec<u64>,
    pub catalog: Option<Catalog>,
    pub tombstones: Option<Tombstones>,
    /// Collection term map with its `num_docs` and `num_tokens` totals.
    pub bm25: Option<(TermMap, u64, u64)>,
}

pub(crate) struct Shared {
    pub root: PathBuf,
    pub config: IndexConfig,
    /// The in-process writer lock. Holding it means holding the current snapshot.
    pub state: Mutex<Arc<Snapshot>>,
    pub next_chunk_id: AtomicU64,
    pub next_segment_id: AtomicU64,
    pub reaper: Mutex<Reaper>,
}

impl Shared {
    pub fn lock(&self) -> MutexGuard<'_, Arc<Snapshot>> {
        self.state.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn snapshot(&self) -> Arc<Snapshot> {
        self.lock().clone()
    }

    pub fn reaper(&self) -> MutexGuard<'_, Reaper> {
        self.reaper.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Copies the current epoch and increments its refcount.
    pub fn acquire(self: &Arc<Self>) -> ReaderGuard {
        let state = self.lock();
        self.reaper().acquire(state.manifest.epoch);
        ReaderGuard { shared: self.clone(), snapshot: state.clone() }
    }

    /// Runs the reaper against the current manifest.
    pub fn reap(&self) {
        let current = self.snapshot().manifest.sealed_segments.clone();
        self.reaper().poll_default(&current);
    }

    /// Queues the directories of `removed` segments under `retired_at`.
    pub fn retire(&self, removed: &[Arc<Segment>], retired_at: u64) {
        let mut reaper = self.reaper();
        for segment in removed {
            reaper.retire(segment.dir.clone(), segment.name(), retired_at);
        }
    }

    pub fn allocate_segment_id(&self) -> u64 {
        self.next_segment_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Writes the changed generation files, then renames the next manifest into place.
    pub fn publish(&self, current: &Snapshot, change: Change, fail_before_rename: bool) -> Result<Snapshot> {
        let root = &self.root;
        let mut manifest = current.manifest.clone();
        let mut retired_files = Vec::new();

        let catalog = match change.catalog {
            Some(catalog) => {
                retired_files.push(root.join(format!("catalog/{}.bin", manifest.catalog_generation)));
                manifest.catalog_generation += 1;
                let path = root.join(format!("catalog/{}.bin", manifest.catalog_generation));
                store::write_synced(&path, &catalog.encode())?;
                Arc::new(catalog)
            }
            None => current.catalog.clone(),
        };
        let tombstones = match change.tombstones {
            Some(tombstones) => {
                retired_files.push(root.join(format!("tombstones/{}.bin", manifest.tombstone_generation)));
                manifest.tombstone_generation += 1;
                let path = root.join(format!("tombstones/{}.bin", manifest.tombstone_generation));
                store::write_synced(&path, &tombstones.encode())?;
                Arc::new(tombstones)
            }
            None => current.tombstones.clone(),
        };
        let bm25 = match change.bm25 {
            Some((map, num_docs, num_tokens)) => {
                retired_files.push(root.join(format!("bm25/{}.bin", manifest.bm25_generation)));
                manifest.bm25_generation += 1;
                let path = root.join(format!("bm25/{}.bin", manifest.bm25_generation));
                store::write_synced(&path, &map.encode())?;
                manifest.bm25_num_docs = num_docs;
                manifest.bm25_num_tokens = num_tokens;
                Arc::new(map)
            }
            None => current.bm25.clone(),
        };
        for dir in ["catalog", "tombstones", "bm25"] {
            store::sync_dir(&root.join(dir))?;
        }

        let mut segments: Vec<Arc<Segment>> =
            current.segments.iter().filter(|s| !change.remove.contains(&s.id)).cloned().collect();
        segments.extend(change.add.iter().cloned());
        manifest.sealed_segments = segments.iter().map(|s| s.name()).collect();
        manifest.epoch += 1;
        manifest.next_chunk_id = self.next_chunk_id.load(Ordering::SeqCst);
        manifest.next_segment_id = self.next_segment_id.load(Ordering::SeqCst);

        if fail_before_rename {
            return Err(Error::InjectedFailure);
        }
        manifest.publish(root)?;
        for file in retired_files {
            let _ = std::fs::remove_file(file);
        }
        Ok(Snapshot::new(manifest, catalog, tombstones, bm25, segments))
    }
}

/// `writer.lock` holds the writer pid. A live pid other than a dropped index is `WriterBusy`.
struct WriterLock {
    path: PathBuf,
}

fn locks_held_here() -> &'static Mutex<HashSet<PathBuf>> {
    static HELD: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();
    HELD.get_or_init(|| Mutex::new(HashSet::new()))
}

fn pid_alive(pid: u32) -> bool {
    if pid == std::process::id() {
        return true;
    }
    #[cfg(target_os = "linux")]
    {
        Path::new(&format!("/proc/{pid}")).exists()
    }
    #[cfg(all(unix, not(target_os = "linux")))]
    {
        std::process::Command::new("kill")
            .args(["-0", &pid.to_string()])
            .stderr(std::process::Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(true)
    }
    #[cfg(not(unix))]
    {
        true
    }
}

impl WriterLock {
    fn acquire(root: &Path) -> Result<WriterLock> {
        let path = root.join("writer.lock");
        let key = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
        let mut held = locks_held_here().lock().unwrap_or_else(|p| p.into_inner());
        if held.contains(&key) {
            return Err(Error::WriterBusy { pid: std::process::id() });
        }
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Ok(pid) = text.trim().parse::<u32>() {
                // Our own pid without an entry in `held` is a lock left by a dropped index.
                if pid != std::process::id() && pid_alive(pid) {
                    return Err(Error::WriterBusy { pid });
                }
            }
        }
        store::write_synced(&path, std::process::id().to_string().as_bytes())?;
        held.insert(key);
        Ok(WriterLock { path })
    }
}

impl Drop for WriterLock {
    fn drop(&mut self) {
        let root = self.path.parent().unwrap_or(Path::new("."));
        let key = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
        locks_held_here().lock().unwrap_or_else(|p| p.into_inner()).remove(&key);
        let _ = std::fs::remove_file(&self.path);
    }
}

pub struct Index {
    pub(crate) shared: Arc<Shared>,
    pub(crate) embedder: Arc<dyn Embedder>,
    pub(crate) reranker: Arc<dyn Reranker>,
    /// Test hook: fail after the segment is fsynced and before the manifest rename.
    pub fail_before_rename: bool,
    /// The open segment seals at this many chunks.
    pub seal_every: usize,
    /// Test hook: HNSW returns nothing so the dense leg falls back to the exact scan.
    pub testing_starve_hnsw: bool,
    pub(crate) last_report: IndexReport,
    _lock: WriterLock,
    tempdir: Option<tempfile::TempDir>,
}

impl Index {
    /// Writes `manifest.json` at epoch 1 and empty generation-0 files.
    pub fn create(path: &Path, config: &IndexConfig) -> Result<Index> {
        if path.join("manifest.json").exists() {
            return Err(Error::AlreadyExists(path.to_path_buf()));
        }
        std::fs::create_dir_all(path)?;
        let lock = WriterLock::acquire(path)?;
        for dir in ["tombstones", "catalog", "bm25", "segments"] {
            std::fs::create_dir_all(path.join(dir))?;
        }
        store::write_synced(&path.join("tombstones/0.bin"), &Tombstones::default().encode())?;
        store::write_synced(&path.join("catalog/0.bin"), &Catalog::default().encode())?;
        store::write_synced(&path.join("bm25/0.bin"), &TermMap::default().encode())?;
        let manifest = Manifest::new(config);
        manifest.publish(path)?;
        Index::from_manifest(path, manifest, lock)
    }

    /// The open gate. Reads `manifest.json`, compares the four identity fields, and counts a
    /// segment open only when they all match. A mismatch maps nothing.
    pub fn open_counting(path: &Path, expected: &IndexConfig, opens: &mut u32) -> Result<Manifest> {
        if !path.is_dir() {
            return Err(Error::IndexNotFound(path.to_path_buf()));
        }
        let manifest = Manifest::read(&path.join("manifest.json"))?;
        check_compatible(&manifest, expected)?;
        *opens += 1;
        Ok(manifest)
    }

    pub fn open(path: &Path, expected: &IndexConfig) -> Result<Index> {
        let mut opens = 0;
        let manifest = Index::open_counting(path, expected, &mut opens)?;
        let lock = WriterLock::acquire(path)?;
        remove_unreferenced(path, &manifest)?;
        Index::from_manifest(path, manifest, lock)
    }

    /// Opens with an embedding session. The session's output width must equal
    /// `manifest.dimension`; a disagreement is `IncompatibleIndex` on `Dimension` and no
    /// segment is mapped.
    pub fn open_with_session<S: crate::embed::EmbedSession + 'static>(path: &Path, expected: &IndexConfig, session: S) -> Result<Index> {
        let mut opens = 0;
        let manifest = Index::open_counting(path, expected, &mut opens)?;
        if session.width() != manifest.dimension as usize {
            return mismatch(IncompatibleField::Dimension, &manifest.dimension.to_string(), &session.width().to_string());
        }
        let mut index = Index::open(path, expected)?;
        index.embedder = Arc::new(crate::embed::SessionEmbedder::new(session, manifest.model_id.clone()));
        Ok(index)
    }

    /// Opens with the ort embedder loaded from `model_dir` (`model.onnx`, `tokenizer.json`).
    pub fn open_with_model(path: &Path, expected: &IndexConfig, model_dir: &Path) -> Result<Index> {
        let mut opens = 0;
        Index::open_counting(path, expected, &mut opens)?;
        let model = crate::ort_session::OrtModel::load(model_dir)?;
        Index::open_with_session(path, expected, model)
    }

    fn from_manifest(path: &Path, manifest: Manifest, lock: WriterLock) -> Result<Index> {
        let catalog = Catalog::decode(&std::fs::read(
            path.join(format!("catalog/{}.bin", manifest.catalog_generation)),
        )?)?;
        let tombstones = Tombstones::decode(&std::fs::read(
            path.join(format!("tombstones/{}.bin", manifest.tombstone_generation)),
        )?)?;
        let bm25 = TermMap::read(&path.join(format!("bm25/{}.bin", manifest.bm25_generation)))?;
        let mut segments = Vec::new();
        for name in &manifest.sealed_segments {
            let id = store::parse_segment_name(name)
                .ok_or_else(|| Error::SegmentCorrupt(format!("bad segment id {name}")))?;
            segments.push(Arc::new(Segment::open(path, id, &manifest)?));
        }
        let config = manifest.config();
        let shared = Arc::new(Shared {
            root: path.to_path_buf(),
            config,
            next_chunk_id: AtomicU64::new(manifest.next_chunk_id),
            next_segment_id: AtomicU64::new(manifest.next_segment_id),
            reaper: Mutex::new(Reaper::default()),
            state: Mutex::new(Arc::new(Snapshot::new(
                manifest,
                Arc::new(catalog),
                Arc::new(tombstones),
                Arc::new(bm25),
                segments,
            ))),
        });
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(shared.config.dimension));
        Ok(Index {
            shared,
            embedder,
            reranker: Arc::new(NoopReranker),
            fail_before_rename: false,
            seal_every: SEAL_EVERY,
            testing_starve_hnsw: false,
            last_report: IndexReport::default(),
            _lock: lock,
            tempdir: None,
        })
    }

    /// Creates a collection in a fresh temporary directory that lives as long as the index.
    pub fn create_in_temp(config: &IndexConfig) -> Result<Index> {
        let dir = tempfile::tempdir()?;
        let mut index = Index::create(dir.path(), config)?;
        index.tempdir = Some(dir);
        Ok(index)
    }

    /// Keeps `dir` alive until the index drops.
    pub fn leak_tempdir(&mut self, dir: tempfile::TempDir) {
        self.tempdir = Some(dir);
    }

    pub fn path(&self) -> &Path {
        &self.shared.root
    }

    pub fn config(&self) -> &IndexConfig {
        &self.shared.config
    }

    /// The manifest this process last published or opened.
    pub fn manifest(&self) -> Manifest {
        self.shared.snapshot().manifest.clone()
    }

    pub fn last_report(&self) -> &IndexReport {
        &self.last_report
    }

    /// Replaces the embedder. Its width is checked against the manifest on every batch.
    pub fn set_embedder(&mut self, embedder: Box<dyn Embedder>) {
        self.embedder = Arc::from(embedder);
    }

    pub fn set_reranker(&mut self, reranker: Box<dyn Reranker>) {
        self.reranker = Arc::from(reranker);
    }

    /// Texts a recording embedder has received.
    #[doc(hidden)]
    pub fn embedder_texts(&self) -> Vec<String> {
        self.embedder.recorded()
    }

    pub(crate) fn check_width(&self, width: usize) -> Result<()> {
        let manifest_dimension = self.shared.config.dimension;
        if width != manifest_dimension as usize {
            return mismatch(IncompatibleField::Dimension, &manifest_dimension.to_string(), &width.to_string());
        }
        Ok(())
    }

    /// Acquires the current epoch. Its directories outlive any swap until the guard drops.
    pub fn acquire(&self) -> ReaderGuard {
        self.shared.acquire()
    }

    /// Chunks, embeds, and seals `docs`. Seals at `seal_every` chunks and seals any remainder
    /// before returning. Each seal publishes a manifest. Does not wait for a merge.
    pub fn index_documents(&mut self, docs: &[Document]) -> Result<IndexReport> {
        let mut records = self.chunk_records(docs)?;
        self.embed_records(&mut records)?;
        self.seal_and_report(records)
    }

    /// `index_documents` with caller-supplied model output: every chunk of a document gets
    /// that document's vector.
    #[doc(hidden)]
    pub fn index_documents_with_vectors(&mut self, rows: &[(Document, Vec<f32>)]) -> Result<IndexReport> {
        let mut records = Vec::new();
        for (doc, vector) in rows {
            self.check_width(vector.len())?;
            let row = quantize(self.shared.config.quantization, vector);
            for mut record in self.chunk_records(std::slice::from_ref(doc))? {
                record.vector = row.clone();
                records.push(record);
            }
        }
        self.seal_and_report(records)
    }

    /// One chunk per document, without the chunker. The accuracy fixture stores sections this way.
    pub fn index_sections(&mut self, docs: &[Document]) -> Result<IndexReport> {
        let mut records = Vec::new();
        for doc in docs {
            let prefix = if doc.contextual_prefix { Some(self.contextual_prefix(doc, &doc.text)?) } else { None };
            records.push(self.record(doc, 0, doc.text.len(), prefix));
        }
        self.embed_records(&mut records)?;
        self.seal_and_report(records)
    }

    pub(crate) fn seal_and_report(&mut self, records: Vec<ChunkRecord>) -> Result<IndexReport> {
        let shared = self.shared.clone();
        let mut state = shared.lock();
        let report = self.seal_records(&mut state, records)?;
        drop(state);
        self.after_publish();
        self.schedule_merge();
        self.last_report = report.clone();
        Ok(report)
    }

    /// Starts the background merger if none is running. Filled in by the merge module.
    pub(crate) fn schedule_merge(&self) {}

    /// Asks the external chat model for a chunk's contextual prefix.
    pub(crate) fn contextual_prefix(&self, _doc: &Document, _body: &str) -> Result<String> {
        Err(Error::Llm("no chat endpoint is configured for contextual prefixes".into()))
    }

    /// Runs after every successful write publish: retries queued deletes.
    pub(crate) fn after_publish(&self) {
        self.shared.reap();
    }

    fn record(&self, doc: &Document, start: usize, end: usize, prefix: Option<String>) -> ChunkRecord {
        let body = doc.text[start..end].to_string();
        let prepared = match &prefix {
            Some(prefix) => prepare_with_prefix(prefix, &body),
            None => prepare(&body),
        };
        ChunkRecord {
            chunk_id: 0,
            doc_id: doc.doc_id.clone(),
            byte_start: start as u64,
            byte_end: end as u64,
            body,
            prefix,
            prepared,
            vector: Vec::new(),
        }
    }

    pub(crate) fn chunk_records(&self, docs: &[Document]) -> Result<Vec<ChunkRecord>> {
        let mut records = Vec::new();
        for doc in docs {
            for span in chunk(&doc.text) {
                let prefix = if doc.contextual_prefix {
                    Some(self.contextual_prefix(doc, &doc.text[span.start..span.end])?)
                } else {
                    None
                };
                records.push(self.record(doc, span.start, span.end, prefix));
            }
        }
        Ok(records)
    }

    /// Embeds `embed_text` in batches and stores the quantized rows.
    pub(crate) fn embed_records(&self, records: &mut [ChunkRecord]) -> Result<()> {
        for batch in records.chunks_mut(EMBED_BATCH) {
            let texts: Vec<String> = batch.iter().map(|r| r.prepared.embed_text.clone()).collect();
            let vectors = self.embedder.embed(&texts)?;
            if vectors.len() != batch.len() {
                return Err(Error::Embed(format!("{} vectors for {} texts", vectors.len(), batch.len())));
            }
            for (record, vector) in batch.iter_mut().zip(vectors) {
                self.check_width(vector.len())?;
                record.vector = quantize(self.shared.config.quantization, &vector);
            }
        }
        Ok(())
    }

    /// Allocates chunk ids and seals `records` in groups of `seal_every`, publishing after each.
    pub(crate) fn seal_records(&self, state: &mut MutexGuard<'_, Arc<Snapshot>>, records: Vec<ChunkRecord>) -> Result<IndexReport> {
        self.seal_records_replacing(state, records, "", &[])
    }

    /// `seal_records` that also tombstones `previous` (the old chunks of `doc_id`) and drops
    /// their catalog rows in the rename that publishes the last new segment.
    pub(crate) fn seal_records_replacing(
        &self,
        state: &mut MutexGuard<'_, Arc<Snapshot>>,
        mut records: Vec<ChunkRecord>,
        doc_id: &str,
        previous: &[u64],
    ) -> Result<IndexReport> {
        let mut report = IndexReport { epoch: state.manifest.epoch, ..IndexReport::default() };
        for record in &mut records {
            record.chunk_id = self.shared.next_chunk_id.fetch_add(1, Ordering::SeqCst);
            report.chunk_ids.push(record.chunk_id);
        }
        let groups = records.len().div_ceil(self.seal_every.max(1));
        if groups == 0 && !previous.is_empty() {
            // An update to empty text only removes the old chunks.
            let rows = state.catalog.rows.iter().filter(|row| !previous.contains(&row.chunk_id)).cloned().collect();
            let change = Change {
                catalog: Some(Catalog::from_rows(rows)),
                tombstones: Some(state.tombstones.with(previous)),
                ..Change::default()
            };
            let next = self.shared.publish(state, change, self.fail_before_rename)?;
            **state = Arc::new(next);
            report.epoch = state.manifest.epoch;
        }
        for (index, group) in records.chunks(self.seal_every.max(1)).enumerate() {
            let last = index + 1 == groups;
            let id = self.shared.allocate_segment_id();
            segment::write_segment(&self.shared.root, id, &self.shared.config, group)?;
            let sealed = Arc::new(Segment::open(&self.shared.root, id, &state.manifest)?);
            let mut rows: Vec<CatalogRow> = if last && !previous.is_empty() {
                state.catalog.rows.iter().filter(|row| !(row.doc_id == doc_id && previous.contains(&row.chunk_id))).cloned().collect()
            } else {
                state.catalog.rows.clone()
            };
            rows.extend(group.iter().map(|record| CatalogRow {
                doc_id: record.doc_id.clone(),
                chunk_id: record.chunk_id,
                segment_id: id,
                byte_start: record.byte_start,
                byte_end: record.byte_end,
            }));
            let mut bm25 = (*state.bm25).clone();
            bm25.add(&sealed.terms);
            let num_docs = state.manifest.bm25_num_docs + sealed.meta.num_docs;
            let num_tokens = state.manifest.bm25_num_tokens + sealed.meta.num_tokens;
            let change = Change {
                add: vec![sealed],
                catalog: Some(Catalog::from_rows(rows)),
                bm25: Some((bm25, num_docs, num_tokens)),
                tombstones: (last && !previous.is_empty()).then(|| state.tombstones.with(previous)),
                ..Change::default()
            };
            let next = self.shared.publish(state, change, self.fail_before_rename)?;
            **state = Arc::new(next);
            report.sealed.push(store::segment_name(id));
            report.epoch = state.manifest.epoch;
        }
        report.warning = state.warning();
        Ok(report)
    }
}

/// Removes segment directories and a stale `manifest.json.tmp` the manifest does not name.
/// Called at open, before any reader exists in this process.
fn remove_unreferenced(root: &Path, manifest: &Manifest) -> Result<()> {
    let _ = std::fs::remove_file(root.join("manifest.json.tmp"));
    let listed: HashSet<&str> = manifest.sealed_segments.iter().map(String::as_str).collect();
    let segments = root.join("segments");
    if let Ok(entries) = std::fs::read_dir(&segments) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if !listed.contains(name.as_ref()) {
                std::fs::remove_dir_all(entry.path())?;
            }
        }
    }
    Ok(())
}
