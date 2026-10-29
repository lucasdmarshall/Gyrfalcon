//! `Index`: the writer lock, the published snapshot, and `index_documents`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};

use crate::buffer::{Batch, Pending};
use crate::chunk::chunk;
use crate::embed::{Embedder, HashEmbedder};
use crate::epoch::{ReaderGuard, Reaper};
use crate::error::{mismatch, Error, IncompatibleField, Result};
use crate::manifest::{check_compatible, IndexConfig, Manifest};

use crate::rerank::Reranker;
use crate::segment::{self, ChunkRecord, Segment};
use crate::store::{self, Tombstones};
use crate::vector::quantize;
use crate::wal::{wal_path, Frame, Wal, WalChunk};

/// Texts per `embed` call during ingest, at most.
pub const EMBED_BATCH: usize = 32;
/// Default `Embedder::batch_bytes`: bytes of text per `embed` call during ingest, counting
/// every text at the batch's longest.
pub const EMBED_BATCH_BYTES: usize = 32 * 1024;

/// The open segment seals when it reaches this many chunks.
pub const SEAL_EVERY: usize = 100_000;
/// Default `Index::write_buffer`: the buffer seals at this many chunks.
pub const WRITE_BUFFER: usize = 4096;
/// Writes of fewer chunks than this (and than `write_buffer`) are logged and buffered. A larger
/// write seals directly: its fixed seal cost is small per chunk, and buffering it would only
/// copy it again at the next seal.
pub const BUFFERED_WRITE_MAX: usize = 256;
/// The buffer also seals once its bodies and vectors hold this many bytes.
pub const WRITE_BUFFER_BYTES: usize = 32 << 20;
/// Sealed segment count above which reports carry `warning: "segment_count"`.
pub const FAN_OUT_WARNING: usize = 32;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Document {
    pub doc_id: String,
    pub text: String,
    /// Ask the external LLM for a contextual prefix per chunk. Off by default.
    pub contextual_prefix: bool,
    /// `(key, value)` pairs every chunk of the document carries and a search can filter on;
    /// a key may repeat for several values.
    pub metadata: Vec<(String, String)>,
}

impl Document {
    /// A document without a contextual prefix or metadata.
    pub fn body(doc_id: impl Into<String>, text: impl Into<String>) -> Document {
        Document { doc_id: doc_id.into(), text: text.into(), contextual_prefix: false, metadata: Vec::new() }
    }

    /// This document with `metadata` added.
    pub fn with_metadata<K: Into<String>, V: Into<String>>(
        mut self,
        metadata: impl IntoIterator<Item = (K, V)>,
    ) -> Document {
        self.metadata.extend(metadata.into_iter().map(|(k, v)| (k.into(), v.into())));
        self
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
    /// The tombstone file of the manifest plus ids killed by logged writes.
    pub tombstones: Arc<Tombstones>,
    pub segments: Vec<Arc<Segment>>,
    /// Live (not tombstoned) chunk count per entry of `segments`.
    pub live: Vec<u64>,
    /// Chunks written since the last seal, durable in the WAL.
    pub pending: Pending,
}

impl Snapshot {
    pub fn new(
        manifest: Manifest,
        tombstones: Arc<Tombstones>,
        segments: Vec<Arc<Segment>>,
        pending: Pending,
    ) -> Snapshot {
        // Dead ids are few and sorted, so each segment counts its own with a range lookup.
        let live = segments
            .iter()
            .map(|segment| {
                let dead = match (segment.ids.first(), segment.ids.last()) {
                    (Some(first), Some(last)) => tombstones
                        .in_range(*first, *last)
                        .iter()
                        .filter(|id| segment.ids.binary_search(id).is_ok())
                        .count() as u64,
                    _ => 0,
                };
                segment.meta.chunk_count - dead
            })
            .collect();
        Snapshot { manifest, tombstones, segments, live, pending }
    }

    /// Chunks holding `term` across the collection, dead ones included: the sum of every
    /// segment's dictionary and the buffer.
    pub fn doc_freq(&self, term: &str) -> u64 {
        self.segments.iter().map(|segment| u64::from(segment.lexical.doc_freq(term))).sum::<u64>()
            + self.pending.doc_freq(term)
    }

    /// BM25 `(num_docs, num_tokens)` over segments and buffer, dead chunks included.
    pub fn bm25_totals(&self) -> (u64, u64) {
        (
            self.manifest.bm25_num_docs + self.pending.len() as u64,
            self.manifest.bm25_num_tokens + self.pending.num_tokens(),
        )
    }

    /// Whether `chunk_id` is stored in a segment or the buffer, dead or alive.
    pub fn stores(&self, chunk_id: u64) -> bool {
        self.segment_of(chunk_id).is_some() || self.pending.find(chunk_id).is_some()
    }

    /// The stored fields of `chunk_id`, from its segment or the buffer.
    pub fn stored(&self, chunk_id: u64) -> Result<Option<crate::lexical::StoredChunk>> {
        if let Some(record) = self.pending.find(chunk_id) {
            return Ok(Some(crate::buffer::stored(record)));
        }
        match self.segment_of(chunk_id) {
            Some(segment) => segment.stored(chunk_id),
            None => Ok(None),
        }
    }

    /// The stored vector of `chunk_id` as `f32`, scaled as stored.
    pub fn stored_vector(&self, chunk_id: u64) -> Option<Vec<f32>> {
        if let Some(record) = self.pending.find(chunk_id) {
            return Some(crate::vector::row_values(self.manifest.quantization, &record.vector));
        }
        let segment = self.segment_of(chunk_id)?;
        segment.vector.stored_vector(chunk_id, &segment.ids)
    }

    /// The live chunk ids of caller document `doc_id`, ascending.
    pub fn chunks_for_doc(&self, doc_id: &str) -> Vec<u64> {
        let mut ids: Vec<u64> = self
            .segments
            .iter()
            .flat_map(|segment| segment.keys.chunks_for(doc_id))
            .chain(self.pending.records().filter(|r| r.doc_id == doc_id).map(|r| r.chunk_id))
            .filter(|id| !self.tombstones.contains(*id))
            .collect();
        ids.sort_unstable();
        ids
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
    pub tombstones: Option<Tombstones>,
    /// The buffer was sealed into `add`: start an empty WAL generation.
    pub sealed_buffer: bool,
}

pub(crate) struct Shared {
    pub root: PathBuf,
    pub config: IndexConfig,
    /// The in-process writer lock. Holding it means holding the current snapshot.
    pub state: Mutex<Arc<Snapshot>>,
    pub next_chunk_id: AtomicU64,
    pub next_segment_id: AtomicU64,
    pub reaper: Mutex<Reaper>,
    pub merge: crate::merge::MergeControl,
    /// The open WAL generation; appended under `state`.
    pub wal: Mutex<Wal>,
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

    /// Deletes the retired directories no reader needs, on this thread.
    pub fn reap(&self) {
        let current = self.snapshot().manifest.sealed_segments.clone();
        crate::epoch::reap(&self.reaper, &current);
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
        store::sync_dir(&root.join("tombstones"))?;

        let mut segments: Vec<Arc<Segment>> =
            current.segments.iter().filter(|s| !change.remove.contains(&s.id)).cloned().collect();
        segments.extend(change.add.iter().cloned());
        manifest.sealed_segments = segments.iter().map(|s| s.name()).collect();
        // BM25 totals are the segments' sums, dead chunks included until a merge drops them.
        manifest.bm25_num_docs = segments.iter().map(|s| s.meta.num_docs).sum();
        manifest.bm25_num_tokens = segments.iter().map(|s| s.meta.num_tokens).sum();
        manifest.epoch += 1;
        manifest.next_chunk_id = self.next_chunk_id.load(Ordering::SeqCst);
        manifest.next_segment_id = self.next_segment_id.load(Ordering::SeqCst);
        if change.sealed_buffer {
            retired_files.push(wal_path(root, manifest.wal_generation));
            manifest.wal_generation += 1;
        }

        if fail_before_rename {
            return Err(Error::InjectedFailure);
        }
        manifest.publish(root)?;
        let pending = if change.sealed_buffer {
            *self.wal.lock().unwrap_or_else(|p| p.into_inner()) = Wal::open(&wal_path(root, manifest.wal_generation))?;
            Pending::default()
        } else {
            current.pending.clone()
        };
        for file in retired_files {
            let _ = std::fs::remove_file(file);
        }
        Ok(Snapshot::new(manifest, tombstones, segments, pending))
    }
}

/// Threads for ingest work: every available core.
pub(crate) fn worker_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// Maps `items` on every core, keeping order, and flattens the per-item outputs.
pub(crate) fn parallel_map<T: Sync, R: Send>(items: &[T], f: impl Fn(&T) -> Vec<R> + Sync) -> Vec<R> {
    let threads = worker_threads();
    if threads == 1 || items.len() < 2 * threads {
        return items.iter().flat_map(&f).collect();
    }
    let per_thread = items.len().div_ceil(threads);
    std::thread::scope(|scope| {
        let f = &f;
        let parts: Vec<_> = items
            .chunks(per_thread)
            .map(|part| scope.spawn(move || part.iter().flat_map(f).collect::<Vec<R>>()))
            .collect();
        parts.into_iter().flat_map(|part| part.join().expect("ingest worker panicked")).collect()
    })
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
    #[cfg(windows)]
    {
        // `tasklist` lists the process when it exists; anything else, including a failure to
        // run it, keeps the lock.
        std::process::Command::new("tasklist")
            .args(["/FI", &format!("PID eq {pid}"), "/FO", "CSV", "/NH"])
            .output()
            .map(|out| String::from_utf8_lossy(&out.stdout).contains(&format!("\"{pid}\"")))
            .unwrap_or(true)
    }
    #[cfg(not(any(unix, windows)))]
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
    /// The cross-encoder `rerank` runs; none until `set_reranker`.
    pub(crate) reranker: Option<Arc<dyn Reranker>>,
    pub(crate) chat: Option<crate::answer::ChatConfig>,
    /// Test hook: fail after the segment is fsynced and before the manifest rename.
    pub fail_before_rename: bool,
    /// The open segment seals at this many chunks.
    pub seal_every: usize,
    /// Writes of fewer chunks than this and than `BUFFERED_WRITE_MAX` go to the WAL and the
    /// in-memory buffer, which seals once it holds this many chunks or `WRITE_BUFFER_BYTES`.
    /// 0 seals every write directly.
    pub write_buffer: usize,
    /// Run the background merger after writes. On by default.
    pub auto_merge: bool,
    /// Test hook: the merger blocks after its build and before the swap.
    pub pause_merge_after_copy: bool,
    /// Test hook: the next `reap_now` fails this many deletes with a sharing violation.
    pub deleter_fail_times: u32,
    /// RRF weights `[lexical, dense]`. The spec default is `[1, 1]`.
    pub rrf_weights: [f32; 2],
    /// How retrieve combines the legs. Rescore with dense weight 0.2 unless set.
    pub fusion: crate::fuse::Fusion,
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
        for dir in ["tombstones", "segments"] {
            std::fs::create_dir_all(path.join(dir))?;
        }
        store::write_synced(&path.join("tombstones/0.bin"), &Tombstones::default().encode())?;
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
    pub fn open_with_session<S: crate::embed::EmbedSession + 'static>(
        path: &Path,
        expected: &IndexConfig,
        session: S,
    ) -> Result<Index> {
        let mut opens = 0;
        let manifest = Index::open_counting(path, expected, &mut opens)?;
        if session.width() != manifest.dimension as usize {
            return mismatch(
                IncompatibleField::Dimension,
                &manifest.dimension.to_string(),
                &session.width().to_string(),
            );
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
        let tombstones = Tombstones::decode(&std::fs::read(
            path.join(format!("tombstones/{}.bin", manifest.tombstone_generation)),
        )?)?;
        let mut segments = Vec::new();
        for name in &manifest.sealed_segments {
            let id = store::parse_segment_name(name)
                .ok_or_else(|| Error::SegmentCorrupt(format!("bad segment id {name}")))?;
            segments.push(Arc::new(Segment::open(path, id, &manifest)?));
        }
        // The first query would otherwise fault these pages one at a time. Seal and merge open
        // segments they just wrote, so they skip this; those pages are already cached.
        for segment in &segments {
            segment.warm()?;
        }
        let config = manifest.config();
        // Logged writes since the last seal: their chunks become the buffer, their dead ids
        // join the tombstones, and chunk ids continue after the highest one logged.
        let wal_file = wal_path(path, manifest.wal_generation);
        let mut records = Vec::new();
        let mut dead = Vec::new();
        for frame in crate::wal::replay(&wal_file)? {
            records.extend(frame.chunks.into_iter().map(record_from_wal));
            dead.extend(frame.dead);
        }
        let next_chunk_id = records.iter().map(|r| r.chunk_id + 1).max().unwrap_or(0).max(manifest.next_chunk_id);
        let pending = if records.is_empty() {
            Pending::default()
        } else {
            let records = records.into_iter().map(Arc::new).collect();
            Pending::default().with(Batch::new(records, config.quantization), config.quantization)
        };
        let tombstones = if dead.is_empty() { tombstones } else { tombstones.with(&dead) };
        let shared = Arc::new(Shared {
            root: path.to_path_buf(),
            config,
            next_chunk_id: AtomicU64::new(next_chunk_id),
            next_segment_id: AtomicU64::new(manifest.next_segment_id),
            reaper: Mutex::new(Reaper::default()),
            merge: crate::merge::MergeControl::default(),
            wal: Mutex::new(Wal::open(&wal_file)?),
            state: Mutex::new(Arc::new(Snapshot::new(manifest, Arc::new(tombstones), segments, pending))),
        });
        let embedder: Arc<dyn Embedder> = Arc::new(HashEmbedder::new(shared.config.dimension));
        Ok(Index {
            shared,
            embedder,
            reranker: None,
            chat: None,
            fail_before_rename: false,
            seal_every: SEAL_EVERY,
            write_buffer: WRITE_BUFFER,
            rrf_weights: [crate::fuse::LEXICAL_WEIGHT, crate::fuse::DENSE_WEIGHT],
            fusion: crate::fuse::Fusion::default(),
            auto_merge: true,
            pause_merge_after_copy: false,
            deleter_fail_times: 0,
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

    /// The embedder this index uses for ingest and queries.
    pub fn embedder(&self) -> Arc<dyn Embedder> {
        self.embedder.clone()
    }

    pub fn set_reranker(&mut self, reranker: Box<dyn Reranker>) {
        self.reranker = Some(Arc::from(reranker));
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
            for record in self.chunk_records(std::slice::from_ref(doc))? {
                records.push(record.with_vector(0, row.clone()));
            }
        }
        self.seal_and_report(records)
    }

    /// One chunk per document, without the chunker. The accuracy fixture stores sections this way.
    pub fn index_sections(&mut self, docs: &[Document]) -> Result<IndexReport> {
        let mut records = if docs.iter().any(|d| d.contextual_prefix) {
            let mut records = Vec::new();
            for doc in docs {
                let prefix = if doc.contextual_prefix { Some(self.contextual_prefix(doc, &doc.text)?) } else { None };
                records.push(self.record(doc, 0, doc.text.len(), prefix));
            }
            records
        } else {
            parallel_map(docs, |doc| vec![self.record(doc, 0, doc.text.len(), None)])
        };
        self.embed_records(&mut records)?;
        self.seal_and_report(records)
    }

    pub(crate) fn seal_and_report(&mut self, records: Vec<ChunkRecord>) -> Result<IndexReport> {
        let shared = self.shared.clone();
        let mut state = shared.lock();
        let report = if self.buffers(records.len()) {
            self.log_records(&mut state, records, &[])?
        } else {
            self.seal_records(&mut state, records)?
        };
        drop(state);
        self.after_publish();
        self.schedule_merge();
        self.last_report = report.clone();
        Ok(report)
    }

    /// Whether a write of `chunks` chunks goes to the WAL and the buffer.
    pub(crate) fn buffers(&self, chunks: usize) -> bool {
        chunks < self.write_buffer.min(BUFFERED_WRITE_MAX)
    }

    /// Seals the write buffer into a segment now. A no-op when it is empty.
    pub fn flush(&mut self) -> Result<IndexReport> {
        let shared = self.shared.clone();
        let mut state = shared.lock();
        let mut report = IndexReport { epoch: state.manifest.epoch, ..IndexReport::default() };
        self.seal_buffer(&mut state, &mut report)?;
        drop(state);
        if !report.sealed.is_empty() {
            self.after_publish();
            self.schedule_merge();
        }
        Ok(report)
    }

    /// Allocates chunk ids for `records`, logs them with the `dead` ids in one WAL frame, and
    /// adds them to the buffer and the tombstones. Seals the buffer once it is full.
    pub(crate) fn log_records(
        &self,
        state: &mut MutexGuard<'_, Arc<Snapshot>>,
        mut records: Vec<ChunkRecord>,
        dead: &[u64],
    ) -> Result<IndexReport> {
        let mut report = IndexReport { epoch: state.manifest.epoch, ..IndexReport::default() };
        for record in &mut records {
            record.chunk_id = self.shared.next_chunk_id.fetch_add(1, Ordering::SeqCst);
            report.chunk_ids.push(record.chunk_id);
        }
        let frame = Frame { chunks: records.iter().map(wal_chunk).collect(), dead: dead.to_vec() };
        self.shared.wal.lock().unwrap_or_else(|p| p.into_inner()).append(&frame)?;
        let quantization = self.shared.config.quantization;
        let pending = if records.is_empty() {
            state.pending.clone()
        } else {
            state.pending.with(Batch::new(records.into_iter().map(Arc::new).collect(), quantization), quantization)
        };
        let tombstones = if dead.is_empty() { state.tombstones.clone() } else { Arc::new(state.tombstones.with(dead)) };
        let next = Snapshot::new(state.manifest.clone(), tombstones, state.segments.clone(), pending);
        **state = Arc::new(next);
        if state.pending.len() >= self.write_buffer.max(1) || state.pending.bytes() >= WRITE_BUFFER_BYTES {
            self.seal_buffer(state, &mut report)?;
        }
        report.warning = state.warning();
        Ok(report)
    }

    /// Writes the buffer as one segment and publishes it with the tombstones it logged, which
    /// starts an empty WAL generation.
    fn seal_buffer(&self, state: &mut MutexGuard<'_, Arc<Snapshot>>, report: &mut IndexReport) -> Result<()> {
        if state.pending.is_empty() {
            return Ok(());
        }
        let id = self.shared.allocate_segment_id();
        let records: Vec<&ChunkRecord> = state.pending.records().collect();
        segment::write_segment(&self.shared.root, id, &self.shared.config, &records)?;
        drop(records);
        let sealed = Arc::new(Segment::open(&self.shared.root, id, &state.manifest)?);
        let change = Change {
            add: vec![sealed],
            tombstones: Some(state.tombstones.as_ref().clone()),
            sealed_buffer: true,
            ..Change::default()
        };
        let next = self.shared.publish(state, change, self.fail_before_rename)?;
        **state = Arc::new(next);
        report.sealed.push(store::segment_name(id));
        report.epoch = state.manifest.epoch;
        Ok(())
    }

    /// Runs after every successful write publish: retries queued deletes.
    pub(crate) fn after_publish(&self) {
        self.shared.reap();
    }

    fn record(&self, doc: &Document, start: usize, end: usize, prefix: Option<String>) -> ChunkRecord {
        ChunkRecord::new(
            doc.doc_id.clone(),
            start as u64,
            end as u64,
            doc.text[start..end].to_string(),
            prefix,
            doc.metadata.clone(),
        )
    }

    pub(crate) fn chunk_records(&self, docs: &[Document]) -> Result<Vec<ChunkRecord>> {
        if !docs.iter().any(|d| d.contextual_prefix) {
            return Ok(parallel_map(docs, |doc| {
                chunk(&doc.text).into_iter().map(|span| self.record(doc, span.start, span.end, None)).collect()
            }));
        }
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

    /// Embeds `embed_text` in batches and stores the quantized rows. Texts are batched by
    /// length, longest first: a padded model (ONNX) computes each batch at its longest text, so
    /// similar lengths waste no padding, and the embedder's `batch_bytes` bounds a batch's
    /// activations.
    /// Batches run on every core; an embedder that serializes internally (an ort session)
    /// simply takes turns. Each text gets the same vector in any batch.
    pub(crate) fn embed_records(&self, records: &mut [ChunkRecord]) -> Result<()> {
        let mut order: Vec<usize> = (0..records.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(records[i].embed_text.len()));
        let budget = self.embedder.batch_bytes();
        let mut batches: Vec<(Vec<usize>, Vec<String>)> = Vec::new();
        let mut start = 0;
        while start < order.len() {
            let longest = records[order[start]].embed_text.len().max(1);
            let end = (start + (budget / longest).clamp(1, EMBED_BATCH)).min(order.len());
            let rows = order[start..end].to_vec();
            // The records keep no copy of the text they were embedded from.
            let texts = rows.iter().map(|&i| std::mem::take(&mut records[i].embed_text)).collect();
            batches.push((rows, texts));
            start = end;
        }
        let next = AtomicUsize::new(0);
        let run = || -> Result<Vec<(usize, Vec<u8>)>> {
            let mut out = Vec::new();
            while let Some((rows, texts)) = batches.get(next.fetch_add(1, Ordering::Relaxed)) {
                let vectors = self.embedder.embed(texts)?;
                if vectors.len() != rows.len() {
                    return Err(Error::Embed(format!("{} vectors for {} texts", vectors.len(), rows.len())));
                }
                for (&row, vector) in rows.iter().zip(vectors) {
                    self.check_width(vector.len())?;
                    out.push((row, quantize(self.shared.config.quantization, &vector)));
                }
            }
            Ok(out)
        };
        let threads = worker_threads().min(batches.len());
        let rows = if threads <= 1 {
            run()?
        } else {
            std::thread::scope(|scope| -> Result<Vec<(usize, Vec<u8>)>> {
                let workers: Vec<_> = (0..threads).map(|_| scope.spawn(run)).collect();
                let mut rows = Vec::with_capacity(records.len());
                for worker in workers {
                    rows.extend(worker.join().map_err(|_| Error::Embed("embed thread panicked".into()))??);
                }
                Ok(rows)
            })?
        };
        for (row, vector) in rows {
            records[row].vector = vector;
        }
        Ok(())
    }

    /// Allocates chunk ids and seals `records` in groups of `seal_every`, publishing after each.
    pub(crate) fn seal_records(
        &self,
        state: &mut MutexGuard<'_, Arc<Snapshot>>,
        records: Vec<ChunkRecord>,
    ) -> Result<IndexReport> {
        self.seal_records_replacing(state, records, &[])
    }

    /// `seal_records` that also tombstones `previous` (a document's old chunks) in the rename
    /// that publishes the last new segment.
    pub(crate) fn seal_records_replacing(
        &self,
        state: &mut MutexGuard<'_, Arc<Snapshot>>,
        mut records: Vec<ChunkRecord>,
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
            let change = Change { tombstones: Some(state.tombstones.with(previous)), ..Change::default() };
            let next = self.shared.publish(state, change, self.fail_before_rename)?;
            **state = Arc::new(next);
            report.epoch = state.manifest.epoch;
        }
        for (index, group) in records.chunks(self.seal_every.max(1)).enumerate() {
            let last = index + 1 == groups;
            let id = self.shared.allocate_segment_id();
            segment::write_segment(&self.shared.root, id, &self.shared.config, group)?;
            let sealed = Arc::new(Segment::open(&self.shared.root, id, &state.manifest)?);
            let change = Change {
                add: vec![sealed],
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

impl Drop for Index {
    fn drop(&mut self) {
        self.stop_merger();
    }
}

/// A logged chunk as a record, its terms recomputed by `prepare`.
fn record_from_wal(chunk: WalChunk) -> ChunkRecord {
    ChunkRecord::new(chunk.doc_id, chunk.byte_start, chunk.byte_end, chunk.body, chunk.prefix, chunk.metadata)
        .with_vector(chunk.chunk_id, chunk.vector)
}

fn wal_chunk(record: &ChunkRecord) -> WalChunk {
    WalChunk {
        chunk_id: record.chunk_id,
        doc_id: record.doc_id.clone(),
        byte_start: record.byte_start,
        byte_end: record.byte_end,
        body: record.body.clone(),
        prefix: record.prefix.clone(),
        metadata: record.metadata.clone(),
        vector: record.vector.clone(),
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
    // WAL generations other than the manifest's were sealed already, or never published.
    let current = wal_path(root, manifest.wal_generation);
    if let Ok(entries) = std::fs::read_dir(root.join("wal")) {
        for entry in entries.flatten() {
            if entry.path() != current {
                std::fs::remove_file(entry.path())?;
            }
        }
    }
    Ok(())
}
