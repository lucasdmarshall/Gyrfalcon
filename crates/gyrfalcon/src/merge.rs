//! The background merger: tiered merges of small segments, then a dead-ratio rewrite of one
//! large segment. Builds run outside the writer lock; the swap is one manifest rename.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::error::Result;
use crate::index::{Change, Index, Shared};
use crate::manifest::{DEAD_RATIO_LIMIT, MAX_MERGE_CHUNKS, SEGMENTS_PER_TIER};
use crate::segment::{self, ChunkRecord, Segment};

/// Picks the lowest tier holding at least 8 segments whose 8 smallest counts sum to at most
/// 100,000. A segment of 100,000 chunks or more is never an input. Tier is
/// `floor(log2(chunk_count))`.
pub fn select_tier(segments: &[(String, u64)]) -> Option<Vec<String>> {
    let per_tier = SEGMENTS_PER_TIER as usize;
    let mut by_tier: BTreeMap<u32, Vec<(String, u64)>> = BTreeMap::new();
    for (id, count) in segments {
        if *count >= MAX_MERGE_CHUNKS {
            continue;
        }
        let tier = (*count).max(1).ilog2();
        by_tier.entry(tier).or_default().push((id.clone(), *count));
    }
    for group in by_tier.values_mut() {
        group.sort_by_key(|(_, count)| *count);
        if group.len() >= per_tier && group.iter().take(per_tier).map(|(_, c)| *c).sum::<u64>() <= MAX_MERGE_CHUNKS {
            return Some(group.iter().take(per_tier).map(|(id, _)| id.clone()).collect());
        }
    }
    None
}

/// `(segment, stored, dead)`: the segment of at least 100,000 stored chunks with the highest
/// dead ratio strictly above 0.20.
pub fn dead_ratio_target(segments: &[(String, u64, u64)]) -> Option<String> {
    segments
        .iter()
        .filter(|(_, stored, dead)| *stored >= MAX_MERGE_CHUNKS && *dead as f64 / *stored as f64 > DEAD_RATIO_LIMIT)
        .max_by(|a, b| (a.2 as f64 / a.1 as f64).total_cmp(&(b.2 as f64 / b.1 as f64)))
        .map(|(id, _, _)| id.clone())
}

#[derive(Default)]
pub(crate) struct MergeState {
    pub requested: bool,
    pub running: bool,
    /// The build finished and the worker is paused before the swap.
    pub copied: bool,
    pub shutdown: bool,
    pub pause_after_copy: bool,
    pub auto_merge: bool,
    pub thread: Option<JoinHandle<()>>,
    pub last_error: Option<String>,
}

#[derive(Default)]
pub(crate) struct MergeControl {
    pub state: Mutex<MergeState>,
    pub changed: Condvar,
}

impl MergeControl {
    pub fn lock(&self) -> MutexGuard<'_, MergeState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn wait<'a>(&self, guard: MutexGuard<'a, MergeState>) -> MutexGuard<'a, MergeState> {
        self.changed.wait_timeout(guard, Duration::from_millis(200)).unwrap_or_else(|p| p.into_inner()).0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Tier,
    DeadRatio,
    /// `gyr compact --all`: every segment into one, every dead id dropped.
    All,
}

fn worker(shared: Arc<Shared>) {
    loop {
        {
            let mut state = shared.merge.lock();
            while !state.requested && !state.shutdown {
                state = shared.merge.wait(state);
            }
            if state.shutdown {
                state.running = false;
                shared.merge.changed.notify_all();
                return;
            }
            state.requested = false;
            state.running = true;
        }
        loop {
            match run_pass(&shared, false) {
                Ok(true) => continue,
                Ok(false) => break,
                Err(err) => {
                    shared.merge.lock().last_error = Some(err.to_string());
                    break;
                }
            }
        }
        let mut state = shared.merge.lock();
        state.running = false;
        shared.merge.changed.notify_all();
    }
}

/// One tier merge or dead-ratio rewrite, or with `all` a rewrite of every segment. Returns
/// whether it published.
pub(crate) fn run_pass(shared: &Arc<Shared>, all: bool) -> Result<bool> {
    // Step 1: pick inputs under the writer lock and copy the tombstone bitmap.
    let (inputs, kind, copied_dead, new_id) = {
        let state = shared.lock();
        if all {
            if state.segments.is_empty() || (state.segments.len() == 1 && state.tombstones.is_empty()) {
                return Ok(false);
            }
            let inputs = state.segments.clone();
            (inputs, Kind::All, state.tombstones.clone(), shared.allocate_segment_id())
        } else {
            let counts: Vec<(String, u64)> = state.segments.iter().map(|s| (s.name(), s.meta.chunk_count)).collect();
            let (names, kind) = match select_tier(&counts) {
                Some(names) => (names, Kind::Tier),
                None => {
                    let ratios: Vec<(String, u64, u64)> = state
                        .segments
                        .iter()
                        .zip(&state.live)
                        .map(|(s, live)| (s.name(), s.meta.chunk_count, s.meta.chunk_count - live))
                        .collect();
                    match dead_ratio_target(&ratios) {
                        Some(name) => (vec![name], Kind::DeadRatio),
                        None => return Ok(false),
                    }
                }
            };
            let inputs: Vec<Arc<Segment>> =
                state.segments.iter().filter(|s| names.contains(&s.name())).cloned().collect();
            (inputs, kind, state.tombstones.clone(), shared.allocate_segment_id())
        }
    };

    // Step 2: build outside the lock. Live chunks keep their chunk ids.
    let mut records = Vec::new();
    for input in &inputs {
        let vectors = std::fs::read(input.dir.join("vectors.bin"))?;
        let row_len = vectors.len() / input.ids.len().max(1);
        for stored in input.lexical.all_stored()? {
            if copied_dead.contains(stored.chunk_id) {
                continue;
            }
            let Ok(row) = input.ids.binary_search(&stored.chunk_id) else { continue };
            let vector = vectors[row * row_len..(row + 1) * row_len].to_vec();
            records.push(
                ChunkRecord::new(
                    stored.doc_key,
                    stored.byte_start,
                    stored.byte_end,
                    stored.body,
                    stored.prefix,
                    stored.metadata,
                )
                .with_vector(stored.chunk_id, vector),
            );
        }
    }
    records.sort_by_key(|r| r.chunk_id);
    let dir = segment::write_segment(&shared.root, new_id, &shared.config, &records)?;
    let abandon = |dir: &PathBuf| {
        let _ = std::fs::remove_dir_all(dir);
    };

    {
        let mut control = shared.merge.lock();
        if control.pause_after_copy {
            control.copied = true;
            shared.merge.changed.notify_all();
            while control.pause_after_copy && !control.shutdown {
                control = shared.merge.wait(control);
            }
            control.copied = false;
        }
        if control.shutdown {
            drop(control);
            abandon(&dir);
            return Ok(false);
        }
    }

    // Step 3: swap under the writer lock against the manifest as it is now.
    let mut state = shared.lock();
    if !inputs.iter().all(|input| state.segments.iter().any(|s| s.id == input.id)) {
        drop(state);
        abandon(&dir);
        return Ok(false);
    }
    let merged = match Segment::open(&shared.root, new_id, &state.manifest) {
        Ok(merged) => Arc::new(merged),
        Err(err) => {
            drop(state);
            abandon(&dir);
            return Err(err);
        }
    };
    let current_dead = state.tombstones.clone();
    let tombstones = match kind {
        Kind::Tier => None,
        Kind::DeadRatio | Kind::All => {
            // Ids stored only in the old files leave the bitmap. A chunk deleted during the
            // build is still stored in the new files, so it stays.
            let dropped: Vec<u64> = inputs
                .iter()
                .flat_map(|input| input.ids.iter().copied())
                .filter(|id| current_dead.contains(*id) && !merged.contains(*id))
                .collect();
            Some(current_dead.without(&dropped))
        }
    };
    let change =
        Change { add: vec![merged], remove: inputs.iter().map(|s| s.id).collect(), tombstones, ..Change::default() };
    let next = match shared.publish(&state, change, false) {
        Ok(next) => next,
        Err(err) => {
            drop(state);
            abandon(&dir);
            return Err(err);
        }
    };
    let epoch = next.manifest.epoch;
    *state = Arc::new(next);
    drop(state);
    // The swap queues the inputs and does not delete them itself.
    shared.retire(&inputs, epoch);
    Ok(true)
}

impl Index {
    fn sync_merge_knobs(&self, control: &mut MergeState) {
        control.pause_after_copy = self.pause_merge_after_copy;
        control.auto_merge = self.auto_merge;
    }

    /// Starts the background merger if none is running, or asks it for another pass.
    pub(crate) fn schedule_merge(&self) {
        let mut control = self.shared.merge.lock();
        self.sync_merge_knobs(&mut control);
        if !control.auto_merge {
            return;
        }
        control.requested = true;
        if control.thread.is_none() {
            let shared = self.shared.clone();
            control.thread = Some(
                std::thread::Builder::new()
                    .name("gyr-merge".into())
                    .spawn(move || worker(shared))
                    .expect("spawn merger"),
            );
        }
        self.shared.merge.changed.notify_all();
    }

    /// Releases a paused merger, runs a pass if auto-merge is on, and blocks until the merger
    /// is idle.
    pub fn wait_until_merge_idle(&mut self) {
        self.pause_merge_after_copy = false;
        self.schedule_merge();
        let mut control = self.shared.merge.lock();
        self.sync_merge_knobs(&mut control);
        self.shared.merge.changed.notify_all();
        while control.requested || control.running {
            control = self.shared.merge.wait(control);
        }
    }

    /// Blocks until a paused merger has finished its build.
    #[doc(hidden)]
    pub fn wait_until_merge_copied(&self) {
        let mut control = self.shared.merge.lock();
        while !control.copied {
            control = self.shared.merge.wait(control);
        }
    }

    /// The last error a background pass hit, if any.
    pub fn merge_error(&self) -> Option<String> {
        self.shared.merge.lock().last_error.clone()
    }

    /// Seals the write buffer, then one synchronous tier pass (`gyr compact`).
    pub fn compact(&mut self) -> Result<bool> {
        self.flush()?;
        self.wait_until_merge_idle();
        let published = run_pass(&self.shared, false)?;
        self.after_publish();
        Ok(published)
    }

    /// Rewrites every segment into one from its live chunks, keeps their chunk ids, clears the
    /// tombstone bitmap, and queues the old directories for the reaper (`gyr compact --all`).
    pub fn compact_all(&mut self) -> Result<bool> {
        self.flush()?;
        self.wait_until_merge_idle();
        let published = run_pass(&self.shared, true)?;
        self.after_publish();
        Ok(published)
    }

    /// Stored chunk count of every published segment, in manifest order.
    pub fn segment_chunk_counts(&self) -> Vec<u64> {
        self.shared.snapshot().segments.iter().map(|s| s.meta.chunk_count).collect()
    }

    pub fn segment_dirs(&self) -> Vec<PathBuf> {
        self.shared.snapshot().segments.iter().map(|s| s.dir.clone()).collect()
    }

    /// Directories queued for the reaper.
    pub fn retired_dirs(&self) -> Vec<PathBuf> {
        self.shared.reaper().retired_dirs()
    }

    /// Runs the reaper now. Applies `deleter_fail_times` first.
    pub fn reap_now(&mut self) {
        if self.deleter_fail_times > 0 {
            self.shared.reaper().deleter.fail_times = std::mem::take(&mut self.deleter_fail_times);
        }
        self.shared.reap();
    }

    pub(crate) fn stop_merger(&mut self) {
        let thread = {
            let mut control = self.shared.merge.lock();
            control.shutdown = true;
            self.shared.merge.changed.notify_all();
            control.thread.take()
        };
        if let Some(thread) = thread {
            let _ = thread.join();
        }
    }

    /// Test-only: seals one segment of `n` chunks whose text is `token`, even when `n` is
    /// 100,000 or more. Vectors are distinct pseudo-random unit vectors.
    #[doc(hidden)]
    pub fn seal_n_chunks(&mut self, n: usize, token: &str) -> Result<()> {
        let dimension = self.shared.config.dimension as usize;
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        let mut records = Vec::with_capacity(n);
        for i in 0..n {
            let vector: Vec<f32> = (0..dimension)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    (seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5
                })
                .collect();
            let row = crate::vector::quantize(self.shared.config.quantization, &vector);
            records.push(
                ChunkRecord::new(format!("{token}-{i}"), 0, token.len() as u64, token.to_string(), None, Vec::new())
                    .with_vector(0, row),
            );
        }
        let seal_every = std::mem::replace(&mut self.seal_every, n.max(1));
        let write_buffer = std::mem::replace(&mut self.write_buffer, 0);
        let result = self.seal_and_report(records);
        self.seal_every = seal_every;
        self.write_buffer = write_buffer;
        result.map(|_| ())
    }
}
