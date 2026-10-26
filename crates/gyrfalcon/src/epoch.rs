//! Reader epochs and the reaper that deletes retired segment directories.
//!
//! One process owns the collection, so the refcount is in-process only.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::thread::JoinHandle;

use crate::index::{Shared, Snapshot};

/// Deletes a retired directory. Any error leaves the path queued for the next poll.
pub trait FileDeleter {
    fn remove(&mut self, path: &Path) -> io::Result<()>;
}

/// `remove_dir_all`, failing the first `fail_times` calls with a Windows sharing violation
/// (raw os error 32) so tests can exercise the retry.
#[derive(Debug, Default)]
pub struct FsDeleter {
    pub fail_times: u32,
}

impl FileDeleter for FsDeleter {
    fn remove(&mut self, path: &Path) -> io::Result<()> {
        if self.fail_times > 0 {
            self.fail_times -= 1;
            return Err(io::Error::from_raw_os_error(32));
        }
        match std::fs::remove_dir_all(path) {
            Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
            other => other,
        }
    }
}

#[derive(Debug, Clone)]
struct Retired {
    dir: PathBuf,
    name: String,
    /// The first epoch whose manifest no longer names `dir`. Readers of any older epoch may
    /// still use it.
    retired_at: u64,
    /// A delete already failed; a release no longer schedules a retry, a publish or
    /// `reap_now` does.
    failed: bool,
}

#[derive(Default)]
pub struct Reaper {
    refcount: BTreeMap<u64, usize>,
    retired: Vec<Retired>,
    pub deleter: FsDeleter,
    /// The delete a reader's release started in the background, if any.
    background: Option<JoinHandle<()>>,
}

impl Reaper {
    pub fn acquire(&mut self, epoch: u64) {
        *self.refcount.entry(epoch).or_insert(0) += 1;
    }

    pub fn release(&mut self, epoch: u64) {
        if let Some(count) = self.refcount.get_mut(&epoch) {
            *count -= 1;
            if *count == 0 {
                self.refcount.remove(&epoch);
            }
        }
    }

    pub fn retire(&mut self, dir: PathBuf, name: String, retired_at: u64) {
        self.retired.push(Retired { dir, name, retired_at, failed: false });
    }

    pub fn retired_dirs(&self) -> Vec<PathBuf> {
        self.retired.iter().map(|r| r.dir.clone()).collect()
    }

    fn readers_before(&self, epoch: u64) -> bool {
        self.refcount.range(..epoch).any(|(_, count)| *count > 0)
    }

    fn deletable(&self, retired: &Retired, current: &[String]) -> bool {
        !self.readers_before(retired.retired_at) && !current.contains(&retired.name)
    }

    /// Deletes each queued directory whose older readers are gone and that the current
    /// manifest does not name. Failures stay queued.
    pub fn poll(&mut self, fs: &mut dyn FileDeleter, current: &[String]) {
        for mut retired in self.take_deletable(current) {
            if fs.remove(&retired.dir).is_err() {
                retired.failed = true;
                self.retired.push(retired);
            }
        }
    }

    /// Removes and returns the queued directories `poll` would delete.
    fn take_deletable(&mut self, current: &[String]) -> Vec<Retired> {
        let (ready, waiting) = std::mem::take(&mut self.retired).into_iter().partition(|r| self.deletable(r, current));
        self.retired = waiting;
        ready
    }

    /// Whether a directory became deletable that no delete has been tried on yet.
    fn has_new_deletable(&self, current: &[String]) -> bool {
        self.retired.iter().any(|r| !r.failed && self.deletable(r, current))
    }
}

/// Deletes the retired directories no reader needs. `reaper` is held only to pick them and to
/// requeue failures, never across a file-system call, so readers acquiring and releasing epochs
/// do not wait on a delete.
pub(crate) fn reap(reaper: &std::sync::Mutex<Reaper>, current: &[String]) {
    let lock = || reaper.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let (ready, mut deleter) = {
        let mut reaper = lock();
        let ready = reaper.take_deletable(current);
        // This batch takes its share of the injected test failures.
        let fail_times = reaper.deleter.fail_times.min(ready.len() as u32);
        reaper.deleter.fail_times -= fail_times;
        (ready, FsDeleter { fail_times })
    };
    let mut failed = Vec::new();
    for mut retired in ready {
        if deleter.remove(&retired.dir).is_err() {
            retired.failed = true;
            failed.push(retired);
        }
    }
    if !failed.is_empty() {
        lock().retired.extend(failed);
    }
}

/// Waits for the delete a release started in the background, if any.
pub(crate) fn join_background(reaper: &std::sync::Mutex<Reaper>) {
    let background = reaper.lock().unwrap_or_else(|poisoned| poisoned.into_inner()).background.take();
    if let Some(handle) = background {
        let _ = handle.join();
    }
}

/// Holds one epoch's snapshot. Dropping it releases the epoch; when that leaves a retired
/// directory deletable, the delete runs on a background thread, never on the reader's.
pub struct ReaderGuard {
    pub(crate) shared: Arc<Shared>,
    pub(crate) snapshot: Arc<Snapshot>,
}

impl ReaderGuard {
    pub fn epoch(&self) -> u64 {
        self.snapshot.manifest.epoch
    }

    pub(crate) fn snapshot(&self) -> &Snapshot {
        &self.snapshot
    }
}

impl Drop for ReaderGuard {
    fn drop(&mut self) {
        // Read the current segments before taking the reaper lock: `acquire` takes the state
        // lock first.
        let current = self.shared.snapshot().manifest.sealed_segments.clone();
        let mut reaper = self.shared.reaper();
        reaper.release(self.snapshot.manifest.epoch);
        let idle = reaper.background.as_ref().is_none_or(|handle| handle.is_finished());
        if idle && reaper.has_new_deletable(&current) {
            let shared = self.shared.clone();
            reaper.background = Some(std::thread::spawn(move || shared.reap()));
        }
    }
}
