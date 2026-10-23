//! Reader epochs and the reaper that deletes retired segment directories.
//!
//! One process owns the collection, so the refcount is in-process only.

use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
}

#[derive(Default)]
pub struct Reaper {
    refcount: BTreeMap<u64, usize>,
    retired: Vec<Retired>,
    pub deleter: FsDeleter,
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
        self.retired.push(Retired { dir, name, retired_at });
    }

    pub fn retired_dirs(&self) -> Vec<PathBuf> {
        self.retired.iter().map(|r| r.dir.clone()).collect()
    }

    fn readers_before(&self, epoch: u64) -> bool {
        self.refcount.range(..epoch).any(|(_, count)| *count > 0)
    }

    /// Deletes each queued directory whose older readers are gone and that the current
    /// manifest does not name. Failures stay queued.
    pub fn poll(&mut self, fs: &mut dyn FileDeleter, current: &[String]) {
        let queued = std::mem::take(&mut self.retired);
        for retired in queued {
            let blocked = self.readers_before(retired.retired_at) || current.contains(&retired.name);
            if blocked || fs.remove(&retired.dir).is_err() {
                self.retired.push(retired);
            }
        }
    }

    /// `poll` with the reaper's own deleter.
    pub fn poll_default(&mut self, current: &[String]) {
        let mut deleter = std::mem::take(&mut self.deleter);
        self.poll(&mut deleter, current);
        self.deleter = deleter;
    }
}

/// Holds one epoch's snapshot. Dropping it releases the epoch and runs the reaper.
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
        self.shared.reaper().release(self.snapshot.manifest.epoch);
        self.shared.reap();
    }
}
