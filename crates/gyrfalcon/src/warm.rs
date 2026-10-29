//! Brings a mapped file into RAM before the first query.
//!
//! Search faults `codes.bin`, `vectors.bin`, the lexical files, and the static embedding
//! table as it touches them. Touching each mapping in order at open moves that wait off the
//! query. The touch goes through the mapping search already holds: opening the file again
//! would drop a FUSE page cache, and the query would fault the same pages from disk.

use std::io::Read;
use std::path::Path;

/// When mapped search files are read into the page cache.
///
/// `Eager` finishes the read before open returns. `None` leaves it to the query, which reads
/// exact rows while pages are missing. `Background` returns immediately and reads on another
/// thread; a query that arrives early uses the same exact-row path until the pages are resident.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Warmup {
    None,
    Background,
    Eager,
}

impl Warmup {
    pub fn parse(text: &str) -> Option<Warmup> {
        match text {
            "none" => Some(Warmup::None),
            "background" => Some(Warmup::Background),
            "eager" => Some(Warmup::Eager),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Warmup::None => "none",
            Warmup::Background => "background",
            Warmup::Eager => "eager",
        }
    }
}

impl Default for Warmup {
    /// The library open reads the mapped files before returning. The CLI, Python, and HTTP
    /// default is the same until the open-time measurement picks one.
    fn default() -> Self {
        Warmup::Eager
    }
}

/// Bytes per `read` for a file that is not already mapped. A mebibyte keeps the scan sequential.
const CHUNK: usize = 1024 * 1024;
const PAGE: usize = 4096;

/// Copies `path` through a buffer and drops the copy. The kernel keeps the file's pages.
/// Prefer [`warm_map`] when the file is already mapped: a second open drops those pages on FUSE.
pub fn warm_file(path: &Path) -> std::io::Result<()> {
    let mut file = std::fs::File::open(path)?;
    let mut buf = vec![0u8; CHUNK];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        std::hint::black_box(&buf[..n]);
    }
    Ok(())
}

/// Faults every page of `map` in order. Sequential faults let the kernel readahead.
pub fn warm_map(map: &memmap2::Mmap) {
    let _ = map.advise(memmap2::Advice::Sequential);
    let _ = map.advise(memmap2::Advice::WillNeed);
    warm_bytes(map);
}

/// Reads one byte per page so a mapping is resident even where `MADV_WILLNEED` returns early.
pub fn warm_bytes(bytes: &[u8]) {
    let mut acc = 0u64;
    for page in bytes.chunks(PAGE) {
        acc = acc.wrapping_add(u64::from(page[0]));
    }
    std::hint::black_box(acc);
}

/// Reads in flight at once. A cloud disk finishes an 800-read query at its IOPS cap once a
/// few dozen are queued; more threads would not shorten that, and fewer would leave the
/// queue empty.
const READ_QUEUE: usize = 32;

extern "C" {
    fn mincore(addr: *mut std::ffi::c_void, length: usize, vec: *mut u8) -> i32;
    /// Linux `_SC_PAGESIZE`.
    fn sysconf(name: i32) -> i64;
}

fn page_size() -> usize {
    let size = unsafe { sysconf(30) };
    if size > 0 {
        size as usize
    } else {
        PAGE
    }
}

/// `true` when every byte of `ranges` (offset, length) is in the page cache. An empty list is
/// resident. `mincore` failing counts as not resident, so the caller reads the bytes itself.
pub fn ranges_resident(map: &memmap2::Mmap, ranges: &[(usize, usize)]) -> bool {
    if ranges.is_empty() {
        return true;
    }
    let pages = resident_pages(map);
    let Some(pages) = pages else { return false };
    let page = page_size();
    ranges.iter().all(|&(start, len)| {
        if len == 0 {
            return true;
        }
        let Some(end) = start.checked_add(len) else { return false };
        if end > map.len() {
            return false;
        }
        let last = (end - 1) / page;
        (start / page..=last).all(|index| pages.get(index).is_some_and(|byte| byte & 1 == 1))
    })
}

fn resident_pages(map: &memmap2::Mmap) -> Option<Vec<u8>> {
    if map.is_empty() {
        return Some(Vec::new());
    }
    let pages = map.len().div_ceil(page_size());
    let mut vector = vec![0u8; pages];
    let rc = unsafe { mincore(map.as_ptr() as *mut std::ffi::c_void, map.len(), vector.as_mut_ptr()) };
    (rc == 0).then_some(vector)
}

/// Reads `(offset, length)` pairs from `file`. Separate threads issue the reads so a cold
/// disk can queue them; each buffer is exactly the requested bytes, with no neighbor pages.
pub fn read_ranges(file: &std::fs::File, ranges: &[(u64, usize)]) -> std::io::Result<Vec<Vec<u8>>> {
    use std::os::unix::fs::FileExt;
    let mut buffers: Vec<Vec<u8>> = ranges.iter().map(|(_, len)| vec![0u8; *len]).collect();
    if ranges.is_empty() {
        return Ok(buffers);
    }
    let queue = READ_QUEUE.min(ranges.len());
    let chunk = ranges.len().div_ceil(queue);
    std::thread::scope(|scope| -> std::io::Result<()> {
        let mut guards = Vec::new();
        for (group, bufs) in ranges.chunks(chunk).zip(buffers.chunks_mut(chunk)) {
            guards.push(scope.spawn(move || -> std::io::Result<()> {
                for ((offset, _), buf) in group.iter().zip(bufs.iter_mut()) {
                    file.read_exact_at(buf, *offset)?;
                }
                Ok(())
            }));
        }
        for guard in guards {
            match guard.join() {
                Ok(result) => result?,
                Err(_) => return Err(std::io::Error::other("a range read panicked")),
            }
        }
        Ok(())
    })?;
    Ok(buffers)
}

#[cfg(test)]
mod tests {
    use super::{read_ranges, Warmup};

    #[test]
    fn read_ranges_returns_the_bytes_at_each_offset() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rows.bin");
        let bytes: Vec<u8> = (0..1000u32).flat_map(|i| i.to_le_bytes()).collect();
        std::fs::write(&path, &bytes).unwrap();
        let file = std::fs::File::open(&path).unwrap();
        let ranges = [(0u64, 4usize), (40, 8), (996, 4)];
        let got = read_ranges(&file, &ranges).unwrap();
        for ((offset, len), buf) in ranges.iter().zip(&got) {
            assert_eq!(buf, &bytes[*offset as usize..*offset as usize + *len]);
        }
    }

    #[test]
    fn warmup_names_round_trip() {
        for mode in [Warmup::None, Warmup::Background, Warmup::Eager] {
            assert_eq!(Warmup::parse(mode.name()), Some(mode));
        }
        assert_eq!(Warmup::parse("later"), None);
    }
}
