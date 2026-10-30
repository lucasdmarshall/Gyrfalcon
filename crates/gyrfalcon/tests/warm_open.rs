//! Opening a collection reads its mapped search files into the page cache, so the first query
//! does not fault them one page at a time.

use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

use gyrfalcon::{Document, Index, IndexConfig};

extern "C" {
    fn mmap(
        addr: *mut std::ffi::c_void,
        length: usize,
        prot: i32,
        flags: i32,
        fd: i32,
        offset: i64,
    ) -> *mut std::ffi::c_void;
    fn munmap(addr: *mut std::ffi::c_void, length: usize) -> i32;
    fn mincore(addr: *mut std::ffi::c_void, length: usize, vec: *mut u8) -> i32;
    fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
    fn sysconf(name: i32) -> i64;
}

const PROT_READ: i32 = 1;
const MAP_PRIVATE: i32 = 2;

/// Linux `_SC_PAGESIZE`.
const SC_PAGESIZE: i32 = 30;
/// Linux `POSIX_FADV_DONTNEED`.
const DONTNEED: i32 = 4;

fn page_size() -> usize {
    let size = unsafe { sysconf(SC_PAGESIZE) };
    assert!(size > 0);
    size as usize
}

fn evict(path: &Path) {
    let file = File::open(path).unwrap();
    let rc = unsafe { posix_fadvise(file.as_raw_fd(), 0, 0, DONTNEED) };
    assert_eq!(rc, 0, "posix_fadvise on {}", path.display());
}

fn resident_pages(path: &Path) -> (usize, usize) {
    let file = File::open(path).unwrap();
    let size = file.metadata().unwrap().len() as usize;
    assert!(size > 0, "{}", path.display());
    // SAFETY: a private read-only map does not fault a page. The file stays open for the call.
    let map = unsafe { mmap(std::ptr::null_mut(), size, PROT_READ, MAP_PRIVATE, file.as_raw_fd(), 0) };
    assert_ne!(map, (-1isize) as *mut std::ffi::c_void, "mmap {}", path.display());
    let page = page_size();
    let pages = size.div_ceil(page);
    let mut vector = vec![0u8; pages];
    let rc = unsafe { mincore(map, size, vector.as_mut_ptr()) };
    assert_eq!(rc, 0, "mincore {}", path.display());
    unsafe { munmap(map, size) };
    (vector.iter().filter(|byte| *byte & 1 == 1).count(), pages)
}

fn files_named(root: &Path, name: &str) -> Vec<PathBuf> {
    let mut found = Vec::new();
    for entry in walkdir_files(root) {
        if entry.file_name().is_some_and(|file| file == name) {
            found.push(entry);
        }
    }
    found.sort();
    found
}

fn walkdir_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();
            if entry.file_type().unwrap().is_dir() {
                stack.push(path);
            } else {
                out.push(path);
            }
        }
    }
    out
}

#[test]
fn open_loads_search_files_into_the_page_cache() {
    let dir = tempfile::tempdir().unwrap();
    let config = IndexConfig::test_default();
    let mut index = Index::create(dir.path(), &config).unwrap();
    index.write_buffer = 0;
    let rows: Vec<_> = (0..80)
        .map(|i| {
            let mut vector = vec![0f32; 384];
            vector[i % 384] = 1.0;
            let body = format!("document {i} about retrieval {}", "word ".repeat(30));
            (Document::body(format!("d{i}"), body), vector)
        })
        .collect();
    index.index_documents_with_vectors(&rows).unwrap();
    drop(index);

    let watched = ["vectors.bin", "codes.bin", "postings.bin", "docs.bin", "norms.bin", "terms.dict"];
    let mut paths = Vec::new();
    for name in watched {
        let found = files_named(dir.path(), name);
        assert!(!found.is_empty(), "missing {name}");
        paths.extend(found);
    }
    for path in &paths {
        evict(path);
    }
    let cold_vectors = files_named(dir.path(), "vectors.bin");
    let (resident, pages) = resident_pages(&cold_vectors[0]);
    assert!(pages > 1, "vectors.bin has {pages} pages; the fixture must span more than one");
    assert!(resident < pages, "eviction left vectors.bin fully resident ({resident}/{pages})");

    Index::open(dir.path(), &config).unwrap();

    for path in &paths {
        let (resident, pages) = resident_pages(path);
        assert_eq!(resident, pages, "{} has {resident}/{pages} pages resident", path.display());
    }
}
