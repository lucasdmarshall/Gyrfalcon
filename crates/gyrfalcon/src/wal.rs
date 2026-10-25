//! The write-ahead log of buffered writes: `wal/{wal_generation}.log`.
//!
//! ```text
//! frame    len u32, crc32 u32 (of the payload), payload
//! payload  record count u32, records, dead count u32, dead chunk ids u64
//! record   chunk_id u64, byte_start u64, byte_end u64, then length-prefixed (u32) doc_id,
//!          body, prefix (u32::MAX for none), and the quantized vector
//! ```
//!
//! A write call appends one frame and fsyncs, so its new chunks and the chunk ids it kills
//! (an update's old chunks, a delete) become durable together. Replay stops at the first frame
//! that is short or fails its checksum, the tail a crash can leave, and truncates the file there.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

const NONE: u32 = u32::MAX;

/// One chunk as the log stores it; `prepare` recomputes its tokens on replay.
#[derive(Debug, Clone, PartialEq)]
pub struct WalChunk {
    pub chunk_id: u64,
    pub doc_id: String,
    pub byte_start: u64,
    pub byte_end: u64,
    pub body: String,
    pub prefix: Option<String>,
    pub vector: Vec<u8>,
}

/// One write call.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Frame {
    pub chunks: Vec<WalChunk>,
    pub dead: Vec<u64>,
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    out.extend_from_slice(&(bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(bytes);
}

impl Frame {
    fn encode(&self) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.extend_from_slice(&(self.chunks.len() as u32).to_le_bytes());
        for chunk in &self.chunks {
            for value in [chunk.chunk_id, chunk.byte_start, chunk.byte_end] {
                payload.extend_from_slice(&value.to_le_bytes());
            }
            put_bytes(&mut payload, chunk.doc_id.as_bytes());
            put_bytes(&mut payload, chunk.body.as_bytes());
            match &chunk.prefix {
                Some(prefix) => put_bytes(&mut payload, prefix.as_bytes()),
                None => payload.extend_from_slice(&NONE.to_le_bytes()),
            }
            put_bytes(&mut payload, &chunk.vector);
        }
        payload.extend_from_slice(&(self.dead.len() as u32).to_le_bytes());
        for id in &self.dead {
            payload.extend_from_slice(&id.to_le_bytes());
        }
        let mut out = Vec::with_capacity(8 + payload.len());
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(&crc32fast::hash(&payload).to_le_bytes());
        out.extend_from_slice(&payload);
        out
    }

    fn decode(payload: &[u8]) -> Option<Frame> {
        let mut at = 0usize;
        let mut take = |n: usize| -> Option<&[u8]> {
            let bytes = payload.get(at..at.checked_add(n)?)?;
            at += n;
            Some(bytes)
        };
        let u32_of = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
        let u64_of = |b: &[u8]| u64::from_le_bytes(b.try_into().unwrap());
        let mut frame = Frame::default();
        let count = u32_of(take(4)?);
        for _ in 0..count {
            let (chunk_id, byte_start, byte_end) = (u64_of(take(8)?), u64_of(take(8)?), u64_of(take(8)?));
            let mut text = |optional: bool| -> Option<Option<Vec<u8>>> {
                let len = u32_of(take(4)?);
                if optional && len == NONE {
                    return Some(None);
                }
                Some(Some(take(len as usize)?.to_vec()))
            };
            let doc_id = String::from_utf8(text(false)??).ok()?;
            let body = String::from_utf8(text(false)??).ok()?;
            let prefix = match text(true)? {
                Some(bytes) => Some(String::from_utf8(bytes).ok()?),
                None => None,
            };
            let vector = text(false)??;
            frame.chunks.push(WalChunk { chunk_id, doc_id, byte_start, byte_end, body, prefix, vector });
        }
        let dead = u32_of(take(4)?);
        for _ in 0..dead {
            frame.dead.push(u64_of(take(8)?));
        }
        (at == payload.len()).then_some(frame)
    }
}

pub fn wal_path(root: &Path, generation: u64) -> PathBuf {
    root.join("wal").join(format!("{generation}.log"))
}

/// Reads every whole frame of `path`. A missing file is an empty log. A torn or corrupt tail
/// is cut off so the next append starts at a frame boundary.
pub fn replay(path: &Path) -> Result<Vec<Frame>> {
    let bytes = match std::fs::read(path) {
        Ok(bytes) => bytes,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(err) => return Err(err.into()),
    };
    let mut frames = Vec::new();
    let mut at = 0;
    while at + 8 <= bytes.len() {
        let len = u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        let crc = u32::from_le_bytes(bytes[at + 4..at + 8].try_into().unwrap());
        let Some(payload) = bytes.get(at + 8..at + 8 + len) else { break };
        if crc32fast::hash(payload) != crc {
            break;
        }
        let Some(frame) = Frame::decode(payload) else { break };
        frames.push(frame);
        at += 8 + len;
    }
    if at < bytes.len() {
        let file = OpenOptions::new().write(true).open(path)?;
        file.set_len(at as u64)?;
        file.sync_all()?;
    }
    Ok(frames)
}

/// An open log, appended to by the writer.
pub struct Wal {
    file: File,
}

impl Wal {
    /// Opens (creating) the log at `path` for appending. The caller has replayed it first.
    pub fn open(path: &Path) -> Result<Wal> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let created = !path.exists();
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        if created {
            crate::store::sync_dir(path.parent().unwrap_or(Path::new(".")))?;
        }
        Ok(Wal { file })
    }

    /// Appends one frame and fsyncs it.
    pub fn append(&mut self, frame: &Frame) -> Result<()> {
        self.file.write_all(&frame.encode()).map_err(Error::from)?;
        self.file.sync_data()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(id: u64, dead: &[u64]) -> Frame {
        Frame {
            chunks: vec![WalChunk {
                chunk_id: id,
                doc_id: format!("doc-{id}"),
                byte_start: 1,
                byte_end: 9,
                body: "မြန်မာ body".into(),
                prefix: id.is_multiple_of(2).then(|| "prefix".to_string()),
                vector: vec![1, 2, 255],
            }],
            dead: dead.to_vec(),
        }
    }

    #[test]
    fn frames_replay_in_order_and_a_torn_tail_is_cut() {
        let dir = tempfile::tempdir().unwrap();
        let path = wal_path(dir.path(), 3);
        let mut wal = Wal::open(&path).unwrap();
        let frames = vec![frame(1, &[]), frame(2, &[7, 8]), Frame { chunks: vec![], dead: vec![1] }];
        for f in &frames {
            wal.append(f).unwrap();
        }
        assert_eq!(replay(&path).unwrap(), frames);

        // A crash mid-append leaves part of a frame; replay keeps the whole ones.
        let whole = std::fs::read(&path).unwrap().len();
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.extend_from_slice(&frame(9, &[]).encode()[..20]);
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(replay(&path).unwrap(), frames);
        assert_eq!(std::fs::metadata(&path).unwrap().len() as usize, whole);

        // A flipped byte fails the checksum: that frame and everything after it are dropped.
        let mut bytes = std::fs::read(&path).unwrap();
        let second = frame(1, &[]).encode().len();
        bytes[second + 12] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();
        assert_eq!(replay(&path).unwrap(), frames[..1].to_vec());
    }

    #[test]
    fn a_missing_log_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert!(replay(&wal_path(dir.path(), 0)).unwrap().is_empty());
    }
}
