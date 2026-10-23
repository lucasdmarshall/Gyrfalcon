//! The dense leg: one USearch HNSW file per segment, searched with a live-id predicate.

use std::path::{Path, PathBuf};
use std::sync::RwLock;

use usearch::{b1x8, IndexOptions, MetricKind, ScalarKind};

use crate::error::{Error, Result};
use crate::manifest::Quantization;

pub const CONNECTIVITY: usize = 16;
pub const EXPANSION_ADD: usize = 128;
pub const EXPANSION_SEARCH: usize = 64;
/// Retries after the first `filtered_search` at 64.
pub const EXPANSION_RETRIES: [usize; 4] = [128, 256, 512, 1024];
/// Binary mode rescores this many live Hamming hits with cosine.
pub const BINARY_RESCORE_WIDTH: usize = 100;

/// Where HNSW stopped for one segment, or for a query at its deepest segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FallbackStep {
    E64,
    E128,
    E256,
    E512,
    E1024,
    Exact,
}

impl FallbackStep {
    pub const ALL: [FallbackStep; 6] =
        [FallbackStep::E64, FallbackStep::E128, FallbackStep::E256, FallbackStep::E512, FallbackStep::E1024, FallbackStep::Exact];

    /// The `fallback_pct` key.
    pub fn label(self) -> &'static str {
        match self {
            FallbackStep::E64 => "64",
            FallbackStep::E128 => "128",
            FallbackStep::E256 => "256",
            FallbackStep::E512 => "512",
            FallbackStep::E1024 => "1024",
            FallbackStep::Exact => "exact",
        }
    }

    fn for_expansion(expansion: usize) -> FallbackStep {
        match expansion {
            0..=64 => FallbackStep::E64,
            65..=128 => FallbackStep::E128,
            129..=256 => FallbackStep::E256,
            257..=512 => FallbackStep::E512,
            _ => FallbackStep::E1024,
        }
    }
}

fn vector_error(err: impl std::fmt::Display) -> Error {
    Error::Vector(err.to_string())
}

/// Quantizes a model output once. `i8` and `b1` L2-normalize and scale to `[-127, 127]`;
/// `f16` narrows; `f32` stores the values as given.
pub fn quantize(quantization: Quantization, vector: &[f32]) -> Vec<u8> {
    match quantization {
        Quantization::F32 => vector.iter().flat_map(|v| v.to_le_bytes()).collect(),
        Quantization::F16 => vector.iter().flat_map(|v| f32_to_f16(*v).to_le_bytes()).collect(),
        Quantization::I8 | Quantization::B1 => {
            let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
            let scale = if norm > 0.0 { 127.0 / norm } else { 0.0 };
            vector.iter().map(|v| ((v * scale).round().clamp(-127.0, 127.0) as i8) as u8).collect()
        }
    }
}

/// The stored row as `f32`, for the exact scan.
pub fn dequantize(quantization: Quantization, row: &[u8]) -> Vec<f32> {
    match quantization {
        Quantization::F32 => row.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect(),
        Quantization::F16 => row.chunks_exact(2).map(|b| f16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect(),
        Quantization::I8 | Quantization::B1 => row.iter().map(|b| *b as i8 as f32).collect(),
    }
}

fn as_i8(row: &[u8]) -> &[i8] {
    // SAFETY: i8 and u8 have the same size and alignment.
    unsafe { std::slice::from_raw_parts(row.as_ptr() as *const i8, row.len()) }
}

/// Binary-mode bits: a bit is set when the int8 byte is greater than 0.
fn to_bits(row: &[u8]) -> Vec<b1x8> {
    row.chunks(8)
        .map(|byte| {
            let mut bits = 0u8;
            for (i, value) in byte.iter().enumerate() {
                if (*value as i8) > 0 {
                    bits |= 1 << i;
                }
            }
            b1x8(bits)
        })
        .collect()
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if exponent == 0xff {
        return sign | 0x7c00 | if mantissa != 0 { 0x200 } else { 0 };
    }
    let exponent = exponent - 127 + 15;
    if exponent >= 0x1f {
        return sign | 0x7c00;
    }
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        let mantissa = (mantissa | 0x80_0000) >> (1 - exponent);
        return sign | ((mantissa + 0x1000) >> 13) as u16;
    }
    let half = sign as u32 | ((exponent as u32) << 10) | (mantissa >> 13);
    // Round to nearest; a carry into the exponent is still the right value.
    (half + ((mantissa >> 12) & 1)) as u16
}

fn f16_to_f32(half: u16) -> f32 {
    let sign = ((half & 0x8000) as u32) << 16;
    let exponent = ((half >> 10) & 0x1f) as u32;
    let mantissa = (half & 0x3ff) as u32;
    let bits = match (exponent, mantissa) {
        (0, 0) => sign,
        (0, _) => {
            let mut e = 127 - 15 + 1;
            let mut m = mantissa;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, _) => sign | 0x7f80_0000 | (mantissa << 13),
        _ => sign | ((exponent + 127 - 15) << 23) | (mantissa << 13),
    };
    f32::from_bits(bits)
}

fn options(quantization: Quantization, dimension: usize) -> IndexOptions {
    let (metric, scalar) = match quantization {
        Quantization::I8 => (MetricKind::Cos, ScalarKind::I8),
        Quantization::F16 => (MetricKind::Cos, ScalarKind::F16),
        Quantization::F32 => (MetricKind::Cos, ScalarKind::F32),
        Quantization::B1 => (MetricKind::Hamming, ScalarKind::B1),
    };
    IndexOptions {
        dimensions: dimension,
        metric,
        quantization: scalar,
        connectivity: CONNECTIVITY,
        expansion_add: EXPANSION_ADD,
        expansion_search: EXPANSION_SEARCH,
        multi: false,
    }
}

fn add_row(index: &usearch::Index, quantization: Quantization, key: u64, row: &[u8]) -> Result<()> {
    match quantization {
        Quantization::I8 => index.add::<i8>(key, as_i8(row)),
        Quantization::F16 | Quantization::F32 => index.add::<f32>(key, &dequantize(quantization, row)),
        Quantization::B1 => index.add::<b1x8>(key, &to_bits(row)),
    }
    .map_err(vector_error)
}

/// Builds and saves `usearch.index` from quantized rows. Adds run on all cores.
pub fn write_usearch(path: &Path, quantization: Quantization, dimension: usize, rows: &[(u64, &[u8])]) -> Result<()> {
    let index = usearch::Index::new(&options(quantization, dimension)).map_err(vector_error)?;
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(16);
    index.reserve_capacity_and_threads(rows.len().max(1), threads).map_err(vector_error)?;
    let per_thread = rows.len().div_ceil(threads).max(1);
    std::thread::scope(|scope| -> Result<()> {
        let handles: Vec<_> = rows
            .chunks(per_thread)
            .map(|part| {
                let index = &index;
                scope.spawn(move || -> Result<()> {
                    for (key, row) in part {
                        add_row(index, quantization, *key, row)?;
                    }
                    Ok(())
                })
            })
            .collect();
        for handle in handles {
            handle.join().map_err(|_| vector_error("usearch add thread panicked"))??;
        }
        Ok(())
    })?;
    index.save(&path.to_string_lossy()).map_err(vector_error)?;
    Ok(())
}

/// The query in every form a segment search needs.
pub struct DenseQuery {
    pub quantization: Quantization,
    pub raw: Vec<f32>,
    pub row: Vec<u8>,
    /// Test hook: every `filtered_search` returns nothing, forcing the exact scan.
    pub starve_hnsw: bool,
}

impl DenseQuery {
    pub fn new(quantization: Quantization, raw: &[f32]) -> DenseQuery {
        DenseQuery { quantization, raw: raw.to_vec(), row: quantize(quantization, raw), starve_hnsw: false }
    }
}

pub struct VectorHits {
    /// `(chunk_id, cosine distance)` ascending by distance.
    pub hits: Vec<(u64, f32)>,
    pub step: FallbackStep,
}

/// A sealed segment's USearch file, viewed from disk.
pub struct VectorSegment {
    index: RwLock<usearch::Index>,
    quantization: Quantization,
    dimension: usize,
    vectors_path: PathBuf,
    /// `vectors.bin`, mapped only in binary mode for the cosine rescore.
    mapped: Option<memmap2::Mmap>,
}

impl VectorSegment {
    pub fn open(dir: &Path, quantization: Quantization, dimension: usize) -> Result<VectorSegment> {
        let path = dir.join("usearch.index");
        let index = usearch::Index::restore_view(&path.to_string_lossy()).map_err(vector_error)?;
        index.change_expansion_search(EXPANSION_SEARCH);
        let vectors_path = dir.join("vectors.bin");
        let mapped = if quantization == Quantization::B1 {
            let file = std::fs::File::open(&vectors_path)?;
            // SAFETY: sealed files never change; the reaper waits for readers to release.
            Some(unsafe { memmap2::Mmap::map(&file)? })
        } else {
            None
        };
        Ok(VectorSegment { index: RwLock::new(index), quantization, dimension, vectors_path, mapped })
    }

    /// Files this segment keeps mapped for search.
    pub fn mapped_paths(&self, dir: &Path) -> Vec<PathBuf> {
        let mut out = vec![dir.join("usearch.index")];
        if self.mapped.is_some() {
            out.push(self.vectors_path.clone());
        }
        out
    }

    fn filtered(&self, index: &usearch::Index, query: &DenseQuery, count: usize, live: &dyn Fn(u64) -> bool) -> Result<Vec<(u64, f32)>> {
        if query.starve_hnsw {
            return Ok(Vec::new());
        }
        let matches = match self.quantization {
            Quantization::I8 => index.filtered_search::<i8, _>(as_i8(&query.row), count, |key| live(key)),
            Quantization::F16 | Quantization::F32 => index.filtered_search::<f32, _>(&query.raw, count, |key| live(key)),
            Quantization::B1 => index.filtered_search::<b1x8, _>(&to_bits(&query.row), count, |key| live(key)),
        }
        .map_err(vector_error)?;
        Ok(matches.keys.into_iter().zip(matches.distances).collect())
    }

    /// Searches with `live` inside the graph walk. Retries at expansion 128..1024 while the
    /// live count is short, then scans `vectors.bin`. `ids` are the segment's stored ids in
    /// `vectors.bin` order and `live_count` is how many of them are live.
    pub fn search(&self, query: &DenseQuery, limit: usize, live: &dyn Fn(u64) -> bool, ids: &[u64], live_count: usize) -> Result<VectorHits> {
        let want = limit.min(live_count);
        if want == 0 {
            return Ok(VectorHits { hits: Vec::new(), step: FallbackStep::E64 });
        }
        let count = if self.quantization == Quantization::B1 { want.max(BINARY_RESCORE_WIDTH) } else { want };
        let target = count.min(live_count);
        let mut hits = {
            let index = self.index.read().unwrap_or_else(|p| p.into_inner());
            self.filtered(&index, query, count, live)?
        };
        let mut step = FallbackStep::E64;
        if hits.len() < target {
            let index = self.index.write().unwrap_or_else(|p| p.into_inner());
            for expansion in EXPANSION_RETRIES {
                index.change_expansion_search(expansion);
                hits = self.filtered(&index, query, count, live)?;
                step = FallbackStep::for_expansion(expansion);
                if hits.len() >= target {
                    break;
                }
            }
            index.change_expansion_search(EXPANSION_SEARCH);
        }
        if hits.len() < target {
            hits = self.exact(query, count, live, ids)?;
            step = FallbackStep::Exact;
        } else if self.quantization == Quantization::B1 {
            hits = self.rescore(query, hits, ids);
        }
        hits.truncate(want);
        Ok(VectorHits { hits, step })
    }

    /// Cosine-rescores binary Hamming hits against the mapped int8 rows.
    fn rescore(&self, query: &DenseQuery, hits: Vec<(u64, f32)>, ids: &[u64]) -> Vec<(u64, f32)> {
        use simsimd::SpatialSimilarity;
        let Some(mapped) = &self.mapped else { return hits };
        let row_len = self.dimension;
        let mut rescored: Vec<(u64, f32)> = hits
            .into_iter()
            .filter_map(|(key, _)| {
                let row = ids.binary_search(&key).ok()?;
                let stored = as_i8(&mapped[row * row_len..(row + 1) * row_len]);
                let distance = i8::cos(as_i8(&query.row), stored)? as f32;
                Some((key, distance))
            })
            .collect();
        rescored.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        rescored
    }

    /// Reads `vectors.bin` and keeps the closest live ids by cosine distance.
    fn exact(&self, query: &DenseQuery, count: usize, live: &dyn Fn(u64) -> bool, ids: &[u64]) -> Result<Vec<(u64, f32)>> {
        let bytes = std::fs::read(&self.vectors_path)?;
        let row_len = self.dimension * self.quantization.bytes_per_component();
        let query_vector = match self.quantization {
            Quantization::I8 | Quantization::B1 => dequantize(self.quantization, &query.row),
            _ => query.raw.clone(),
        };
        let mut hits: Vec<(u64, f32)> = ids
            .iter()
            .enumerate()
            .filter(|(_, id)| live(**id))
            .map(|(row, id)| {
                let stored = dequantize(self.quantization, &bytes[row * row_len..(row + 1) * row_len]);
                (*id, cosine_distance(&query_vector, &stored))
            })
            .collect();
        hits.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        hits.truncate(count);
        Ok(hits)
    }
}

pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        1.0
    } else {
        1.0 - dot / (na * nb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trips_common_values() {
        for value in [0.0f32, 1.0, -1.0, 0.5, 0.333, 65504.0, -0.0001] {
            let back = f16_to_f32(f32_to_f16(value));
            assert!((back - value).abs() <= value.abs() * 1e-3 + 1e-4, "{value} -> {back}");
        }
    }

    #[test]
    fn int8_scales_to_127() {
        let row = quantize(Quantization::I8, &[3.0, 4.0]);
        assert_eq!(as_i8(&row), &[76, 102]);
        let row = quantize(Quantization::I8, &[1.0, 0.0]);
        assert_eq!(as_i8(&row), &[127, 0]);
    }
}
