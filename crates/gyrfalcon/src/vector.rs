//! The dense leg: a graph-free index per segment. Search is a three-stage cascade over every
//! row, so a seal builds no graph and tombstones need no retry.
//!
//! ```text
//! codes.bin   magic "GYRCODE1", rows u32, words u32, dimension u32, rounds u32, seed u64,
//!             the segment centroid (`dimension` f32, padded to 8 bytes), `words` u64 of sign
//!             bits per row, then four f32 per row (see `RowStats`)
//! vectors.bin the quantized rows (see `quantize`)
//! ```
//!
//! This is RaBitQ (Gao and Long, SIGMOD 2024) without the IVF partitions. A row is
//! dequantized, centered on the segment's mean, rotated by a randomized Hadamard transform
//! (padded to a power of two), and stored as the sign bits of that residual plus the scalars
//! that turn a bit-level inner product into an estimated cosine. A query then runs:
//!
//! 1. Hamming distance from its own sign bits to every live row; the closest
//!    `PREFILTER_PER_HIT * limit` rows survive (a histogram cut, no sort).
//! 2. RaBitQ's asymmetric estimate of the cosine for those rows, from the query quantized to
//!    four bit planes; the best `REFINE_PER_HIT * limit` survive.
//! 3. The exact cosine against the stored rows.
//!
//! The rotation spreads every embedding dimension over all bits, and centering removes the
//! direction all embeddings of one model share, which a sign bit cannot see. Against an exact
//! scan, recall@50 was 0.996 to 1.000 on SciFact, NFCorpus, and FiQA with both
//! `potion-retrieval-32M` and `bge-small-en-v1.5`; USearch HNSW (M 16, ef 128/64) reached 0.94
//! to 0.985 and took 4.5 s to build FiQA's 57,638 rows (`examples/profile_dense`).

use std::path::{Path, PathBuf};

use memmap2::Mmap;

use crate::error::{Error, Result};
use crate::manifest::Quantization;
#[cfg(test)]
use crate::store::Tombstones;
use crate::store::{Excluded, RowMask};

/// Rows kept by the Hamming prefilter, per requested hit, and at least `PREFILTER_MIN`.
pub const PREFILTER_PER_HIT: usize = 80;
pub const PREFILTER_MIN: usize = 4000;
/// Rows kept by the 4-bit estimate for the exact rescore, per requested hit.
pub const REFINE_PER_HIT: usize = 16;
pub const REFINE_MIN: usize = 800;
/// The smallest prefilter a segment with enough rows runs, however small its share.
pub const PREFILTER_SEGMENT_MIN: usize = 1000;

const CODES_MAGIC: &[u8; 8] = b"GYRCODE1";
const CODES_HEADER: usize = 32;
const ROTATION_ROUNDS: u32 = 1;
const ROTATION_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// How deep a dense search had to go. The graph-free cascade always answers in one pass, so
/// it reports `E64`; the other steps remain for the bench report's `fallback_pct` shape.
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
    pub const ALL: [FallbackStep; 6] = [
        FallbackStep::E64,
        FallbackStep::E128,
        FallbackStep::E256,
        FallbackStep::E512,
        FallbackStep::E1024,
        FallbackStep::Exact,
    ];

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
        Quantization::F32 => row.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
        Quantization::F16 => row.as_chunks::<2>().0.iter().map(|b| f16_to_f32(u16::from_le_bytes(*b))).collect(),
        Quantization::I8 | Quantization::B1 => row.iter().map(|b| *b as i8 as f32).collect(),
    }
}

fn as_i8(row: &[u8]) -> &[i8] {
    // SAFETY: i8 and u8 have the same size and alignment.
    unsafe { std::slice::from_raw_parts(row.as_ptr() as *const i8, row.len()) }
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

/// The row as `f32` in the scale it is stored; binary mode stores int8 rows.
pub(crate) fn row_values(quantization: Quantization, row: &[u8]) -> Vec<f32> {
    match quantization {
        Quantization::B1 => dequantize(Quantization::I8, row),
        other => dequantize(other, row),
    }
}

fn stored_quantization(quantization: Quantization) -> Quantization {
    if quantization == Quantization::B1 {
        Quantization::I8
    } else {
        quantization
    }
}

/// A randomized Hadamard rotation: per round, flip signs, then an orthonormal Walsh-Hadamard
/// transform. Inputs are zero-padded to `padded`, a power of two.
struct Rotation {
    signs: Vec<Vec<f32>>,
    padded: usize,
}

impl Rotation {
    fn new(dimension: usize, rounds: u32, seed: u64) -> Rotation {
        let padded = dimension.next_power_of_two().max(64);
        let mut state = seed;
        let signs = (0..rounds)
            .map(|_| {
                (0..padded)
                    .map(|_| {
                        state ^= state << 13;
                        state ^= state >> 7;
                        state ^= state << 17;
                        if state & 1 == 1 {
                            1.0
                        } else {
                            -1.0
                        }
                    })
                    .collect()
            })
            .collect();
        Rotation { signs, padded }
    }

    fn apply(&self, values: &[f32]) -> Vec<f32> {
        let mut v = vec![0f32; self.padded];
        v[..values.len()].copy_from_slice(values);
        let scale = 1.0 / (self.padded as f32).sqrt();
        for signs in &self.signs {
            for (value, sign) in v.iter_mut().zip(signs) {
                *value *= sign;
            }
            let mut half = 1;
            while half < self.padded {
                for block in (0..self.padded).step_by(2 * half) {
                    for i in block..block + half {
                        let (a, b) = (v[i], v[i + half]);
                        v[i] = a + b;
                        v[i + half] = a - b;
                    }
                }
                half *= 2;
            }
            v.iter_mut().for_each(|value| *value *= scale);
        }
        v
    }

    fn words(&self) -> usize {
        self.padded / 64
    }
}

fn sign_bits(rotated: &[f32], words: usize) -> Vec<u64> {
    let mut bits = vec![0u64; words];
    for (i, value) in rotated.iter().enumerate() {
        if *value > 0.0 {
            bits[i / 64] |= 1 << (i % 64);
        }
    }
    bits
}

/// Per-row scalars of the residual `r = row - centroid`, after rotation.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
struct RowStats {
    /// `<sign(r) / sqrt(P), r / |r|>`, which RaBitQ divides its estimate by; 1 for a zero `r`.
    factor: f32,
    /// `|r|`.
    residual_norm: f32,
    /// `<centroid, row>`.
    centroid_dot: f32,
    /// `|row|`, to turn the inner product into a cosine.
    norm: f32,
}

fn centroid_bytes(dimension: usize) -> usize {
    (dimension * 4).next_multiple_of(8)
}

/// Encodes `codes.bin` for quantized rows in ordinal order. Rows encode on every core.
pub fn encode_codes(quantization: Quantization, dimension: usize, rows: &[&[u8]]) -> Vec<u8> {
    let rotation = Rotation::new(dimension, ROTATION_ROUNDS, ROTATION_SEED);
    let words = rotation.words();
    let mut sum = vec![0f64; dimension];
    for row in rows {
        for (total, value) in sum.iter_mut().zip(row_values(quantization, row)) {
            *total += f64::from(value);
        }
    }
    let centroid: Vec<f32> = sum.iter().map(|total| (total / rows.len().max(1) as f64) as f32).collect();

    let encode = |part: &[&[u8]]| -> (Vec<u64>, Vec<RowStats>) {
        let mut codes = Vec::with_capacity(part.len() * words);
        let mut stats = Vec::with_capacity(part.len());
        for row in part {
            let values = row_values(quantization, row);
            let residual: Vec<f32> = values.iter().zip(&centroid).map(|(v, c)| v - c).collect();
            let rotated = rotation.apply(&residual);
            codes.extend(sign_bits(&rotated, words));
            let residual_norm = rotated.iter().map(|v| v * v).sum::<f32>().sqrt();
            let factor = if residual_norm == 0.0 {
                1.0
            } else {
                let spread = rotated.iter().map(|v| v.abs()).sum::<f32>();
                (spread / ((rotated.len() as f32).sqrt() * residual_norm)).max(1e-6)
            };
            stats.push(RowStats {
                factor,
                residual_norm,
                centroid_dot: values.iter().zip(&centroid).map(|(v, c)| v * c).sum(),
                norm: values.iter().map(|v| v * v).sum::<f32>().sqrt(),
            });
        }
        (codes, stats)
    };
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let parts: Vec<(Vec<u64>, Vec<RowStats>)> = if rows.len() < 4096 || threads == 1 {
        vec![encode(rows)]
    } else {
        std::thread::scope(|scope| {
            let handles: Vec<_> =
                rows.chunks(rows.len().div_ceil(threads)).map(|part| scope.spawn(move || encode(part))).collect();
            handles.into_iter().map(|h| h.join().expect("codes thread panicked")).collect()
        })
    };

    let mut out = Vec::with_capacity(CODES_HEADER + centroid_bytes(dimension) + rows.len() * (words * 8 + 16));
    out.extend_from_slice(CODES_MAGIC);
    out.extend_from_slice(&(rows.len() as u32).to_le_bytes());
    out.extend_from_slice(&(words as u32).to_le_bytes());
    out.extend_from_slice(&(dimension as u32).to_le_bytes());
    out.extend_from_slice(&ROTATION_ROUNDS.to_le_bytes());
    out.extend_from_slice(&ROTATION_SEED.to_le_bytes());
    for value in &centroid {
        out.extend_from_slice(&value.to_le_bytes());
    }
    out.resize(CODES_HEADER + centroid_bytes(dimension), 0);
    for (codes, _) in &parts {
        for word in codes {
            out.extend_from_slice(&word.to_le_bytes());
        }
    }
    for (_, stats) in &parts {
        for row in stats {
            for value in [row.factor, row.residual_norm, row.centroid_dot, row.norm] {
                out.extend_from_slice(&value.to_le_bytes());
            }
        }
    }
    out
}

/// The query quantized to 4-bit levels and stored as bit planes, so `<q, sign(r)>` is four
/// AND-popcounts per word.
struct PlaneQuery {
    planes: Vec<[u64; 4]>,
    /// `<q, c> - |c|^2`, which makes estimates comparable across segments.
    offset: f32,
    low: f32,
    step: f32,
    level_sum: f32,
    padded: f32,
}

impl PlaneQuery {
    fn new(rotated: &[f32]) -> PlaneQuery {
        let low = rotated.iter().copied().fold(f32::INFINITY, f32::min);
        let high = rotated.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let step = ((high - low) / 15.0).max(f32::MIN_POSITIVE);
        let mut planes = vec![[0u64; 4]; rotated.len() / 64];
        let mut level_sum = 0f32;
        for (i, value) in rotated.iter().enumerate() {
            let level = ((value - low) / step).round().clamp(0.0, 15.0) as u32;
            level_sum += level as f32;
            for (bit, plane) in planes[i / 64].iter_mut().enumerate() {
                if level >> bit & 1 == 1 {
                    *plane |= 1 << (i % 64);
                }
            }
        }
        PlaneQuery { planes, offset: 0.0, low, step, level_sum, padded: rotated.len() as f32 }
    }

    /// `sqrt(P) * <q, r / |r|>`, estimated from `r`'s sign bits and its correction factor.
    #[inline(always)]
    fn estimate(&self, code: &[u64], factor: f32) -> f32 {
        let mut ones = 0u32;
        let mut weighted = 0u32;
        for (word, planes) in code.iter().zip(&self.planes) {
            ones += word.count_ones();
            weighted += (word & planes[0]).count_ones()
                + 2 * (word & planes[1]).count_ones()
                + 4 * (word & planes[2]).count_ones()
                + 8 * (word & planes[3]).count_ones();
        }
        // sum (2b - 1)(low + step * level) over every dimension.
        let dot = 2.0 * self.low * ones as f32 - self.padded * self.low
            + self.step * (2.0 * weighted as f32 - self.level_sum);
        dot / factor
    }
}

/// The hot loops, as plain loops (no closures) so that each feature level below compiles them
/// with its own instructions.
mod kernels {
    use super::{PlaneQuery, RowStats};

    #[inline(always)]
    fn hamming_fixed<const W: usize>(codes: &[u64], query: &[u64], out: &mut [u16]) {
        let query: &[u64; W] = query.try_into().unwrap();
        let (rows, _) = codes.as_chunks::<W>();
        for (row, slot) in rows.iter().zip(out.iter_mut()) {
            let mut distance = 0u32;
            for i in 0..W {
                distance += (row[i] ^ query[i]).count_ones();
            }
            *slot = distance as u16;
        }
    }

    #[inline(always)]
    fn hamming_any(codes: &[u64], words: usize, query: &[u64], out: &mut [u16]) {
        for (row, slot) in codes.chunks_exact(words).zip(out.iter_mut()) {
            let mut distance = 0u32;
            for (a, b) in row.iter().zip(query) {
                distance += (a ^ b).count_ones();
            }
            *slot = distance as u16;
        }
    }

    /// Hamming distance from `query` to every row. Fixed word counts unroll; the common padded
    /// widths are 256 to 1024 bits.
    #[inline(always)]
    pub fn hamming(codes: &[u64], words: usize, query: &[u64], out: &mut [u16]) {
        match words {
            4 => hamming_fixed::<4>(codes, query, out),
            8 => hamming_fixed::<8>(codes, query, out),
            16 => hamming_fixed::<16>(codes, query, out),
            _ => hamming_any(codes, words, query, out),
        }
    }

    /// The estimated cosine of each row: `<q, row> = |r| <q - c, r / |r|> + <c, row> + <q, c> -
    /// |c|^2`, divided by `|row|`.
    #[inline(always)]
    pub fn estimate(
        codes: &[u64],
        stats: &[RowStats],
        words: usize,
        query: &PlaneQuery,
        rows: &[u32],
        out: &mut Vec<(u32, f32)>,
    ) {
        let root = query.padded.sqrt();
        for (index, &row) in rows.iter().enumerate() {
            if let Some(ahead) = rows.get(index + super::PREFETCH_AHEAD) {
                let ahead = *ahead as usize;
                let code = &codes[ahead * words..(ahead + 1) * words];
                // SAFETY: `u64` words viewed as bytes.
                super::prefetch(unsafe { std::slice::from_raw_parts(code.as_ptr() as *const u8, words * 8) });
            }
            let at = row as usize;
            let stat = stats[at];
            let residual = query.estimate(&codes[at * words..(at + 1) * words], stat.factor) / root;
            let inner = stat.residual_norm * residual + stat.centroid_dot + query.offset;
            out.push((row, if stat.norm > 0.0 { inner / stat.norm } else { f32::NEG_INFINITY }));
        }
    }

    #[inline(always)]
    pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
        let n = a.len().min(b.len());
        let (a, b) = (&a[..n], &b[..n]);
        let mut sum = 0i32;
        for i in 0..n {
            sum += i32::from(a[i]) * i32::from(b[i]);
        }
        sum
    }
}

type HammingFn = unsafe fn(&[u64], usize, &[u64], &mut [u16]);
type EstimateFn = unsafe fn(&[u64], &[RowStats], usize, &PlaneQuery, &[u32], &mut Vec<(u32, f32)>);
type DotFn = unsafe fn(&[i8], &[i8]) -> i32;

/// The kernels compiled for the best feature level this CPU has.
struct Kernels {
    hamming: HammingFn,
    estimate: EstimateFn,
    dot_i8: DotFn,
}

/// Wrappers that compile `kernels` with `$features` enabled.
macro_rules! feature_level {
    ($module:ident, $features:literal) => {
        #[cfg(target_arch = "x86_64")]
        mod $module {
            use super::{kernels, PlaneQuery, RowStats};

            #[target_feature(enable = $features)]
            pub unsafe fn hamming(codes: &[u64], words: usize, query: &[u64], out: &mut [u16]) {
                kernels::hamming(codes, words, query, out)
            }

            #[target_feature(enable = $features)]
            pub unsafe fn estimate(
                codes: &[u64],
                stats: &[RowStats],
                words: usize,
                query: &PlaneQuery,
                rows: &[u32],
                out: &mut Vec<(u32, f32)>,
            ) {
                kernels::estimate(codes, stats, words, query, rows, out)
            }

            #[target_feature(enable = $features)]
            pub unsafe fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
                kernels::dot_i8(a, b)
            }
        }
    };
}

feature_level!(x86_avx2, "avx2,popcnt");
feature_level!(x86_popcnt, "popcnt");

fn kernels() -> &'static Kernels {
    static KERNELS: std::sync::OnceLock<Kernels> = std::sync::OnceLock::new();
    KERNELS.get_or_init(|| {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("popcnt") {
                return Kernels { hamming: x86_avx2::hamming, estimate: x86_avx2::estimate, dot_i8: x86_avx2::dot_i8 };
            }
            if std::arch::is_x86_feature_detected!("popcnt") {
                return Kernels {
                    hamming: x86_popcnt::hamming,
                    estimate: x86_popcnt::estimate,
                    dot_i8: x86_popcnt::dot_i8,
                };
            }
        }
        Kernels { hamming: kernels::hamming, estimate: kernels::estimate, dot_i8: kernels::dot_i8 }
    })
}

/// The query in every form a segment search needs.
pub struct DenseQuery {
    pub quantization: Quantization,
    pub raw: Vec<f32>,
    pub row: Vec<u8>,
}

impl DenseQuery {
    pub fn new(quantization: Quantization, raw: &[f32]) -> DenseQuery {
        DenseQuery { quantization, raw: raw.to_vec(), row: quantize(quantization, raw) }
    }
}

pub struct VectorHits {
    /// `(chunk_id, cosine distance)` ascending by distance, then chunk id.
    pub hits: Vec<(u64, f32)>,
    pub step: FallbackStep,
}

fn corrupt(what: &str) -> Error {
    Error::SegmentCorrupt(format!("dense: {what}"))
}

/// A sealed segment's dense index: `codes.bin` and `vectors.bin`, memory-mapped.
pub struct VectorSegment {
    codes: Mmap,
    vectors: Mmap,
    /// Same file as `vectors`, kept so a cold query can read the exact rows without opening
    /// the path again. A second open drops the page cache on FUSE.
    vectors_file: std::fs::File,
    rows: usize,
    words: usize,
    rotation: Rotation,
    quantization: Quantization,
    dimension: usize,
    dir: PathBuf,
}

impl VectorSegment {
    pub fn open(dir: &Path, quantization: Quantization, dimension: usize) -> Result<VectorSegment> {
        if cfg!(target_endian = "big") {
            return Err(Error::Vector("codes.bin is little-endian; big-endian hosts are not supported".into()));
        }
        let map = |name: &str| -> Result<Mmap> {
            let file = std::fs::File::open(dir.join(name))?;
            // SAFETY: sealed files never change; the reaper waits for readers to release.
            Ok(unsafe { Mmap::map(&file)? })
        };
        let codes = map("codes.bin")?;
        let vectors_file = std::fs::File::open(dir.join("vectors.bin"))?;
        // SAFETY: sealed files never change; the reaper waits for readers to release.
        let vectors = unsafe { Mmap::map(&vectors_file)? };
        if codes.len() < CODES_HEADER || &codes[..8] != CODES_MAGIC {
            return Err(corrupt("codes.bin has no header"));
        }
        let u32_at = |at: usize| u32::from_le_bytes(codes[at..at + 4].try_into().unwrap());
        let (rows, words, stored_dimension, rounds) =
            (u32_at(8) as usize, u32_at(12) as usize, u32_at(16) as usize, u32_at(20));
        let seed = u64::from_le_bytes(codes[24..32].try_into().unwrap());
        let rotation = Rotation::new(dimension, rounds, seed);
        let row_len = dimension * stored_quantization(quantization).bytes_per_component();
        if stored_dimension != dimension || words != rotation.words() || rounds > 8 {
            return Err(corrupt("codes.bin shape disagrees with the segment"));
        }
        if codes.len() != CODES_HEADER + centroid_bytes(dimension) + rows * (words * 8 + 16)
            || vectors.len() != rows * row_len
        {
            return Err(corrupt("codes.bin or vectors.bin length disagrees with the row count"));
        }
        if !(codes.as_ptr() as usize + CODES_HEADER).is_multiple_of(8) {
            return Err(corrupt("codes.bin is not mapped at an 8-byte boundary"));
        }
        Ok(VectorSegment {
            codes,
            vectors,
            vectors_file,
            rows,
            words,
            rotation,
            quantization,
            dimension,
            dir: dir.to_path_buf(),
        })
    }

    fn centroid(&self) -> &[f32] {
        // SAFETY: `open` checked the file length and its 8-byte alignment; the file is
        // little-endian like the host.
        unsafe { std::slice::from_raw_parts(self.codes.as_ptr().add(CODES_HEADER) as *const f32, self.dimension) }
    }

    fn code_words(&self) -> &[u64] {
        let at = CODES_HEADER + centroid_bytes(self.dimension);
        // SAFETY: as in `centroid`; the codes start 8-byte aligned.
        unsafe { std::slice::from_raw_parts(self.codes.as_ptr().add(at) as *const u64, self.rows * self.words) }
    }

    fn stats(&self) -> &[RowStats] {
        let at = CODES_HEADER + centroid_bytes(self.dimension) + self.rows * self.words * 8;
        // SAFETY: as in `centroid`; `RowStats` is four `f32` with C layout.
        unsafe { std::slice::from_raw_parts(self.codes.as_ptr().add(at) as *const RowStats, self.rows) }
    }

    fn row_len(&self) -> usize {
        self.vectors.len() / self.rows.max(1)
    }

    fn row(&self, ordinal: usize) -> &[u8] {
        let len = self.row_len();
        &self.vectors[ordinal * len..(ordinal + 1) * len]
    }

    /// The stored rows are already in RAM, so the mapped bytes can be scored directly.
    fn rows_resident(&self, rows: &[u32]) -> bool {
        let len = self.row_len();
        let ranges: Vec<(usize, usize)> = rows.iter().map(|row| (*row as usize * len, len)).collect();
        crate::warm::ranges_resident(&self.vectors, &ranges)
    }

    /// The stored vector of `key` as `f32`, scaled as stored.
    pub fn stored_vector(&self, key: u64, ids: &[u64]) -> Option<Vec<f32>> {
        let ordinal = ids.binary_search(&key).ok()?;
        (ordinal < self.rows).then(|| row_values(self.quantization, self.row(ordinal)))
    }

    /// Faults `codes.bin` and `vectors.bin` in order. A query scans every code and then a
    /// scattered set of vector rows; both have to be resident for that query to stay in RAM.
    pub(crate) fn warm(&self) {
        crate::warm::warm_map(&self.codes);
        crate::warm::warm_map(&self.vectors);
        // The rows a query rescores are scattered. After they are resident, later faults
        // (a collection that no longer fits) should not pull the neighbors in again.
        let _ = self.vectors.advise(memmap2::Advice::Random);
    }

    /// Files this segment keeps mapped for search.
    pub fn mapped_paths(&self) -> Vec<PathBuf> {
        vec![self.dir.join("codes.bin"), self.dir.join("vectors.bin")]
    }

    fn exact_distance(&self, query: &DenseQuery, query_norm: f32, ordinal: usize) -> f32 {
        exact_distance(self.quantization, query, query_norm, self.row(ordinal), self.stats()[ordinal].norm)
    }

    /// Stages 1 and 2: up to `budget.refine` live rows with their estimated cosine, which is
    /// comparable across segments. `ids` are the segment's stored ids in row order and
    /// `live_count` is how many of them are live.
    pub fn candidates(
        &self,
        query: &DenseQuery,
        budget: Budget,
        dead: &impl Excluded,
        ids: &[u64],
        live_count: usize,
    ) -> Result<Vec<(u32, f32)>> {
        if budget.refine == 0 || live_count == 0 || self.rows == 0 {
            return Ok(Vec::new());
        }
        if ids.len() != self.rows || query.raw.len() != self.dimension {
            return Err(Error::Vector("query or ids disagree with the segment".into()));
        }
        let mask = dead.rows(ids);

        // The query in the rows' scale, centered like them.
        let values = row_values(self.quantization, &query.row);
        let centroid = self.centroid();
        let residual: Vec<f32> = values.iter().zip(centroid).map(|(v, c)| v - c).collect();
        let rotated = self.rotation.apply(&residual);
        let survivors: Vec<u32> = match &mask {
            // Few enough searchable rows: every one goes on to the estimate.
            RowMask::Only(rows) if rows.len() <= budget.prefilter => rows.iter().map(|&r| r as u32).collect(),
            RowMask::Except(dead_rows) if live_count <= budget.prefilter => {
                (0..self.rows).filter(|r| dead_rows.binary_search(r).is_err()).map(|r| r as u32).collect()
            }
            _ => {
                let bits = sign_bits(&rotated, self.words);
                let mut distances = vec![0u16; self.rows];
                // SAFETY: `kernels()` only returns functions this CPU supports.
                unsafe { (kernels().hamming)(self.code_words(), self.words, &bits, &mut distances) };
                match &mask {
                    RowMask::Except(dead_rows) => {
                        for row in dead_rows {
                            distances[*row] = u16::MAX;
                        }
                    }
                    RowMask::Only(rows) => {
                        let mut allowed = rows.iter().peekable();
                        for (row, distance) in distances.iter_mut().enumerate() {
                            if allowed.peek() == Some(&&row) {
                                allowed.next();
                            } else {
                                *distance = u16::MAX;
                            }
                        }
                    }
                }
                cut(&distances, self.words * 64, budget.prefilter)
            }
        };
        let mut planes = PlaneQuery::new(&rotated);
        planes.offset =
            values.iter().zip(centroid).map(|(v, c)| v * c).sum::<f32>() - centroid.iter().map(|c| c * c).sum::<f32>();
        let mut estimated = Vec::with_capacity(survivors.len());
        // SAFETY: `kernels()` only returns functions this CPU supports.
        unsafe {
            (kernels().estimate)(self.code_words(), self.stats(), self.words, &planes, &survivors, &mut estimated)
        };
        if estimated.len() > budget.refine {
            estimated.select_nth_unstable_by(budget.refine - 1, |a, b| b.1.total_cmp(&a.1));
            estimated.truncate(budget.refine);
        }
        Ok(estimated)
    }

    /// Stage 3: the exact cosine distance of `rows`, as `(chunk_id, distance)`.
    pub fn exact(&self, query: &DenseQuery, rows: &mut [u32], ids: &[u64]) -> Vec<(u64, f32)> {
        // Ascending rows read the mapped vectors in file order, which the prefetch below and the
        // page cache both prefer.
        rows.sort_unstable();
        let query_norm = query_norm(self.quantization, query);
        // A cold page fault also pulls the neighboring pages. Read just these rows, queued
        // together, when any of them is still on disk.
        if !self.rows_resident(rows) {
            let len = self.row_len();
            let ranges: Vec<(u64, usize)> = rows.iter().map(|row| (*row as u64 * len as u64, len)).collect();
            if let Ok(buffers) = crate::warm::read_ranges(&self.vectors_file, &ranges) {
                return rows
                    .iter()
                    .zip(buffers)
                    .map(|(row, bytes)| {
                        (
                            ids[*row as usize],
                            exact_distance(
                                self.quantization,
                                query,
                                query_norm,
                                &bytes,
                                self.stats()[*row as usize].norm,
                            ),
                        )
                    })
                    .collect();
            }
        }
        let mut hits = Vec::with_capacity(rows.len());
        for (at, row) in rows.iter().enumerate() {
            if let Some(ahead) = rows.get(at + PREFETCH_AHEAD) {
                prefetch(self.row(*ahead as usize));
            }
            hits.push((ids[*row as usize], self.exact_distance(query, query_norm, *row as usize)));
        }
        hits
    }

    /// The `limit` closest live rows of this segment alone. The list is short only when live
    /// rows run out.
    pub fn search(
        &self,
        query: &DenseQuery,
        limit: usize,
        dead: &impl Excluded,
        ids: &[u64],
        live_count: usize,
    ) -> Result<VectorHits> {
        let budget = Budget::split(limit, &[live_count])[0];
        let mut rows: Vec<u32> =
            self.candidates(query, budget, dead, ids, live_count)?.into_iter().map(|(row, _)| row).collect();
        let mut hits = self.exact(query, &mut rows, ids);
        hits.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        hits.truncate(limit.min(live_count));
        Ok(VectorHits { hits, step: FallbackStep::E64 })
    }
}

/// How much of the cascade one segment runs for a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Budget {
    /// Rows the Hamming prefilter keeps.
    pub prefilter: usize,
    /// Rows the 4-bit estimate keeps for the global exact rescore.
    pub refine: usize,
}

impl Budget {
    /// Budgets for `limit` hits over segments with these live counts. The collection keeps
    /// `PREFILTER_PER_HIT` prefiltered rows per hit; a segment gets twice its share of them, at
    /// least `PREFILTER_SEGMENT_MIN`. Any segment may hold every best hit, so each may pass the
    /// whole refine count on.
    pub fn split(limit: usize, live: &[usize]) -> Vec<Budget> {
        let total: usize = live.iter().sum();
        let want = limit.min(total);
        let prefilter = (want * PREFILTER_PER_HIT).max(PREFILTER_MIN);
        let refine = Budget::refine_total(limit, total);
        live.iter()
            .map(|&count| {
                if want == 0 || count == 0 {
                    return Budget { prefilter: 0, refine: 0 };
                }
                let share = (2 * prefilter * count).div_ceil(total).min(prefilter);
                Budget { prefilter: share.max(PREFILTER_SEGMENT_MIN).min(count), refine: refine.min(count) }
            })
            .collect()
    }

    /// Rows the collection rescores exactly for `limit` hits among `live` rows.
    pub fn refine_total(limit: usize, live: usize) -> usize {
        let want = limit.min(live);
        if want == 0 {
            0
        } else {
            (want * REFINE_PER_HIT).max(REFINE_MIN)
        }
    }
}

/// `|row|` as `codes.bin` stores it: the norm of the dequantized values.
pub(crate) fn row_norm(quantization: Quantization, row: &[u8]) -> f32 {
    row_values(quantization, row).iter().map(|v| v * v).sum::<f32>().sqrt()
}

/// The int8 query's norm, the other operand of `exact_distance`; 0 for float modes.
pub(crate) fn query_norm(quantization: Quantization, query: &DenseQuery) -> f32 {
    if stored_quantization(quantization) == Quantization::I8 {
        (as_i8(&query.row).iter().map(|v| i32::from(*v) * i32::from(*v)).sum::<i32>() as f32).sqrt()
    } else {
        0.0
    }
}

/// Exact cosine distance from the query to a stored row with norm `norm`; a zero vector is
/// distance 1. Int8 rows use the integer dot product against the quantized query.
pub(crate) fn exact_distance(
    quantization: Quantization,
    query: &DenseQuery,
    query_norm: f32,
    row: &[u8],
    norm: f32,
) -> f32 {
    if stored_quantization(quantization) == Quantization::I8 {
        if query_norm == 0.0 || norm == 0.0 {
            return 1.0;
        }
        // SAFETY: `kernels()` only returns functions this CPU supports.
        let dot = unsafe { (kernels().dot_i8)(as_i8(&query.row), as_i8(row)) };
        return 1.0 - dot as f32 / (query_norm * norm);
    }
    cosine_distance(&query.raw, &row_values(quantization, row))
}

/// Rows ahead of the current one whose bytes are requested early.
const PREFETCH_AHEAD: usize = 8;

/// Asks the CPU to start loading `bytes` into cache. A no-op where unsupported.
#[inline(always)]
fn prefetch(bytes: &[u8]) {
    #[cfg(target_arch = "x86_64")]
    for line in bytes.chunks(64) {
        // SAFETY: a prefetch is only a hint; it never faults, even for an unmapped address.
        unsafe { std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(line.as_ptr() as *const i8) };
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = bytes;
}

/// The rows whose distance is at most the smallest cutoff that keeps `keep` of them, in row
/// order. `u16::MAX` marks a row to skip. Four interleaved histograms keep consecutive equal
/// distances from waiting on each other's increments.
fn cut(distances: &[u16], max_distance: usize, keep: usize) -> Vec<u32> {
    let width = max_distance + 1;
    let mut histograms = vec![0u32; 4 * width];
    let (quads, rest) = distances.as_chunks::<4>();
    for quad in quads {
        for (lane, d) in quad.iter().enumerate() {
            if *d != u16::MAX {
                histograms[lane * width + *d as usize] += 1;
            }
        }
    }
    for d in rest.iter().filter(|d| **d != u16::MAX) {
        histograms[*d as usize] += 1;
    }
    let mut kept = 0usize;
    let mut cutoff = max_distance;
    for d in 0..width {
        kept += (0..4).map(|lane| histograms[lane * width + d] as usize).sum::<usize>();
        if kept >= keep {
            cutoff = d;
            break;
        }
    }
    // Branch-free append: every row is written, and only kept rows advance the cursor.
    let mut out = vec![0u32; distances.len() + 1];
    let mut len = 0;
    for (row, d) in distances.iter().enumerate() {
        out[len] = row as u32;
        len += usize::from(*d as usize <= cutoff);
    }
    out.truncate(len);
    out
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

    #[test]
    fn rotation_is_orthonormal() {
        let rotation = Rotation::new(384, 2, 7);
        let a: Vec<f32> = (0..384).map(|i| ((i * 37 % 101) as f32 - 50.0) / 13.0).collect();
        let b: Vec<f32> = (0..384).map(|i| ((i * 53 % 89) as f32 - 44.0) / 7.0).collect();
        let (ra, rb) = (rotation.apply(&a), rotation.apply(&b));
        assert_eq!(ra.len(), 512);
        let dot = |x: &[f32], y: &[f32]| x.iter().zip(y).map(|(p, q)| p * q).sum::<f32>();
        assert!((dot(&ra, &rb) - dot(&a, &b)).abs() < 1e-2 * dot(&a, &a).sqrt() * dot(&b, &b).sqrt());
        assert!((dot(&ra, &ra) - dot(&a, &a)).abs() < 1e-3 * dot(&a, &a));
    }

    #[test]
    fn plane_estimate_tracks_the_true_inner_product() {
        let rotation = Rotation::new(512, 1, 3);
        let mut state = 99u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 2001) as f32 / 1000.0 - 1.0
        };
        let query: Vec<f32> = (0..512).map(|_| next()).collect();
        let rq = rotation.apply(&query);
        let planes = PlaneQuery::new(&rq);
        let mut pairs = Vec::new();
        for _ in 0..300 {
            // Rows correlated with the query to different degrees.
            let mix = next().abs();
            let row: Vec<f32> = query.iter().map(|q| mix * q + (1.0 - mix) * next()).collect();
            let rr = rotation.apply(&row);
            let factor = rr.iter().map(|v| v.abs()).sum::<f32>() / ((rr.len() as f32).sqrt() * norm_f32(&rr));
            let estimate = planes.estimate(&sign_bits(&rr, 8), factor);
            let norm = rr.iter().map(|v| v * v).sum::<f32>().sqrt();
            let truth = rr.iter().zip(&rq).map(|(a, b)| a * b).sum::<f32>() / norm;
            pairs.push((estimate, truth));
        }
        // Rank correlation between estimate and truth is high.
        let n = pairs.len();
        let concordant = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .filter(|&(i, j)| (pairs[i].0 - pairs[j].0) * (pairs[i].1 - pairs[j].1) > 0.0)
            .count();
        let total = n * (n - 1) / 2;
        assert!(concordant as f64 / total as f64 > 0.85, "{concordant} of {total}");
    }

    fn norm_f32(v: &[f32]) -> f32 {
        v.iter().map(|x| x * x).sum::<f32>().sqrt()
    }

    fn clustered_rows(count: usize, dimension: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut state = seed;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % 2001) as f32 / 1000.0 - 1.0
        };
        let centers: Vec<Vec<f32>> = (0..40).map(|_| (0..dimension).map(|_| next()).collect()).collect();
        (0..count).map(|i| centers[i % centers.len()].iter().map(|c| c + 0.6 * next()).collect()).collect()
    }

    fn segment_of(dir: &Path, quantization: Quantization, rows: &[Vec<f32>]) -> (VectorSegment, Vec<u64>) {
        let quantized: Vec<Vec<u8>> = rows.iter().map(|r| quantize(quantization, r)).collect();
        let slices: Vec<&[u8]> = quantized.iter().map(|r| r.as_slice()).collect();
        std::fs::write(dir.join("codes.bin"), encode_codes(quantization, rows[0].len(), &slices)).unwrap();
        std::fs::write(dir.join("vectors.bin"), quantized.concat()).unwrap();
        let ids = (0..rows.len() as u64).map(|i| i * 3 + 1).collect();
        (VectorSegment::open(dir, quantization, rows[0].len()).unwrap(), ids)
    }

    #[test]
    fn cascade_finds_the_exact_neighbors_and_skips_the_dead() {
        let rows = clustered_rows(12_000, 384, 5);
        let dir = tempfile::tempdir().unwrap();
        let (segment, ids) = segment_of(dir.path(), Quantization::I8, &rows);
        let dead = Tombstones::from_ids(ids.iter().copied().filter(|id| id % 7 == 1).collect());
        let live_count = ids.iter().filter(|id| !dead.contains(**id)).count();
        // Queries near the data, as real queries are: a stored row plus noise.
        let noise = clustered_rows(40, 384, 77);
        let bases: Vec<usize> = (0..40).map(|i| i * 293).collect();
        let queries: Vec<Vec<f32>> =
            noise.iter().zip(&bases).map(|(n, b)| rows[*b].iter().zip(n).map(|(r, n)| r + 0.3 * n).collect()).collect();
        let mut found = 0;
        for (raw, base) in queries.iter().zip(&bases) {
            let query = DenseQuery::new(Quantization::I8, raw);
            let hits = segment.search(&query, 10, &dead, &ids, live_count).unwrap().hits;
            assert_eq!(hits.len(), 10);
            // The row a query was made from is far closer than anything else.
            if !dead.contains(ids[*base]) {
                assert_eq!(hits[0].0, ids[*base]);
            }
            assert!(hits.iter().all(|(id, _)| !dead.contains(*id)));
            assert!(hits.windows(2).all(|w| w[0].1 <= w[1].1));
            let mut exact: Vec<(u64, f32)> = (0..rows.len())
                .filter(|r| !dead.contains(ids[*r]))
                .map(|r| (ids[r], segment.exact_distance(&query, 0.0_f32.max(norm_i8(&query.row)), r)))
                .collect();
            exact.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            // This data has near-ties (the top 50 span about 0.015), so a hit as close as the
            // exact 10th, within 1e-3, counts as found.
            found += hits.iter().filter(|h| h.1 <= exact[9].1 + 1e-3).count();
        }
        assert!(found as f64 / 400.0 > 0.85, "tie-aware recall@10 {}", found as f64 / 400.0);
    }

    fn norm_i8(row: &[u8]) -> f32 {
        (as_i8(row).iter().map(|v| i32::from(*v) * i32::from(*v)).sum::<i32>() as f32).sqrt()
    }

    #[test]
    fn scattered_row_reads_match_the_mapping() {
        let rows = clustered_rows(50, 64, 9);
        let dir = tempfile::tempdir().unwrap();
        let (segment, _) = segment_of(dir.path(), Quantization::I8, &rows);
        let want = [0u32, 7, 49, 3];
        let len = segment.row_len();
        let ranges: Vec<(u64, usize)> = want.iter().map(|row| (*row as u64 * len as u64, len)).collect();
        let got = crate::warm::read_ranges(&segment.vectors_file, &ranges).unwrap();
        for (row, bytes) in want.iter().zip(&got) {
            assert_eq!(bytes, segment.row(*row as usize));
        }
    }

    #[test]
    fn small_and_float_segments_are_exact() {
        for quantization in [Quantization::F32, Quantization::F16, Quantization::B1, Quantization::I8] {
            let rows = clustered_rows(300, 100, 9);
            let dir = tempfile::tempdir().unwrap();
            let (segment, ids) = segment_of(dir.path(), quantization, &rows);
            let query = DenseQuery::new(quantization, &rows[17]);
            let hits = segment.search(&query, 5, &Tombstones::default(), &ids, rows.len()).unwrap().hits;
            assert_eq!(hits[0].0, ids[17], "{quantization:?}");
            assert!(hits[0].1 < 1e-2, "{quantization:?} {}", hits[0].1);
            let stored = segment.stored_vector(ids[17], &ids).unwrap();
            assert_eq!(stored.len(), 100);
        }
    }

    #[test]
    fn a_damaged_codes_file_is_an_error() {
        let rows = clustered_rows(50, 64, 1);
        let dir = tempfile::tempdir().unwrap();
        let _ = segment_of(dir.path(), Quantization::I8, &rows);
        let path = dir.path().join("codes.bin");
        let bytes = std::fs::read(&path).unwrap();
        for cut in [0, 7, 31, bytes.len() - 1] {
            std::fs::write(&path, &bytes[..cut]).unwrap();
            assert!(VectorSegment::open(dir.path(), Quantization::I8, 64).is_err(), "cut {cut}");
        }
        let mut wrong = bytes.clone();
        wrong[16] = 65;
        std::fs::write(&path, &wrong).unwrap();
        assert!(VectorSegment::open(dir.path(), Quantization::I8, 64).is_err());
    }
}
