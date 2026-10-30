# Warm mode on the cloud-gp3 profile, 2026-10-29

Open time and the first ten queries for `warm=none`, `background`, and `eager`. Lite collections
of FiQA (76 MB) and Quora (368 MB), potion-retrieval-32M (131 MB) on the virtio disk. The
collection was mounted with `bench/slowfs.py --latency-ms 0.65 --mb-per-s 125`. Unprivileged
`fusermount` returned EPERM in this container, so the daemon ran as root with `allow_other`.
Each open was preceded by `posix_fadvise(DONTNEED)` on the collection and the model; mincore
reported 0.0 MB cached. Three repeats. The median is the middle repeat. 4 vCPU. One process.

Measured through the mount: FiQA random 4 KB 2,062 µs and 125 MB/s sequential; Quora 1,967 µs and
118 MB/s. FUSE does not model IOPS. The first `none` repeat was slower than the next two (FiQA
open 266 ms and first query 471 ms; Quora open 634 ms and first query 856 ms) because `pread`
under the FUSE daemon still sees the host cache. The medians below are the later repeats. Raw
runs: `gp3-fiqa.json`, `gp3-quora.json`.

| | FiQA open | FiQA first query | FiQA first-10 p50 | Quora open | Quora first query | Quora first-10 p50 |
|---|---:|---:|---:|---:|---:|---:|
| none | 170 ms | 126 ms | 48.8 ms | 365 ms | 434 ms | 79.9 ms |
| background | 169 ms | 117 ms | 32.4 ms | 364 ms | 417 ms | 63.3 ms |
| eager | 1,096 ms | 1.5 ms | 1.0 ms | 5,558 ms | 7.3 ms | 3.4 ms |

Time to the first hit (open plus the first query), median of the three repeats: FiQA 295 ms
(`none`), 286 ms (`background`), 1,098 ms (`eager`). Quora 799 ms, 777 ms, 5,565 ms.

`background` does not make the first query fast. It stays within a few milliseconds of `none`.
What it changes is the open, which matches `none`, and the queries after the prefetch has had
time to run. One extra run, not in the median table: FiQA `background` was at 1.3–2 ms by query
21, next to eager's 1 ms. Quora was still 27 ms at query 40 and 4.6 ms p50 over queries 111–120,
next to eager's 3.4 ms, after 3.2 s of query time. At 118 MB/s the 368 MB collection takes about
three seconds to read, and the first ten queries finish before that.

`eager` is the wrong default for a short-lived process. On Quora it spends 5.6 s before the first
search, and the time to the first hit is about seven times `none`. A process that must answer
once and exit should not pay that. A process that stays open and passes `eager` still gets a
1–7 ms first query if it can wait for open.

The default is `background` for the Rust `Index::open`, the CLI (`--warm` / `GYR_WARM`), Python
(`Collection(..., warm=...)`), and therefore HTTP. Pass `eager` when the first query has to be
the fast one and the process can wait at open. Pass `none` to skip the prefetch.

These eager opens (1.1 s and 5.6 s) are faster than the 3.7 s and 8.9 s reported for the same
flags when warm-open was first measured. This file is the run the default was chosen from. The
profile flags are the published cloud-gp3 pair; the measured 4 KB latency here was about 2.0 ms,
not the 1.2 ms of that earlier writeup.

Reproduce: mount the collection, then

```
python3 bench/warm_modes.py MOUNT .models BEIR_DIR --label gp3-fiqa --repeats 3 --out gp3-fiqa.json
```
