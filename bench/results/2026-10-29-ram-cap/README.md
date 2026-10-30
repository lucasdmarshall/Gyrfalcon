# Quora larger than the cgroup, 2026-10-29

Lite Quora (522,931 chunks, 368 MB) plus potion-retrieval-32M (131 MB), 20 judged test queries in id order, `k=10`. The process ran in a sibling cgroup with `memory.max` of 128 MB or 256 MB and `memory.swap.max=0`, so the machine's swap file could not absorb the cap. `none` is an open that does not prefetch. `eager` is the PR #3 warm-open. No run was OOM-killed (`oom_kill` 0).

`read_bytes` is `/proc/self/io` of the query process. On the real disk it includes readahead. On the FUSE mount the bytes the daemon reads from the backing file are not in that counter; major faults are the comparable figure there. A major fault is not one page when the kernel readaheads: eager open's ~3,300 faults on gp3 line up with ~128 KB windows over the collection, and the real-disk eager open's `read_bytes` is 500 MB, the collection plus the model.

The real disk is this VM's virtio device with a warm host cache. Dropping the guest page cache does not drop that host cache, so real-disk latency stays low while the byte counts still move. The gp3 column is `bench/slowfs.py --latency-ms 0.65 --mb-per-s 125` (measured about 2.0 ms per random 4 KB and 118 MB/s on this mount). FUSE does not model IOPS. Unprivileged `fusermount` failed; the daemon ran as root with `allow_other`.

| | open | open faults | open read | query p50 | query p99 | faults/query p50 | read/query p50 | hit memory.max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| real, 256 MB, none | 44 ms | 94 | 25 MB | 12.8 ms | 42.5 ms | 13 | 4.2 MB | 0 |
| real, 256 MB, eager | 504 ms | 3,413 | 500 MB | 11.1 ms | 39.4 ms | 17 | 3.5 MB | 2,296 |
| real, 128 MB, none | 42 ms | 94 | 25 MB | 13.1 ms | 40.2 ms | 19 | 5.0 MB | 392 |
| real, 128 MB, eager | 488 ms | 3,415 | 508 MB | 13.1 ms | 45.1 ms | 31 | 4.0 MB | 3,333 |
| gp3, 256 MB, none | 382 ms | 94 | 1.5 MB | 78.3 ms | 446 ms | 13 | 0.05 MB | 0 |
| gp3, 256 MB, eager | 6,102 ms | 3,244 | 131 MB | 73.6 ms | 388 ms | 20 | 0.00 MB | 2,299 |
| gp3, 128 MB, none | 377 ms | 94 | 1.5 MB | 80.7 ms | 438 ms | 22 | 0.05 MB | 394 |
| gp3, 128 MB, eager | 6,370 ms | 3,559 | 131 MB | 80.0 ms | 472 ms | 32 | 0.04 MB | 3,310 |

`hit memory.max` is `memory.events` `max`: how many times the cgroup went over the cap and reclaimed.

## Where it degrades

The 128 MB cap cannot hold the 131 MB model and the index together. `memory.current` sat on the cap for every 128 MB run, and `max` fired a few hundred times even with warmup off. The 256 MB cap holds the model plus the rows these 20 queries touch when warmup is off: `max` stayed 0 and `memory.current` ended near 190 MB.

Eager warmup does not survive the cap. It reads the whole collection at open (500 MB of `read_bytes` on the real disk, ~3,300 major faults on gp3) and the cgroup reclaims it during that read (`max` about 2,300 at 256 MB and about 3,300 at 128 MB). The queries after that open still fault. On gp3 the open costs 6.1 s at 256 MB and 6.4 s at 128 MB, and the query p50 is 74 ms and 80 ms against 78 ms and 81 ms with no warmup. The prefetch is not kept, so the open cost is not paid back.

Query p50 does not fall off a cliff between 256 MB and 128 MB on this 20-query slice: about 1 ms on the real disk (13 ms either way) and about 2–6 ms on gp3 (78 ms to 81 ms). The first query is the p99, ~40 ms on the host-cached virtio disk and ~400–470 ms on gp3, at both caps. What the cap changes on the clock is the eager open. What it changes on the counters is that every query still fetches a few megabytes from the real disk (3–5 MB p50, ~50 MB on the first query) because those pages are not resident. The host cache hides that in the latency. A cold host, or a disk without this VM's cache, would show those megabytes at the disk's own rate. gp3 already does, for the pages the query actually faults: p50 ~80 ms, matching the uncapped gp3 first-ten from `bench/results/2026-10-29-warm-modes` rather than a new, worse regime.

No corpus of 1M+ chunks was built. Quora lite is 522,931 chunks. A million-chunk index was not feasible beside these runs.

Reproduce: `python3 bench/ram_cap.py run COLLECTION .models BEIR/quora --max-mb 128 --warm none --queries 20 --out run.json`. The parent creates the cgroup with sudo. Swap for that cgroup is 0.
