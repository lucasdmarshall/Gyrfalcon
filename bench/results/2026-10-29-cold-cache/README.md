# Reading from disk, 2026-10-29

`bench/cold_cache.py` over lite collections of FiQA (57,638 documents, 76 MB) and Quora
(522,931 documents, 368 MB). Each document is one section, built with potion-retrieval-32M at
commit 61a4293. Before each cold measurement the collection and the model are dropped from the
guest page cache. `mincore` reported 0.0 MB cached afterwards.

This VM's own virtio disk is cached by its host: a random 4 KB read returns in about 1 to 2
μs and sequential reads exceed 2 GB/s. That is not a production disk. Cloud gp3 and a hard
disk are emulated with `bench/slowfs.py` mounted over the collection, the same profiles as
`2026-10-28-cold-cache`:

| Profile | How | Random 4 KB | Sequential |
|---|---|---|---|
| `real` | this VM's virtio disk | 1 to 2 μs | 2.2 to 4.3 GB/s |
| `gp3` | `--latency-ms 0.65 --mb-per-s 125` | 987 to 1,000 μs | 200 to 233 MB/s |
| `hdd` | `--latency-ms 7.65 --mb-per-s 150` | 8,140 to 8,144 μs | 54 to 57 MB/s |

The model files stay on the virtio disk, so after the guest cache drop they are still served
from the host cache. Open is faster here than in the 2026-10-28 report for that reason.

| | FiQA real | FiQA gp3 | FiQA HDD | Quora real | Quora gp3 | Quora HDD |
|---|---:|---:|---:|---:|---:|---:|
| open | 30 ms | 167 ms | 687 ms | 49 ms | 384 ms | 1,551 ms |
| first query | 32 ms | 608 ms | 2,774 ms | 79 ms | 1,738 ms | 6,839 ms |
| first 10, p50 | 2.4 ms | 22 ms | 74 ms | 15 ms | 377 ms | 1,670 ms |
| whole cold-start run, p50 / p99 | 1.06 / 3.2 ms | 1.07 / 23 ms | 1.07 / 83 ms | 3.15 / 7.5 ms | 3.04 / 110 ms | 2.85 / 536 ms |
| warm, p50 / p99 | 0.85 / 1.25 ms | 0.84 / 1.27 ms | 0.85 / 1.25 ms | 3.03 / 3.56 ms | 2.81 / 3.52 ms | 2.74 / 3.44 ms |
| every query cold, p50 / p99 | 28 / 60 ms | 584 / 635 ms | 2,709 / 2,836 ms | 82 / 93 ms | 1,644 / 1,769 ms | 6,818 / 7,132 ms |

`every query cold` uses 40 queries on real and gp3, and 15 on HDD, reopening after each
eviction. Query time excludes the open. FiQA's cold-start run has 648 test queries. Quora's
has the first 1,000 test queries in id order.
