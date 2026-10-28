# Reading from disk, 2026-10-28

`bench/cold_cache.py` over lite collections of FiQA (57,638 documents, 79 MB) and Quora (522,931,
370 MB), built with the Python package, at commit 0d5e31b on a 4-vCPU cloud VM. Before each cold
measurement every page of the collection and of potion-retrieval-32M (131 MB) is dropped from the
page cache with `posix_fadvise(DONTNEED)`; `mincore` confirmed 0.0 MB cached afterwards.

Profiles, each a pair of JSON files here:

| Profile | How | Measured random 4 KB read | Sequential |
|---|---|---|---|
| `real` | this VM's virtio disk, directly | 97 to 166 us | 58 to 536 MB/s |
| `fuse-overhead` | `bench/slowfs.py --latency-ms 0 --mb-per-s 0` | 291 to 434 us | 258 to 373 MB/s |
| `cloud-gp3` | `bench/slowfs.py --latency-ms 0.65 --mb-per-s 125` | 1,160 to 1,188 us | 184 to 193 MB/s |
| `hdd` | `bench/slowfs.py --latency-ms 7.65 --mb-per-s 150` | 8,182 to 8,373 us | 53 to 54 MB/s |

| | FiQA cloud | FiQA HDD | Quora cloud | Quora HDD |
|---|---|---|---|---|
| open | 275 ms | 1,038 ms | 1,031 ms | 3,446 ms |
| first query | 710 ms | 3,075 ms | 1,859 ms | 7,054 ms |
| first 10, p50 | 29.8 ms | 119 ms | 414 ms | 1,753 ms |
| whole cold-start run, p50 / p99 | 2.84 / 32.0 ms | 2.46 / 128 ms | 12.7 / 193 ms | 11.9 / 522 ms |
| warm, p50 / p99 | 1.81 / 3.67 ms | 1.64 / 3.00 ms | 9.90 / 14.8 ms | 6.68 / 13.3 ms |
| every query cold, p50 / p99 | 735 / 833 ms | 2,990 / 3,097 ms | 1,797 / 2,013 ms | 7,131 / 7,479 ms |

`every query cold` reopens the collection after each eviction (open p50: FiQA 294 ms cloud,
957 ms HDD; Quora 854 ms cloud, 3,499 ms HDD), because mapped pages cannot be dropped; the query
time excludes the open. It ran 40 queries on the cloud profile and 15 on the HDD profile.

The `real` runs show how much this VM's disk depends on its host's cache: the first open of the day
took 4.4 s (FiQA) and 10.3 s (Quora), while later cold queries took 51 ms and 141 ms p50 once the
host held the blocks. Measured with `/proc/self/io` on a cold FiQA collection, the open reads
9.8 MB and the first query 36 MB, mostly read-around from random row reads.

Reproduce: build the collections with the Python package, mount `bench/slowfs.py` over them with a
profile, and run `bench/cold_cache.py` on the mount (needs libfuse 2 and `pip install fusepy`).
