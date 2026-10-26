<sub>[Docs](../README.md) / Guides</sub>

# Indexing large corpora

How to load hundreds of thousands of documents quickly, keep memory bounded, and leave the
collection in a shape that queries fast. Read [Architecture](../concepts/architecture.md) first
if the terms segment, buffer, and merge are new.

## Reference points

Lite tier, 4-vCPU cloud VM, one `gyr bench beir` run per dataset (`bench/results/2026-10-26-quality-sweep`):

| Corpus | Documents | Index time | Query p50 |
|---|---|---|---|
| SciFact | 5,183 | 0.55 s | 1.8 ms |
| FiQA | 57,638 | 2.0 s | 2.7 ms |
| TREC-COVID | 171,332 | 13.8 s | 3.4 ms |
| Touché-2020 | 382,545 | 50.5 s | 5.9 ms |
| Quora | 522,931 | 7.1 s | 6.3 ms |

Index time follows bytes of text more than document count: Quora's half a million documents are
one-line questions, Touché's are long arguments.

## Write in large batches

Every write call has a fixed cost, and the engine decides how to store a write by its size:

| Write size | What happens |
|---|---|
| fewer than 256 chunks | appended to the write-ahead log (one fsync, about 0.3 ms) and added to the in-memory buffer |
| 256 chunks or more | sealed directly into a new segment directory |

The buffer seals into a segment once it holds 4,096 chunks or 32 MiB. Small writes are therefore
cheap one at a time, and large writes skip the buffer because copying them through it would
only add work.

For bulk loads, pass many documents per call. Measured on FiQA (57,638 documents), lite tier
(spec, "Write buffer and WAL"):

| Documents per call | Total time |
|---|---|
| 1 | 44 s |
| 100 | 6.6 s |
| 10,000 | 2.5 s |

In Python, give `add_many` the whole iterable. It sends documents to the engine 512 at a time
and does not hold your corpus in memory:

```python
import json
from gyrfalcon import Collection

def documents(path):
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            yield row["id"], row["text"], {"source": row["source"]}

with Collection("./big-index") as col:
    print(col.add_many(documents("corpus.jsonl")), "chunks")
```

In Rust, pass slices of a few thousand `Document`s to `index_documents`. One call seals a
segment every 100,000 chunks, so a single very large slice is also fine.

## Use every core, or fewer

Chunking, analysis, and static embedding run on every core the process may use. To leave cores
for other work, restrict the process's CPU affinity; the engine follows it:

```sh
taskset -c 0-1 gyr index ./big-index corpus/
```

The full tier's ONNX Runtime uses `min(4, cores)` intra-op threads unless `GYR_ORT_THREADS` sets
the number.

## Memory

Peak resident memory while indexing FiQA with the lite tier is 414 MB on 4 cores and 352 MB on
one core (`bench/results/2026-10-26-lite-refresh`), and that figure includes the benchmark
harness's own copy of the corpus. What grows with the corpus:

- **The write buffer**, at most 4,096 chunks or 32 MiB before it seals.
- **A merge in progress**, which holds about twice the text of the segments it merges, up to
  100,000 chunks.
- **The lite embedding table** is memory-mapped, so a process only pays for the rows of the
  tokens it meets.
- **The full tier's activations**, bounded by 8 KB batches sorted by length. On SciFact this
  took indexing from 2.1 GB to 0.5 GB peak without slowing it (spec, "Embedding batches sized
  for the model").

Segment files are memory-mapped. Their pages count toward resident memory while they are hot and
are released by the OS under pressure.

## Merges and segment count

Each sealed segment is immutable. A background merger keeps their number down: whenever eight
segments of the same size class (by powers of two) together hold at most 100,000 chunks, it
merges them into one. Segments of 100,000 chunks or more are never merged with others; one is
rewritten alone when more than 20% of its chunks have been deleted.

A query visits every segment, so many small segments cost latency. Above 32 segments, the Rust
reports, `gyr` output, and HTTP responses carry the warning `segment_count`.

After a bulk load, or after many deletes, compact explicitly:

```sh
gyr compact ./big-index          # one round of the background policy, synchronously
gyr compact ./big-index --all    # rewrite everything into one segment and drop every deleted chunk
```

`gyr compact` takes the same `--tier` and model flags as the collection was built with. The
Python `close()` waits for the background merger but does not compact.

## Files and PDFs

`gyr index DIR PATH...` and `Collection.add_path` walk directories in path order, skip hidden
entries, and read PDF, HTML, Markdown, and text files. A file that fails to read is skipped with
its reason; the run continues.

PDF extraction runs through PDFium at about 3.4 ms of CPU per page plus 0.15 ms for the layout
layer. PDFium serializes calls, so PDF extraction uses one core however many the machine has;
100 arXiv papers (2,747 pages) index in about 12 s. For large PDF sets, extract text in several
processes yourself and add the text with `add_many`:

```python
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import gyrfalcon

def extract(path):
    text, _page_starts = gyrfalcon.read_file(path)
    return str(path), text

if __name__ == "__main__":
    paths = sorted(Path("papers").glob("*.pdf"))
    with ProcessPoolExecutor() as pool, gyrfalcon.Collection("./papers-index") as col:
        col.add_many(pool.map(extract, paths, chunksize=8))
```

`read_file` returns the text and the byte offset where each page starts, which you can store as
metadata or use to map a hit's `start` offset back to a page.

## Scale limits

These are the known limits, stated so you can plan around them:

- The dense leg scans every live row's sign bits, about 3 ns per row per core. At a million
  chunks that is a few milliseconds per query on one core. Partitioned search inside large
  segments is designed but not built.
- The BM25 leg scores every posting of the query's terms, about 11 ns per posting. Long queries
  over very large collections pay for that. Pruned scoring was tried at BEIR sizes and was
  slower (spec, "Block-max MaxScore, tried and reverted"); it remains open for collections of a
  million chunks or more.
- The largest collection measured end to end is Quora, 522,931 documents, at 6.3 ms p50.

---

<sub>Previous: [Choosing a tier](choosing-a-tier.md) · Next: [Custom embedders](custom-embedders.md)</sub>
