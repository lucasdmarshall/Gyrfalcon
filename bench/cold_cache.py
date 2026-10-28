"""Query latency when a collection is read from disk instead of the page cache.

    python3 bench/cold_cache.py COLLECTION_DIR MODELS_DIR BEIR_DIR [--queries N] [--every N] [--out FILE]

COLLECTION_DIR is a lite collection built from BEIR_DIR's corpus (see docs/benchmarks/reproducing.md),
MODELS_DIR holds potion-retrieval-32M, and BEIR_DIR supplies the test queries. Linux only. Before
each cold measurement every file of the collection and of the model is dropped from the page
cache with posix_fadvise(DONTNEED), which needs no privileges, and mincore confirms it.

Reported, each over the same queries in the same order:
  disk           median latency of random 4 KB reads and sequential throughput, through the same
                 path, so an emulated disk (bench/slowfs.py) shows what it emulates
  cold start     evict, open, then every query once: the open, the first query, the first ten,
                 and the whole run while the cache fills
  warm           the same queries again on the now cached collection
  cold per query evict before each of --every queries: every query reads from disk
"""
import argparse
import ctypes
import json
import mmap
import os
import random
import statistics
import time

from gyrfalcon import Collection

LIBC = ctypes.CDLL("libc.so.6", use_errno=True)


def files(*roots):
    for root in roots:
        for directory, _, names in os.walk(root):
            for name in names:
                yield os.path.join(directory, name)


def evict(*roots):
    for path in files(*roots):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def cached_bytes(*roots):
    """Bytes of the files under `roots` in the page cache, by mincore over a private mapping."""
    page = mmap.PAGESIZE
    cached = 0
    for path in files(*roots):
        size = os.path.getsize(path)
        if size == 0:
            continue
        with open(path, "rb") as f:
            # A private mapping is writable for ctypes but touches no page until written.
            mapping = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_COPY)
            pages = (size + page - 1) // page
            vector = (ctypes.c_ubyte * pages)()
            buffer = ctypes.c_char.from_buffer(mapping)
            LIBC.mincore(ctypes.c_void_p(ctypes.addressof(buffer)), ctypes.c_size_t(size), vector)
            cached += sum(v & 1 for v in vector) * page
            del buffer
            mapping.close()
    return cached


def disk_probe(path):
    """Random 4 KB read latency (median, microseconds) and sequential MB/s on `path`, uncached."""
    size = os.path.getsize(path)
    evict(path)
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
    rng = random.Random(1)
    latencies = []
    for _ in range(min(300, size // 4096)):
        offset = rng.randrange(0, max(size - 4096, 1)) // 4096 * 4096
        started = time.perf_counter()
        os.pread(fd, 4096, offset)
        latencies.append((time.perf_counter() - started) * 1e6)
    os.close(fd)
    evict(path)
    fd = os.open(path, os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_SEQUENTIAL)
    started = time.perf_counter()
    total = 0
    while True:
        block = os.read(fd, 1 << 20)
        if not block:
            break
        total += len(block)
    elapsed = time.perf_counter() - started
    os.close(fd)
    return statistics.median(latencies), total / 1e6 / elapsed


def percentile(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, round(p / 100 * (len(values) - 1)))]


def stats(ms):
    return {"p50_ms": statistics.median(ms), "p99_ms": percentile(ms, 99), "max_ms": max(ms), "n": len(ms)}


def timed(collection, queries):
    out = []
    for query in queries:
        started = time.perf_counter()
        collection.search(query, k=10)
        out.append((time.perf_counter() - started) * 1000)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("collection")
    parser.add_argument("models")
    parser.add_argument("beir")
    parser.add_argument("--queries", type=int, default=1000, help="test queries to run, in id order")
    parser.add_argument("--every", type=int, default=40, help="queries for the cold-per-query run")
    parser.add_argument("--label", default="")
    parser.add_argument("--out")
    args = parser.parse_args()

    model_dir = os.path.join(args.models, "potion-retrieval-32M")
    with open(os.path.join(args.beir, "qrels", "test.tsv")) as f:
        next(f)
        judged = {line.split("\t")[0] for line in f}
    queries = [q for q in map(json.loads, open(os.path.join(args.beir, "queries.jsonl"))) if q["_id"] in judged]
    queries = [q["text"] for q in sorted(queries, key=lambda q: q["_id"])][: args.queries]

    largest = max(files(args.collection), key=os.path.getsize)
    random_us, sequential = disk_probe(largest)
    report = {
        "label": args.label,
        "collection_mb": sum(os.path.getsize(p) for p in files(args.collection)) / 1e6,
        "model_mb": sum(os.path.getsize(p) for p in files(model_dir)) / 1e6,
        "queries": len(queries),
        "disk": {"random_4k_read_us": random_us, "sequential_mb_per_s": sequential},
    }

    evict(args.collection, model_dir)
    report["cached_after_eviction_mb"] = cached_bytes(args.collection, model_dir) / 1e6
    started = time.perf_counter()
    collection = Collection(args.collection, models=args.models, create=False)
    open_ms = (time.perf_counter() - started) * 1000
    cold = timed(collection, queries)
    warm = timed(collection, queries)
    collection.close()
    report["cold_start"] = {"open_ms": open_ms, "first_query_ms": cold[0], "first_10": stats(cold[:10]),
                            "all": stats(cold)}
    report["warm"] = stats(warm)

    per_query, opens = [], []
    for query in queries[: args.every]:
        evict(args.collection, model_dir)
        started = time.perf_counter()
        collection = Collection(args.collection, models=args.models, create=False)
        opens.append((time.perf_counter() - started) * 1000)
        per_query += timed(collection, [query])
        collection.close()
    report["cold_per_query"] = stats(per_query) | {"open_p50_ms": statistics.median(opens)}

    c, w = report["cold_start"], report["warm"]
    print(f"{args.label or args.collection}: {report['collection_mb']:.0f} MB collection, {report['model_mb']:.0f} MB model, "
          f"{len(queries)} queries")
    print(f"  disk            random 4 KB read {random_us:,.0f} us, sequential {sequential:,.0f} MB/s; "
          f"{report['cached_after_eviction_mb']:.1f} MB cached after eviction")
    print(f"  cold start      open {c['open_ms']:.0f} ms, first query {c['first_query_ms']:.1f} ms, "
          f"first 10 p50 {c['first_10']['p50_ms']:.1f} ms, all p50 {c['all']['p50_ms']:.2f} / p99 {c['all']['p99_ms']:.1f} ms")
    print(f"  warm            p50 {w['p50_ms']:.2f} / p99 {w['p99_ms']:.2f} ms")
    q = report["cold_per_query"]
    print(f"  cold per query  p50 {q['p50_ms']:.1f} / p99 {q['p99_ms']:.1f} ms (open p50 {q['open_p50_ms']:.0f} ms)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
