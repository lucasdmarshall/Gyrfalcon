"""Open time and the first ten queries for warm=none, background, and eager.

    python3 bench/warm_modes.py COLLECTION MODELS BEIR --out FILE [--repeats 3] [--label NAME]

COLLECTION is a lite collection (often a bench/slowfs.py mount). MODELS holds
potion-retrieval-32M. BEIR supplies the test queries, taken in id order. Before every open the
collection and the model are dropped from the page cache with posix_fadvise(DONTNEED).

Each repeat records the time until Collection() returns and the latency of the next ten
searches. background's prefetch keeps running during those searches. The process is one repeat
at a time; close() joins a background read before the next open.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from gyrfalcon import Collection

from cold_cache import cached_bytes, disk_probe, evict, files, stats


def load_queries(beir, n):
    with open(os.path.join(beir, "qrels", "test.tsv")) as handle:
        next(handle)
        judged = {line.split("\t")[0] for line in handle}
    queries = [q for q in map(json.loads, open(os.path.join(beir, "queries.jsonl"))) if q["_id"] in judged]
    queries = sorted(queries, key=lambda q: q["_id"])[:n]
    return [q["text"] for q in queries]


def once(collection_dir, models, queries, mode):
    model_dir = os.path.join(models, "potion-retrieval-32M")
    evict(collection_dir, model_dir)
    cached = cached_bytes(collection_dir, model_dir)
    started = time.perf_counter()
    collection = Collection(collection_dir, models=models, create=False, warm=mode)
    open_ms = (time.perf_counter() - started) * 1000
    samples = []
    for query in queries:
        started = time.perf_counter()
        collection.search(query, k=10)
        samples.append((time.perf_counter() - started) * 1000)
    collection.close()
    return {
        "open_ms": open_ms,
        "cached_after_eviction_mb": cached / 1e6,
        "queries_ms": samples,
        "first_query_ms": samples[0],
        "first_10": stats(samples),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("collection")
    parser.add_argument("models")
    parser.add_argument("beir")
    parser.add_argument("--modes", default="none,background,eager")
    parser.add_argument("--queries", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--label", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    queries = load_queries(args.beir, args.queries)
    largest = max(files(args.collection), key=os.path.getsize)
    random_us, sequential = disk_probe(largest)
    report = {
        "label": args.label,
        "collection": args.collection,
        "beir": args.beir,
        "collection_mb": sum(os.path.getsize(p) for p in files(args.collection)) / 1e6,
        "model_mb": sum(os.path.getsize(p) for p in files(os.path.join(args.models, "potion-retrieval-32M"))) / 1e6,
        "queries": len(queries),
        "repeats": args.repeats,
        "disk": {"random_4k_read_us": random_us, "sequential_mb_per_s": sequential},
        "modes": {},
    }
    for mode in args.modes.split(","):
        runs = [once(args.collection, args.models, queries, mode) for _ in range(args.repeats)]
        def median(values):
            ordered = sorted(values)
            return ordered[len(ordered) // 2]

        report["modes"][mode] = {
            "runs": runs,
            "open_ms_median": median(run["open_ms"] for run in runs),
            "first_query_ms_median": median(run["first_query_ms"] for run in runs),
            "first_10_p50_median": median(run["first_10"]["p50_ms"] for run in runs),
        }
        summary = report["modes"][mode]
        print(
            f"{args.label or args.collection} {mode}: open {summary['open_ms_median']:.0f} ms, "
            f"first query {summary['first_query_ms_median']:.1f} ms, "
            f"first 10 p50 {summary['first_10_p50_median']:.1f} ms "
            f"(disk {random_us:,.0f} us, {sequential:,.0f} MB/s)"
        )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
