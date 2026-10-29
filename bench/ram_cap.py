"""Query a collection from inside a cgroup whose memory.max is below the index.

    # one configuration, already inside the cgroup named by --cgroup
    python3 bench/ram_cap.py child COLLECTION MODELS BEIR --warm none --queries 20 --out run.json

    # parent: creates the cgroup, runs child, keeps going if the child is killed
    python3 bench/ram_cap.py run COLLECTION MODELS BEIR --max-mb 128 --warm eager --out run.json

The parent sets memory.max and memory.swap.max=0, then the child moves itself in before
opening the collection. Swap is off so a cap is reclaim, not the machine's swap file.
Major faults are field 12 of /proc/self/stat. Bytes are read_bytes from /proc/self/io.
Linux only. Needs sudo to write the cgroup.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))


def major_faults():
    tail = open("/proc/self/stat").read().rsplit(")", 1)[1].split()
    # Fields after comm: state is stat field 3, majflt is stat field 12.
    return int(tail[9])


def io_bytes():
    out = {}
    for line in open("/proc/self/io"):
        key, value = line.split()
        out[key.rstrip(":")] = int(value)
    return out["read_bytes"]


def cgroup_of_self():
    for line in open("/proc/self/cgroup"):
        path = line.strip().split(":", 2)[2]
        if path.startswith("/"):
            return "/sys/fs/cgroup" + path
    raise SystemExit("no cgroup path")


def child_main(args):
    parent = os.path.dirname(args.cgroup.rstrip("/"))
    os.makedirs(args.cgroup, exist_ok=True)
    subprocess.run(["sudo", "tee", os.path.join(args.cgroup, "memory.max")], input=f"{args.max_mb * 1024 * 1024}\n", text=True, check=True, stdout=subprocess.DEVNULL)
    swap = os.path.join(args.cgroup, "memory.swap.max")
    if os.path.exists(swap):
        subprocess.run(["sudo", "tee", swap], input="0\n", text=True, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["sudo", "tee", os.path.join(args.cgroup, "cgroup.procs")], input=f"{os.getpid()}\n", text=True, check=True, stdout=subprocess.DEVNULL)

    from cold_cache import evict, files
    from gyrfalcon import Collection

    model_dir = os.path.join(args.models, "potion-retrieval-32M")
    with open(os.path.join(args.beir, "qrels", "test.tsv")) as handle:
        next(handle)
        judged = {line.split("\t")[0] for line in handle}
    queries = [q for q in map(json.loads, open(os.path.join(args.beir, "queries.jsonl"))) if q["_id"] in judged]
    queries = [q["text"] for q in sorted(queries, key=lambda q: q["_id"])[: args.queries]]

    evict(args.collection, model_dir)
    faults_before = major_faults()
    bytes_before = io_bytes()
    started = time.perf_counter()
    collection = Collection(args.collection, models=args.models, create=False, warm=args.warm)
    open_ms = (time.perf_counter() - started) * 1000
    open_faults = major_faults() - faults_before
    open_bytes = io_bytes() - bytes_before

    samples = []
    for query in queries:
        faults_before = major_faults()
        bytes_before = io_bytes()
        started = time.perf_counter()
        collection.search(query, k=10)
        samples.append({
            "ms": (time.perf_counter() - started) * 1000,
            "major_faults": major_faults() - faults_before,
            "read_bytes": io_bytes() - bytes_before,
        })
    collection.close()
    events = {}
    events_path = os.path.join(args.cgroup, "memory.events")
    if os.path.exists(events_path):
        for line in open(events_path):
            key, value = line.split()
            events[key] = int(value)
    current = None
    current_path = os.path.join(args.cgroup, "memory.current")
    if os.path.exists(current_path):
        current = int(open(current_path).read().strip())

    def median(values):
        ordered = sorted(values)
        return ordered[len(ordered) // 2]

    def percentile(values, p):
        ordered = sorted(values)
        rank = max(1, int(__import__("math").ceil(p / 100 * len(ordered))))
        return ordered[min(rank, len(ordered)) - 1]

    latencies = [row["ms"] for row in samples]
    report = {
        "label": args.label,
        "collection": args.collection,
        "warm": args.warm,
        "max_mb": args.max_mb,
        "collection_mb": sum(os.path.getsize(p) for p in files(args.collection)) / 1e6,
        "model_mb": sum(os.path.getsize(p) for p in files(model_dir)) / 1e6,
        "queries": len(samples),
        "open_ms": open_ms,
        "open_major_faults": open_faults,
        "open_read_bytes": open_bytes,
        "p50_ms": median(latencies),
        "p99_ms": percentile(latencies, 99),
        "major_faults_p50": median([row["major_faults"] for row in samples]),
        "read_bytes_p50": median([row["read_bytes"] for row in samples]),
        "read_bytes_mean": sum(row["read_bytes"] for row in samples) / len(samples),
        "memory_current": current,
        "memory_events": events,
        "samples": samples,
        "cgroup_parent_note": parent,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(
        f"{args.label}: open {open_ms:.0f} ms ({open_faults} faults, {open_bytes/1e6:.1f} MB), "
        f"p50 {report['p50_ms']:.1f} / p99 {report['p99_ms']:.1f} ms, "
        f"faults/query p50 {report['major_faults_p50']}, "
        f"read/query p50 {report['read_bytes_p50']/1e6:.2f} MB"
    )


def run_main(args):
    base = cgroup_of_self()
    # The workload cgroup already has processes, so the cap is a sibling under its parent,
    # which has the memory controller delegated and no processes of its own.
    parent = os.path.dirname(base)
    name = f"gyr-ram-{args.max_mb}-{args.warm}-{os.getpid()}"
    cgroup = os.path.join(parent, name)
    subprocess.run(["sudo", "mkdir", "-p", cgroup], check=True)
    cmd = [
        sys.executable, os.path.abspath(__file__), "child",
        args.collection, args.models, args.beir,
        "--warm", args.warm, "--max-mb", str(args.max_mb), "--queries", str(args.queries),
        "--cgroup", cgroup, "--label", args.label, "--out", args.out,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
    env["RAYON_NUM_THREADS"] = "4"
    try:
        completed = subprocess.run(cmd, env=env, timeout=args.timeout)
        status = completed.returncode
    except subprocess.TimeoutExpired:
        status = "timeout"
    events = {}
    events_path = os.path.join(cgroup, "memory.events")
    if os.path.exists(events_path):
        for line in open(events_path):
            key, value = line.split()
            events[key] = int(value)
    if status != 0:
        failure = {"label": args.label, "status": status, "memory_events": events, "max_mb": args.max_mb, "warm": args.warm}
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(failure, handle, indent=2)
            handle.write("\n")
        print(f"{args.label}: child status {status} events {events}", file=sys.stderr)
    # The child has exited; drop the empty cgroup.
    subprocess.run(["sudo", "rmdir", cgroup], check=False)
    return 0 if status == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("child", "run"):
        command = sub.add_parser(name)
        command.add_argument("collection")
        command.add_argument("models")
        command.add_argument("beir")
        command.add_argument("--warm", required=True)
        command.add_argument("--max-mb", type=int, required=True)
        command.add_argument("--queries", type=int, default=20)
        command.add_argument("--label", default="")
        command.add_argument("--out", required=True)
        command.add_argument("--cgroup", default="")
        command.add_argument("--timeout", type=int, default=240)
    args = parser.parse_args()
    if args.cmd == "child":
        child_main(args)
        return 0
    return run_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
