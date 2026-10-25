"""Head-to-head grid: every system on every dataset, pinned to N cores, one process per cell.

    python compare.py OUT_DIR [--cores 1,4] [--datasets scifact,nfcorpus] [--embedders potion]
                              [--systems gyrfalcon,llamaindex,langchain,haystack]

Each cell runs in its own process under `taskset`, with every thread pool capped to the core
count. Peak RSS is the child's `ru_maxrss`. All run files are scored by `common.evaluate`, so
Gyrfalcon's numbers come from the same evaluator as everyone else's.
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time

from common import evaluate, load_beir, read_run

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
PYTHON = sys.executable


def run_cell(command, cores, log):
    env = dict(os.environ)
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS", "GYR_ORT_THREADS"]:
        env[name] = str(cores)
    env.setdefault("ORT_DYLIB_PATH", os.path.join(ROOT, ".models", "ort", "libonnxruntime.so"))
    pinned = ["taskset", "-c", f"0-{cores - 1}", *command]
    started = time.perf_counter()
    # A fresh interpreter per cell, so RUSAGE_CHILDREN holds this cell's peak only.
    probe = (
        "import resource,subprocess,sys,json;"
        "r=subprocess.run(sys.argv[2:],stdout=open(sys.argv[1],'w'),stderr=subprocess.STDOUT);"
        "print(json.dumps({'code':r.returncode,'peak_rss_mb':resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss/1024}))"
    )
    out = subprocess.run([PYTHON, "-c", probe, log, *pinned], env=env, capture_output=True, text=True, cwd=ROOT)
    measured = json.loads(out.stdout.strip().splitlines()[-1])
    measured["wall_seconds"] = time.perf_counter() - started
    return measured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--cores", default="1,4")
    parser.add_argument("--datasets", default="scifact,nfcorpus,arguana,fiqa")
    parser.add_argument("--embedders", default="potion")
    parser.add_argument("--systems", default="gyrfalcon,llamaindex,langchain,haystack")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    results_path = os.path.join(args.out_dir, "results.jsonl")
    done = set()
    if os.path.exists(results_path):
        for line in open(results_path):
            r = json.loads(line)
            done.add((r["system"], r["dataset"], r["embedder"], r["threads"]))

    for cores in [int(c) for c in args.cores.split(",")]:
        for embedder in args.embedders.split(","):
            for dataset in args.datasets.split(","):
                beir = os.path.join(ROOT, ".data", "beir", dataset)
                _, queries, qrels = load_beir(beir)
                for system in args.systems.split(","):
                    key = (system, dataset, embedder, cores)
                    if key in done:
                        continue
                    prefix = os.path.join(os.path.abspath(args.out_dir), f"{system}-{dataset}-{embedder}-{cores}core")
                    if system == "gyrfalcon":
                        tier = "lite" if embedder == "potion" else "full"
                        run_dir = prefix + "-runs"
                        command = [os.path.join(ROOT, "target", "release", "gyr"), "bench", "beir", "--tier", tier,
                                   "--dataset", beir, "--fusions", "configured", "--run-dir", run_dir,
                                   "--out", prefix + ".json"]
                    else:
                        command = [PYTHON, os.path.join(HERE, "run_framework.py"), system, beir, embedder,
                                   str(cores), prefix]
                    measured = run_cell(command, cores, prefix + ".log")
                    if measured["code"] != 0:
                        print(f"FAILED {key}, see {prefix}.log", flush=True)
                        continue
                    if system == "gyrfalcon":
                        g = json.load(open(prefix + ".json"))["datasets"][0]
                        row = g["results"][0]
                        report = {
                            "index_seconds": g["index_seconds"],
                            "query_p50_ms": row["query_p50_ms"],
                            "query_p99_ms": row["query_p99_ms"],
                            "gyr_ndcg_at_10": row["ndcg_at_10"],
                        }
                        run = read_run(os.path.join(run_dir, f"{dataset}-configured.tsv"))
                    else:
                        report = json.load(open(prefix + ".json"))
                        run = read_run(prefix + ".tsv")
                    report.update(evaluate(run, qrels, queries))
                    report.update(measured)
                    report.update({"system": system, "dataset": dataset, "embedder": embedder, "threads": cores})
                    with open(results_path, "a") as f:
                        f.write(json.dumps(report) + "\n")
                    print(json.dumps({k: report[k] for k in ["system", "dataset", "embedder", "threads",
                                                             "index_seconds", "query_p50_ms", "ndcg_at_10",
                                                             "peak_rss_mb"]}), flush=True)


if __name__ == "__main__":
    main()
