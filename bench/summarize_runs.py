"""Median and range over repeated `gyr bench beir` reports, as a Markdown table.

    python3 bench/summarize_runs.py RUN1.json RUN2.json ...

Each report must cover the same datasets in the same order. nDCG@10 is printed once when every
run agrees (it is deterministic for the lite tier) and as a range otherwise.
"""
import json
import statistics
import sys


def spread(values, digits):
    values = sorted(values)
    median = statistics.median(values)
    return f"{median:.{digits}f} ({values[0]:.{digits}f} to {values[-1]:.{digits}f})"


def main(paths):
    reports = [json.load(open(p)) for p in paths]
    env = reports[0]["environment"]
    revision = env.get("git_revision", "unknown")[:7]
    print(f"{len(reports)} runs, commit {revision}, {env.get('model_id')}, "
          f"{env.get('physical_cores')} cores, {env.get('cpu_model')}")
    print()
    print("| Dataset | Documents | Queries | nDCG@10 | Index s, median (range) "
          "| Query p50 ms, median (range) | Query p99 ms, median (range) |")
    print("|---|---|---|---|---|---|---|")
    for i, first in enumerate(reports[0]["datasets"]):
        runs = [r["datasets"][i] for r in reports]
        if any(run["dataset"] != first["dataset"] for run in runs):
            sys.exit(f"reports disagree on dataset {i}")
        rows = [run["results"][0] for run in runs]
        ndcg = sorted(row["ndcg_at_10"] for row in rows)
        ndcg_cell = f"{ndcg[0]:.4f}" if ndcg[0] == ndcg[-1] else f"{ndcg[0]:.4f} to {ndcg[-1]:.4f}"
        print(f"| {first['dataset']} | {first['docs']:,} | {first['queries']:,} | {ndcg_cell} "
              f"| {spread([run['index_seconds'] for run in runs], 2)} "
              f"| {spread([row['query_p50_ms'] for row in rows], 2)} "
              f"| {spread([row['query_p99_ms'] for row in rows], 2)} |")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
