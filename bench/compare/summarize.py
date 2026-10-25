"""Markdown tables from a `compare.py` results file.

    python summarize.py OUT_DIR/results.jsonl
"""
import json
import sys
from collections import defaultdict

rows = [json.loads(line) for line in open(sys.argv[1])]
systems = ["gyrfalcon", "llamaindex", "langchain", "haystack"]
cells = defaultdict(dict)
for r in rows:
    cells[(r["embedder"], r["threads"], r["dataset"])][r["system"]] = r


def table(title, key, fmt, better):
    print(f"\n### {title}\n")
    print("| embedder | cores | dataset | " + " | ".join(systems) + " |")
    print("|---|---|---|" + "---|" * len(systems))
    for (embedder, threads, dataset), by_system in sorted(cells.items()):
        values = {s: by_system[s][key] for s in systems if s in by_system}
        best = (max if better == "high" else min)(values.values()) if values else None
        shown = []
        for s in systems:
            if s not in values:
                shown.append("-")
                continue
            text = fmt.format(values[s])
            shown.append(f"**{text}**" if values[s] == best else text)
        print(f"| {embedder} | {threads} | {dataset} | " + " | ".join(shown) + " |")


table("nDCG@10 (higher is better)", "ndcg_at_10", "{:.4f}", "high")
table("Recall@100", "recall_at_100", "{:.4f}", "high")
table("Index build, seconds (lower is better)", "index_seconds", "{:.2f}", "low")
table("Query p50, ms (lower is better)", "query_p50_ms", "{:.2f}", "low")
table("Query p99, ms", "query_p99_ms", "{:.2f}", "low")
table("Peak RSS, MB (lower is better)", "peak_rss_mb", "{:.0f}", "low")
