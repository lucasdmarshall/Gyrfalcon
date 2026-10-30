#!/usr/bin/env python3
"""Replay a cross-encoder depth sweep and a selective-rerank estimate from trace jsonl.

This does not add a planner. It reads traces written by `gyr bench trace` and applies
`Policy::RescoreMax` (lite, dense weight 0.45) or `Policy::Rescore` (full, dense weight
0.65) the way `gyrfalcon::planner::replay` does.

Latency is the sum of the stored step times. `rerank_per_pair` was timed on the trace's
union batch (fused heads at weights 0.2, 0.4 and 0.6, plus the configured fusion's head),
not on a fresh batch of exactly K pairs.

The selective threshold is chosen on validation queries of the tuning datasets only
(scifact, nfcorpus, arguana, fiqa). Held-out datasets are scored with that threshold and
do not move it. The acceptance band is predeclared: pooled tuning-validation nDCG@10 must
stay within 0.005 of always reranking at the same K. Among thresholds inside the band,
the one with the smallest total rerank milliseconds wins. If none qualify, the smallest
drop is reported and the band is marked missed.

Overlap is the Jaccard of the BM25 top 10 and the dense top 10. Margin is the configured
fusion's rank-1 score minus its rank-2 score. A query is reranked when overlap < T, or,
as a separate rule, when margin < T.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from collections import defaultdict
from pathlib import Path

TUNING = ("scifact", "nfcorpus", "arguana", "fiqa")
HELDOUT = ("trec-covid", "scidocs", "quora", "webis-touche2020")
DEPTHS = (0, 8, 16, 20, 32, 50)
OVERLAP_GRID = tuple(i / 10 for i in range(0, 12))  # 0.0 .. 1.1
BAND = 0.005
LIMIT = 100
LITE = ("rescoremax", 0.45)
FULL = ("rescore", 0.65)


def f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


def ndcg_at(ranked: list[str], qrels: dict, k: int = 10) -> float:
    def gain(doc: str) -> float:
        return max(int(qrels.get(doc, 0)), 0)

    dcg = sum(gain(doc) / math.log2(i + 2) for i, doc in enumerate(ranked[:k]))
    ideal = sorted((max(int(g), 0) for g in qrels.values() if int(g) > 0), reverse=True)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at(ranked: list[str], qrels: dict, k: int = 100) -> float:
    relevant = sum(1 for g in qrels.values() if int(g) > 0)
    if relevant == 0:
        return 0.0
    found = sum(1 for doc in ranked[:k] if int(qrels.get(doc, 0)) > 0)
    return found / relevant


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    rank = max(1, math.ceil((p / 100.0) * len(sorted_values)))
    return sorted_values[min(rank, len(sorted_values)) - 1]


def min_max(pairs: list[tuple[int, float]]) -> list[tuple[int, float]]:
    if not pairs:
        return []
    lo = min(score for _, score in pairs)
    hi = max(score for _, score in pairs)
    span = f32(hi - lo)
    out = []
    for chunk, score in pairs:
        scaled = f32(f32(score - lo) / span) if span > 0 else f32(1.0)
        out.append((chunk, scaled))
    return out


def score_fusion(lexical: list[tuple[int, float]], dense: list[tuple[int, float]], weight: float) -> list[tuple[int, float]]:
    weight = f32(weight)
    lexical = min_max(lexical)
    dense = min_max(dense)
    scores: dict[int, float] = {}
    for chunk, value in lexical:
        scores[chunk] = f32(scores.get(chunk, 0.0) + f32((1.0 - weight) * value))
    for chunk, value in dense:
        scores[chunk] = f32(scores.get(chunk, 0.0) + f32(weight * value))
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


def max_fusion(lexical: list[tuple[int, float]], sims: list[tuple[int, float]], weight: float) -> list[tuple[int, float]]:
    weight = f32(weight)
    top = max((score for _, score in lexical), default=0.0)
    table = {chunk: (f32(score / top) if top > 0 else f32(0.0)) for chunk, score in lexical}
    out = []
    for chunk, sim in sims:
        lex = table.get(chunk, 0.0)
        out.append((chunk, f32(f32((1.0 - weight) * lex) + f32(weight * sim))))
    out.sort(key=lambda item: (-item[1], item[0]))
    return out


def fused_order(trace: dict, kind: str, weight: float) -> tuple[list[str], list[float]]:
    """Document order and fused scores, matching planner.rs rescore_order / max_order."""
    chunk_of = {doc: int(chunk) for doc, chunk in trace["chunks"].items()}

    def chunk(doc: str) -> int:
        return chunk_of.get(doc, 2**64 - 1)

    if kind == "rescore":
        ids = sorted({doc for doc, _ in trace["lexical"]} | {doc for doc, _ in trace["dense"]})
        # HashMap collect keeps the last document that shares a chunk id.
        doc_of = {}
        for doc in ids:
            doc_of[chunk(doc)] = doc
        lexical = [(chunk(doc), f32(score)) for doc, score in trace["lexical"]]
        exact = [(chunk(doc), f32(trace["exact"][doc])) for doc in ids if doc in trace["exact"]]
        fused = score_fusion(lexical, exact, weight)
    else:
        docs = sorted(trace["exact"])
        doc_of = {}
        for doc in docs:
            doc_of[chunk(doc)] = doc
        lexical = [(chunk(doc), f32(score)) for doc, score in trace["lexical"]]
        sims = [(chunk(doc), f32(trace["exact"][doc])) for doc in docs]
        fused = max_fusion(lexical, sims, weight)
    ordered = [doc_of[chunk_id] for chunk_id, _ in fused if chunk_id in doc_of]
    scores = [score for chunk_id, score in fused if chunk_id in doc_of]
    return ordered, scores


def replay(trace: dict, kind: str, weight: float, rerank_k: int) -> tuple[list[str], float]:
    timings = trace["timings"]
    ranked, _scores = fused_order(trace, kind, weight)
    cost = timings["prepare"] + timings["lexical"] + timings["embed"] + timings["dense"] + timings["exact"]
    if rerank_k > 0 and ranked:
        k = min(rerank_k, len(ranked))
        head, tail = ranked[:k], ranked[k:]
        rerank = trace.get("rerank") or {}

        def key(doc: str):
            return rerank.get(doc, float("-inf"))

        head.sort(key=key, reverse=True)
        ranked = head + tail
        cost += k * timings["rerank_per_pair"]
    seen = set()
    deduped = []
    for doc in ranked:
        if doc not in seen:
            seen.add(doc)
            deduped.append(doc)
    cost += timings.get("overhead", 0.0) + min(len(deduped), LIMIT) * timings.get("fetch_per_hit", 0.0)
    ranked = [doc for doc in deduped if doc != trace["query_id"]][:LIMIT]
    return ranked, cost


def jaccard10(trace: dict) -> float:
    lexical = {doc for doc, _ in trace["lexical"][:10]}
    dense = {doc for doc, _ in trace["dense"][:10]}
    union = lexical | dense
    return (len(lexical & dense) / len(union)) if union else 1.0


def margin_of(scores: list[float]) -> float:
    if len(scores) < 2:
        return 1.0
    return scores[0] - scores[1]


def load_traces(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def official_test(trace: dict) -> bool:
    """True when the query belongs to the BEIR test file, including a hash-split half."""
    if trace["dataset"] in ("scifact", "nfcorpus", "fiqa"):
        return trace["split"] == "test"
    return True


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"queries": 0}
    costs = sorted(row["cost_ms"] for row in rows)
    n = len(rows)
    return {
        "queries": n,
        "ndcg_at_10": sum(row["ndcg_at_10"] for row in rows) / n,
        "recall_at_100": sum(row["recall_at_100"] for row in rows) / n,
        "p50_ms": percentile(costs, 50),
        "p99_ms": percentile(costs, 99),
        "mean_ms": sum(costs) / n,
        "rerank_ms": sum(row["rerank_ms"] for row in rows),
        "reranked_queries": sum(1 for row in rows if row["reranked"]),
    }


def score_query(trace: dict, kind: str, weight: float, k: int, rerank: bool) -> dict:
    ranked, cost = replay(trace, kind, weight, k if rerank else 0)
    base_ranked, base_cost = (ranked, cost) if not rerank else replay(trace, kind, weight, 0)
    rerank_ms = max(0.0, cost - base_cost) if rerank else 0.0
    return {
        "dataset": trace["dataset"],
        "split": trace["split"],
        "query_id": trace["query_id"],
        "ndcg_at_10": ndcg_at(ranked, trace["qrels"]),
        "recall_at_100": recall_at(base_ranked, trace["qrels"]),
        "cost_ms": cost,
        "rerank_ms": rerank_ms,
        "reranked": rerank,
        "ms_per_pair": trace["timings"].get("rerank_per_pair", 0.0),
    }


def attach_features(traces: list[dict], kind: str, weight: float) -> None:
    for trace in traces:
        trace["_overlap"] = jaccard10(trace)
        _order, scores = fused_order(trace, kind, weight)
        trace["_margin"] = margin_of(scores)


def fit_threshold(traces: list[dict], kind: str, weight: float, k: int, rule: str) -> dict:
    """Pick T on tuning validation only. `traces` must already carry features."""
    pool = [t for t in traces if t["dataset"] in TUNING and t["split"] == "validation"]
    always = [score_query(t, kind, weight, k, True) for t in pool]
    always_ndcg = summarize(always)["ndcg_at_10"] if always else 0.0
    if rule == "overlap":
        grid = OVERLAP_GRID
        trigger = lambda trace, threshold: trace["_overlap"] < threshold
    else:
        peak = max((t["_margin"] for t in pool), default=0.0)
        step = peak / 20 if peak > 0 else 1.0
        grid = tuple(step * i for i in range(0, 22))
        trigger = lambda trace, threshold: trace["_margin"] < threshold
    best = None
    tried = []
    for threshold in grid:
        scored = [score_query(t, kind, weight, k, trigger(t, threshold)) for t in pool]
        summary = summarize(scored)
        drop = always_ndcg - summary.get("ndcg_at_10", 0.0)
        row = {"threshold": threshold, "drop": drop, "inside_band": drop <= BAND, **summary}
        tried.append(row)
        if row["inside_band"] and (best is None or row["rerank_ms"] < best["rerank_ms"]):
            best = row
    if best is None and tried:
        best = min(tried, key=lambda row: (row["drop"], row["rerank_ms"]))
        best = {**best, "band_missed": True}
    elif best is not None:
        best = {**best, "band_missed": False}
    return {
        "rule": rule,
        "k": k,
        "validation_queries": len(pool),
        "always_ndcg_at_10": always_ndcg,
        "chosen": best,
        "grid": tried,
    }


def apply_rule(traces: list[dict], kind: str, weight: float, k: int, rule: str, threshold: float) -> list[dict]:
    if rule == "overlap":
        trigger = lambda trace: trace["_overlap"] < threshold
    else:
        trigger = lambda trace: trace["_margin"] < threshold
    return [score_query(t, kind, weight, k, trigger(t)) for t in traces]


def dataset_rows(scored: list[dict]) -> dict:
    by_dataset = defaultdict(list)
    for row in scored:
        by_dataset[row["dataset"]].append(row)
    return {name: summarize(rows) for name, rows in sorted(by_dataset.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path, help="jsonl files from gyr bench trace")
    parser.add_argument("--tier", choices=("lite", "full"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    kind, weight = LITE if args.tier == "lite" else FULL
    traces = load_traces(args.traces)
    attach_features(traces, kind, weight)
    test = [t for t in traces if official_test(t)]
    sweep = []
    for k in DEPTHS:
        scored = [score_query(t, kind, weight, k, k > 0) for t in test]
        sweep.append({"k": k, "datasets": dataset_rows(scored), "all_test": summarize(scored)})
    selective = []
    for k in DEPTHS:
        if k == 0:
            continue
        for rule in ("overlap", "margin"):
            fit = fit_threshold(traces, kind, weight, k, rule)
            chosen = fit["chosen"] or {"threshold": 0.0}
            threshold = chosen["threshold"]
            # Tuning sets: the hash/official test half, which the fit did not see.
            tuning_test = [t for t in traces if t["dataset"] in TUNING and t["split"] == "test"]
            # Held-out: the whole BEIR test file. Its queries were not in the fit.
            heldout = [t for t in traces if t["dataset"] in HELDOUT and official_test(t)]
            selective.append(
                {
                    "k": k,
                    "rule": rule,
                    "fit": {key: fit[key] for key in ("validation_queries", "always_ndcg_at_10", "chosen")},
                    "tuning_test": dataset_rows(apply_rule(tuning_test, kind, weight, k, rule, threshold)),
                    "heldout_test": dataset_rows(apply_rule(heldout, kind, weight, k, rule, threshold)),
                    "always_tuning_test": dataset_rows(
                        [score_query(t, kind, weight, k, True) for t in tuning_test]
                    ),
                    "always_heldout_test": dataset_rows(
                        [score_query(t, kind, weight, k, True) for t in heldout]
                    ),
                    "never_tuning_test": dataset_rows(
                        [score_query(t, kind, weight, k, False) for t in tuning_test]
                    ),
                    "never_heldout_test": dataset_rows(
                        [score_query(t, kind, weight, k, False) for t in heldout]
                    ),
                }
            )
    report = {
        "tier": args.tier,
        "fusion": {"kind": kind, "dense_weight": weight},
        "queries": len(traces),
        "test_queries": len(test),
        "band": BAND,
        "note": (
            "Latency is replayed step time. rerank_per_pair is from the union batch, "
            "not a fresh batch of K pairs. The selective threshold was fit on tuning "
            "validation only, with a predeclared 0.005 nDCG band."
        ),
        "sweep": sweep,
        "selective": selective,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.out} ({len(traces)} traces)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
