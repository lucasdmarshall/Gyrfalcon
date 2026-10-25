"""Shared pieces of the head-to-head: BEIR loading, the two embedders, run files, metrics.

Every framework gets the same embedder object, so differences come from the framework, not
from the model or its runtime. Metrics follow pytrec_eval and `gyr bench beir`: graded nDCG@10
with linear gain, recall@100, duplicate doc ids dropped, and a retrieved doc whose id equals the
query id ignored (ArguAna).
"""
import json
import math
import os
import time

import numpy as np


def load_beir(path):
    docs = []
    for line in open(os.path.join(path, "corpus.jsonl")):
        d = json.loads(line)
        title = (d.get("title") or "").strip()
        text = d.get("text") or ""
        docs.append((d["_id"], f"{title}\n\n{text}" if title else text))
    qrels = {}
    with open(os.path.join(path, "qrels", "test.tsv")) as f:
        next(f)
        for line in f:
            q, doc, score = line.rstrip("\n").split("\t")[:3]
            qrels.setdefault(q, {})[doc] = int(score)
    queries = []
    for line in open(os.path.join(path, "queries.jsonl")):
        q = json.loads(line)
        if q["_id"] in qrels:
            queries.append((q["_id"], q["text"]))
    queries.sort()
    return docs, queries, qrels


class Embedder:
    """`embed(texts) -> np.ndarray` of L2-normalized rows."""

    def __init__(self, name, threads):
        self.name = name
        if name == "bge-small":
            from fastembed import TextEmbedding

            self.model = TextEmbedding("BAAI/bge-small-en-v1.5", threads=threads)
            self.dim = 384
        elif name == "potion":
            from model2vec import StaticModel

            here = os.path.dirname(os.path.abspath(__file__))
            self.model = StaticModel.from_pretrained(os.path.join(here, "..", "..", ".models", "potion-retrieval-32M"))
            self.dim = self.model.dim
        else:
            raise ValueError(name)

    def embed(self, texts):
        if self.name == "bge-small":
            rows = np.array(list(self.model.embed(list(texts), batch_size=32)), dtype=np.float32)
        else:
            rows = np.asarray(self.model.encode(list(texts)), dtype=np.float32)
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        return rows / np.maximum(norms, 1e-12)


def write_run(path, ranked_by_query):
    with open(path, "w") as f:
        for qid, docs in ranked_by_query.items():
            for rank, doc in enumerate(docs):
                f.write(f"{qid}\t{doc}\t{rank + 1}\t{len(docs) - rank}\n")


def read_run(path):
    run = {}
    for line in open(path):
        qid, doc, rank, _ = line.rstrip("\n").split("\t")
        run.setdefault(qid, []).append((int(rank), doc))
    return {q: [d for _, d in sorted(v)] for q, v in run.items()}


def clean(qid, docs):
    seen, out = set(), []
    for doc in docs:
        if doc != qid and doc not in seen:
            seen.add(doc)
            out.append(doc)
    return out


def ndcg_at(ranked, judged, k=10):
    dcg = sum(max(judged.get(d, 0), 0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((g for g in judged.values() if g > 0), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at(ranked, judged, k=100):
    relevant = sum(1 for g in judged.values() if g > 0)
    if relevant == 0:
        return 0.0
    return sum(1 for d in ranked[:k] if judged.get(d, 0) > 0) / relevant


def evaluate(run, qrels, queries):
    n = len(queries)
    ndcg = sum(ndcg_at(clean(q, run.get(q, [])), qrels[q]) for q, _ in queries) / n
    recall = sum(recall_at(clean(q, run.get(q, [])), qrels[q]) for q, _ in queries) / n
    return {"ndcg_at_10": ndcg, "recall_at_100": recall}


def timed_queries(queries, search):
    """Runs `search(text) -> [doc ids]` once per query, sequentially."""
    run, millis = {}, []
    for qid, text in queries:
        started = time.perf_counter()
        run[qid] = search(text)
        millis.append((time.perf_counter() - started) * 1000)
    millis.sort()
    return run, {
        "query_p50_ms": millis[len(millis) // 2],
        "query_p99_ms": millis[min(len(millis) - 1, int(len(millis) * 0.99))],
        "query_mean_ms": sum(millis) / len(millis),
    }
