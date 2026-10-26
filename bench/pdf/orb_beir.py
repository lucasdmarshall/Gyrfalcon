"""Builds a BEIR dataset from one extractor's text of Open RAG Bench papers, for `gyr bench beir`.

    python orb_beir.py ORB_DIR IDS_FILE TEXT_DIR OUT_DIR

ORB_DIR is Open RAG Bench's `pdf/arxiv` directory (corpus/, queries.json, qrels.json); IDS_FILE
lists the papers extracted into TEXT_DIR as `{id}.txt`. Every extractor's text is cut into
passages by the same rule as Gyrfalcon's chunker (a window of 2,048 bytes, cut at a blank line,
else a newline, else ". "), so each passage is one chunk and the extractors differ only in the
text and line structure they produce.

A query's relevant passages are those of its paper sharing at least half of the word trigrams of
the smaller of the two: the passage, or the query's reference section (a short section fills
part of one passage). Garbled or reordered text has fewer such passages, and a query
left with none keeps a placeholder judgment it can never retrieve, so it scores zero.
"""
import json
import os
import re
import sys

WINDOW = 2048
# GYR_BACK_HALF=1 looks for each cut in the window's back half before the whole window.
BACK_HALF = os.environ.get("GYR_BACK_HALF") == "1"
WORD = re.compile(r"[a-z0-9]+")
MATH = re.compile(r"\$\$.*?\$\$|\$[^$]*\$|\\\[.*?\\\]|\\\(.*?\\\)", re.S)
COMMAND = re.compile(r"\\[a-zA-Z]+")
OVERLAP = 0.5


def trigrams(text):
    text = COMMAND.sub(" ", MATH.sub(" ", text))
    words = [w for w in WORD.findall(text.lower()) if len(w) > 1 and not w.isdigit()]
    return set(zip(words, words[1:], words[2:]))


def passages(text):
    data = text.encode()
    out, start = [], 0
    while start < len(data):
        end = min(start + WINDOW, len(data))
        if end < len(data):
            window = data[start:end]
            floors = (len(window) // 2, 1) if BACK_HALF else (1,)
            cut = None
            for floor in floors:
                for mark in (b"\n\n", b"\n", b". "):
                    at = window.rfind(mark)
                    if at >= floor:
                        cut = at + len(mark)
                        break
                if cut:
                    break
            if cut:
                end = start + cut
        piece = data[start:end].decode(errors="ignore").strip()
        if piece:
            out.append(piece)
        start = end
    return out


orb, ids_file, text_dir, out = sys.argv[1:5]
ids = open(ids_file).read().split()
os.makedirs(os.path.join(out, "qrels"), exist_ok=True)
paper_passages = {}
with open(os.path.join(out, "corpus.jsonl"), "w") as corpus:
    for pid in ids:
        path = os.path.join(text_dir, pid + ".txt")
        text = open(path, errors="replace").read() if os.path.exists(path) else ""
        paper_passages[pid] = []
        for n, body in enumerate(passages(text)):
            doc = f"{pid}#{n}"
            paper_passages[pid].append((doc, trigrams(body)))
            corpus.write(json.dumps({"_id": doc, "title": "", "text": body}) + "\n")

queries = json.load(open(os.path.join(orb, "queries.json")))
qrels = json.load(open(os.path.join(orb, "qrels.json")))
sections = {}
judged = unanswerable = 0
with open(os.path.join(out, "queries.jsonl"), "w") as qf, open(os.path.join(out, "qrels", "test.tsv"), "w") as rf:
    rf.write("query-id\tcorpus-id\tscore\n")
    for qid, target in qrels.items():
        pid = target["doc_id"]
        if pid not in paper_passages:
            continue
        if pid not in sections:
            paper = json.load(open(os.path.join(orb, "corpus", pid + ".json")))
            sections[pid] = [trigrams(s["text"]) for s in paper["sections"]]
        section = sections[pid][target["section_id"]]
        relevant = [
            doc
            for doc, grams in paper_passages[pid]
            if grams and section and len(grams & section) >= OVERLAP * min(len(grams), len(section))
        ]
        if not relevant:
            relevant = [f"{pid}#unanswerable"]
            unanswerable += 1
        judged += 1
        qf.write(json.dumps({"_id": qid, "text": queries[qid]["query"]}) + "\n")
        for doc in relevant:
            rf.write(f"{qid}\t{doc}\t1\n")
print(json.dumps({"passages": sum(len(p) for p in paper_passages.values()), "queries": judged, "unanswerable": unanswerable}))
