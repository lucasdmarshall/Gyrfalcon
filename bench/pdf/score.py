"""Scores extracted PDF text against Open RAG Bench's reference text of the same arXiv paper.

    python score.py REFERENCE_DIR OUT_DIR TIMINGS.jsonl...

REFERENCE_DIR holds `{arxiv id}.json` (title, abstract, sections). For each extractor:

- word F1: multiset overlap of lowercase alphanumeric words, extracted vs reference;
- trigram recall: share of the reference's word trigrams found in the extraction, which drops
  when columns interleave, hyphenation splits words, or words run together;
- pages/s and MB/s from the timing lines, and the files that failed.

The reference is the benchmark's own Markdown rendering, so no extractor can reach 1.0; the
numbers rank extractors against one target.
"""
import json
import os
import re
import sys
from collections import Counter

WORD = re.compile(r"[a-z0-9]+")


def words(text):
    return WORD.findall(text.lower())


def reference(path):
    d = json.load(open(path))
    parts = [d.get("title", ""), d.get("abstract", "")] + [s.get("text", "") for s in d.get("sections", [])]
    return "\n".join(parts)


def scores(ref_words, got_words):
    ref, got = Counter(ref_words), Counter(got_words)
    overlap = sum((ref & got).values())
    precision = overlap / max(sum(got.values()), 1)
    recall = overlap / max(sum(ref.values()), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    ref3 = Counter(zip(ref_words, ref_words[1:], ref_words[2:]))
    got3 = Counter(zip(got_words, got_words[1:], got_words[2:]))
    tri = sum((ref3 & got3).values()) / max(sum(ref3.values()), 1)
    return f1, tri


ref_dir, out_dir = sys.argv[1], sys.argv[2]
pages = {}
pages_dir = os.path.join(out_dir, "pages")
for name in os.listdir(pages_dir):
    pages[name[:-4]] = int(open(os.path.join(pages_dir, name)).read())
rows = []
for timing in sys.argv[3:]:
    lines = [json.loads(l) for l in open(timing) if l.startswith("{")]
    extractor = lines[0]["extractor"]
    f1s, tris, ms, mb, pg, errors = [], [], 0.0, 0.0, 0, 0
    for line in lines:
        ms += line["ms"]
        mb += line["bytes"] / 1e6
        pg += pages.get(line["file"], 0)
        if line["error"]:
            errors += 1
            f1s.append(0.0)
            tris.append(0.0)
            continue
        got = open(os.path.join(out_dir, extractor, line["file"] + ".txt"), errors="replace").read()
        f1, tri = scores(words(reference(os.path.join(ref_dir, line["file"] + ".json"))), words(got))
        f1s.append(f1)
        tris.append(tri)
    rows.append((extractor, sum(f1s) / len(f1s), sum(tris) / len(tris), pg / (ms / 1e3), mb / (ms / 1e3), errors, len(lines)))

print("| extractor | word F1 | trigram recall | pages/s | MB/s | failed files |")
print("|---|---|---|---|---|---|")
for e, f1, tri, pps, mbs, err, n in sorted(rows, key=lambda r: -r[1]):
    print(f"| {e} | {f1:.4f} | {tri:.4f} | {pps:.1f} | {mbs:.2f} | {err}/{n} |")
