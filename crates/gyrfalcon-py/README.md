# gyrfalcon

Hybrid retrieval for RAG (BM25 and dense vectors, fused) in one Rust engine, built to be fast
on any CPU. One `pip install`, no server, no vector database.

```python
from gyrfalcon import Collection

with Collection("./my-index") as col:              # creates the collection on first use
    col.add("intro", "Gyrfalcons are the largest falcons.")
    col.add_many({"a": "first text", "b": "second text"})
    col.add_path("docs/")                           # PDF, HTML, Markdown, and text files
    for hit in col.search("largest falcon", k=5):
        print(f"{hit.score:.3f} {hit.doc_id} {hit.text[:80]}")
```

- `update(doc_id, text)` replaces a document atomically; `delete(doc_id)` removes it.
- Writes are durable when the call returns; searches see them at once.
- Two tiers. `lite` (default) embeds with static potion-retrieval-32M and needs no runtime.
  `full` embeds with int8 bge-small-en-v1.5 and needs ONNX Runtime: `pip install gyrfalcon[full]`.
- PDFs need PDFium: `pip install gyrfalcon[pdf]`.
- Models download on first use into `$GYR_MODELS` (default `~/.cache/gyrfalcon/models`).

Building from the repository: `pip install maturin && maturin develop --release` in
`crates/gyrfalcon-py`, then `pytest crates/gyrfalcon-py/tests`.
