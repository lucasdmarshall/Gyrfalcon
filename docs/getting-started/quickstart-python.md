<sub>[Docs](../README.md) / Getting started</sub>

# Python quickstart

This page takes you from `pip install` to filtered hybrid search over your own files. It uses
the default `lite` tier, which needs no model runtime and runs on any CPU.

## 1. Install

```sh
pip install gyrfalcon
```

Add `[pdf]` if you will index PDF files. See [Installation](installation.md) for the other
options.

## 2. Create a collection and add text

A collection is a directory. Opening one that does not exist creates it.

```python
from gyrfalcon import Collection

col = Collection("./my-index")

col.add("falcon", "The gyrfalcon is the largest of the falcon species.",
        metadata={"topic": "birds", "year": 2024})
col.add_many({
    "peregrine": "Peregrine falcons dive faster than any other animal.",
    "owl": "Owls hunt at night and fly almost silently.",
})
```

The first call downloads the lite model (about 130 MB) into `~/.cache/gyrfalcon/models`.
Later opens take tens of milliseconds.

Each write is durable when the call returns, and the next search sees it.

## 3. Search

```python
for hit in col.search("which bird is the fastest?", k=3):
    print(f"{hit.score:.3f}  {hit.doc_id:10}  {hit.text}")
```

```text
0.140  peregrine   Peregrine falcons dive faster than any other animal.
0.109  falcon      The gyrfalcon is the largest of the falcon species.
0.061  owl         Owls hunt at night and fly almost silently.
```

Scores are only comparable within one query. Their scale depends on the tier's fusion rule and
on the collection, so do not use a fixed score threshold across queries.

Every query runs a BM25 search and a dense vector search and fuses the two lists. You do not
choose between keyword and semantic search; you get both. [Hybrid retrieval](../concepts/hybrid-retrieval.md)
explains how the scores are combined.

A `Hit` carries:

| Field | Meaning |
|---|---|
| `doc_id` | the id you gave the document |
| `text` | the chunk's text |
| `score` | the fused score; higher is better |
| `start`, `end` | byte offsets of the chunk in the document's text |
| `chunk_id` | the engine's id for the chunk, stable across merges |
| `metadata` | the document's metadata |

## 4. Index files

`add_path` takes a file or a directory. A directory is walked in path order and hidden entries
are skipped. PDF, HTML, Markdown, and plain text files are read; other files are ignored.

```python
report = col.add_path("docs/", metadata={"source": "handbook"})
print(report["documents"], "documents,", report["chunks"], "chunks")
for path, reason in report["skipped"]:
    print("skipped", path, reason)
```

Each file becomes one document whose `doc_id` is its path. Long documents are split into
chunks of about 2 KB, cut at paragraph and sentence boundaries.

## 5. Filter on metadata

A filter maps keys to a value or a list of values. Every key must match; a list matches any of
its values.

```python
col.search("falcon", filter={"topic": "birds"})
col.search("falcon", filter={"topic": ["birds", "raptors"], "year": 2024})
```

Metadata values can be strings, numbers, or booleans. They are stored as text, so `2024` and
`"2024"` are the same value.

## 6. Update and delete

```python
col.update("owl", "Snowy owls hunt in daylight on the Arctic tundra.")
col.delete("peregrine")
```

`update` replaces every chunk of the document in one atomic step; a search never sees both the
old and the new version. Calling `add` again with an existing `doc_id` keeps the old chunks and
adds new ones beside them, so use `update` to replace.

## 7. Close

```python
col.close()
```

Or use the collection as a context manager, which closes it on exit:

```python
with Collection("./my-index") as col:
    hits = col.search("arctic owl")
```

Closing seals buffered writes into a segment, waits for background merges, and releases the
collection's lock. One process at a time can hold a collection; a second process that opens it
gets an error naming the holder's process id.

## The whole program

```python
from gyrfalcon import Collection

with Collection("./my-index") as col:
    col.add("falcon", "The gyrfalcon is the largest of the falcon species.",
            metadata={"topic": "birds"})
    col.add_many({
        "peregrine": "Peregrine falcons dive faster than any other animal.",
        "owl": "Owls hunt at night and fly almost silently.",
    })
    for hit in col.search("which bird is the fastest?", k=3):
        print(f"{hit.score:.3f}  {hit.doc_id:10}  {hit.text}")
```

## Next steps

- [Choosing a tier](../guides/choosing-a-tier.md): when the `full` tier is worth its ingest
  cost.
- [Indexing large corpora](../guides/indexing-large-corpora.md): batching, memory, and merges.
- [Migrating from LangChain](../guides/migrating-from-langchain.md) or
  [from LlamaIndex](../guides/migrating-from-llamaindex.md): use Gyrfalcon as the retriever of
  an existing pipeline.
- [Python API reference](../reference/python-api.md)

---

<sub>Next: [Rust quickstart](quickstart-rust.md)</sub>
