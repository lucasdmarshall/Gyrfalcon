<sub>[Docs](../README.md) / Getting started</sub>

# Rust quickstart

This page builds a small program that creates a collection, writes documents, searches with
and without a metadata filter, updates, and deletes. It uses the `lite` tier.

## 1. Add the crate

```sh
cargo new falcon-search && cd falcon-search
cargo add gyrfalcon --git https://github.com/lucasdmarshall/Gyrfalcon-
```

The engine's dependencies are slow when built without optimization. Add this to your
`Cargo.toml` so debug builds stay usable:

```toml
[profile.dev.package."*"]
opt-level = 3
```

## 2. Get the lite model

The Rust crate does not download models. From a clone of the repository:

```sh
scripts/fetch-models.sh /path/to/models
```

or, with the Python package installed, `python -c "import gyrfalcon; print(gyrfalcon.ensure_model('lite'))"`.
Either way you get a directory named `potion-retrieval-32M` holding `config.json`,
`model.safetensors`, and `tokenizer.json`. The program below expects it under `.models/`.

## 3. Write the program

`src/main.rs`:

```rust
use std::path::Path;

use gyrfalcon::{Document, Filter, RetrieveOptions, Tier};

fn main() -> gyrfalcon::Result<()> {
    // The lite model directory: config.json, model.safetensors, tokenizer.json.
    let models = Path::new(".models").join(Tier::Lite.model_dir());
    // Opens ./my-index, creating it on first use, with the lite tier's model and fusion.
    let mut index = Tier::Lite.open(Path::new("my-index"), &models, true)?;

    index.index_documents(&[
        Document::body("falcon", "The gyrfalcon is the largest of the falcon species.")
            .with_metadata([("topic", "birds")]),
        Document::body("peregrine", "Peregrine falcons dive faster than any other animal.")
            .with_metadata([("topic", "birds")]),
        Document::body("owl", "Owls hunt at night and fly almost silently."),
    ])?;

    let options = RetrieveOptions { rerank: false, limit: 3 };
    let hits = index.retrieve("which bird is the fastest?", options)?;
    for hit in &hits.hits {
        println!("{:.3}  {:10}  {}", hit.score, hit.doc_id, hit.body);
    }

    let birds = Filter::new().eq("topic", "birds");
    let filtered = index.retrieve_filtered("night hunter", options, Some(&birds))?;
    println!("{:?}", filtered.hits.iter().map(|h| h.doc_id.as_str()).collect::<Vec<_>>());

    index.update_document(Document::body("owl", "Snowy owls hunt in daylight on the Arctic tundra."))?;
    index.delete_document("peregrine")?;

    // Seal buffered writes and let background merges finish before the process exits.
    index.flush()?;
    index.wait_until_merge_idle();
    Ok(())
}
```

## 4. Run it

```sh
cargo run --release
```

```text
0.140  peregrine   Peregrine falcons dive faster than any other animal.
0.109  falcon      The gyrfalcon is the largest of the falcon species.
0.061  owl         Owls hunt at night and fly almost silently.
["falcon", "peregrine"]
```

The filtered query asks for a night hunter but only birds tagged `topic=birds` may answer, so
the owl is excluded even though it matches best.

## What the program did

| Call | Effect |
|---|---|
| `Tier::Lite.open(dir, models, true)` | Creates the collection if `dir` has no `manifest.json`, checks the collection's model, dimension, quantization, and analyzer against the tier, takes the writer lock, and attaches the lite embedder and fusion rule. |
| `index_documents` | Chunks each document, embeds the chunks, and writes them. Small writes go to a write-ahead log and an in-memory buffer; large ones seal a segment directly. Durable on return. |
| `retrieve` | Runs BM25 and dense search in parallel, fuses, and returns the top `limit` chunks. |
| `retrieve_filtered` | The same, restricted to chunks whose document matches the filter. |
| `update_document` | Writes the new chunks and retires the old ones in one atomic step. |
| `delete_document` | Retires every chunk of the document. |
| `flush` | Seals the write buffer into a segment. Not needed for durability. |
| `wait_until_merge_idle` | Blocks until the background merger has nothing left to do. |

`rerank: true` reorders the top 32 with a cross-encoder attached through `Index::set_reranker`.
It is off by default, and without a reranker attached it changes nothing.

## Errors

Every call returns `gyrfalcon::Result<T>`. The variants you are most likely to meet:

| Error | When |
|---|---|
| `IncompatibleIndex { field, .. }` | The collection was built with another model, dimension, quantization, analyzer, or format version. Nothing is opened. |
| `WriterBusy { pid }` | Another live process holds the collection. |
| `IndexNotFound` | Opening with `create = false` and no collection at the path. |
| `ModelUnavailable` | A model file or ONNX Runtime is missing. |
| `EmptyQuery` | The query has no text after trimming. |

## Next steps

- [Custom embedders](../guides/custom-embedders.md): implement `Embedder` for your own model.
- [Persistence](../guides/persistence.md): what is on disk and what survives a crash.
- [Rust API reference](../reference/rust-api.md)

---

<sub>Previous: [Python quickstart](quickstart-python.md) · Next: [Choosing a tier](../guides/choosing-a-tier.md)</sub>
