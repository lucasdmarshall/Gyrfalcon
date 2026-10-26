<sub>[Docs](../README.md) / Guides</sub>

# Custom embedders

The two tiers cover most needs. When they do not, for example when you need a multilingual model,
a domain model, or a hosted embedding API, Gyrfalcon can embed with something else. There are
four ways, from least to most code:

| Way | Interface | Code | Model runs |
|---|---|---|---|
| Another Model2Vec static model | `gyr --embedder static` | none | in process, no runtime |
| Another ONNX encoder | `gyr --embedder ort` | none | in process, ONNX Runtime |
| An OpenAI-compatible embeddings endpoint | `gyr --embedder http` | none | wherever the endpoint runs |
| Anything else | the Rust `Embedder` trait | one `impl` | your choice |

The Python package opens collections through the two tiers only. To use a custom embedder from
Python, build and serve the collection with `gyr` and call its [HTTP API](../reference/configuration.md#http-api),
or write the embedder in Rust.

## What every embedder must respect

- **A collection belongs to one model.** The manifest records the model id and the vector width
  at creation. Opening with a different id or width fails with `IncompatibleIndex` before
  anything is loaded. Give each model, and each version of a model, its own id.
- **The same function embeds queries and chunks.** The embedder receives text only, with no
  flag saying whether it is a query or a passage. Models that need a different prefix for each
  role (for example `query:` and `passage:`) are not supported as such; models trained to work
  without one, such as the BGE family, are.
- **Text arrives as written.** The embedder gets the chunk text after Unicode NFC normalization
  and trimming, with its original case.
- **Vectors are compared by cosine.** With the default `i8` storage the engine L2-normalizes
  each vector and scales it to signed bytes, so vector length carries no meaning.
- **Pick a fusion rule.** The fusion weights were tuned for the two tier models. For another
  transformer, start from the full tier's `rescore` at 0.65; for another static model, from the
  lite tier's `rescore-max` at 0.45. Then check on your own queries.

## A Model2Vec static model

Any [Model2Vec](https://github.com/MinishLab/model2vec) model directory with
`model.safetensors` and `tokenizer.json` (and optionally `config.json`) works:

```sh
gyr index ./idx docs/ \
  --embedder static \
  --model-dir models/potion-base-32M \
  --model-id minishlab/potion-base-32M \
  --fusion rescore-max --dense-alpha 0.45
```

Every later `gyr` command on that collection must pass the same `--embedder`, `--model-dir`,
and `--model-id`. Setting them once as `GYR_EMBEDDER`, `GYR_MODEL_DIR`, and `GYR_MODEL_ID`
avoids repeating them.

For a Matryoshka model, `--dimension` below the table's width keeps the first that many
components of every vector, which shrinks the index at some cost in quality.

## An ONNX encoder

A directory with `model.onnx` (or `onnx/model.onnx`) and `tokenizer.json`:

```sh
gyr index ./idx docs/ \
  --embedder ort \
  --model-dir models/my-encoder \
  --model-id my-org/my-encoder-v1 \
  --dimension 768 \
  --fusion rescore --dense-alpha 0.65
```

The ONNX path takes the first token's output vector (CLS pooling) and L2-normalizes it, as BGE
models expect. Inputs are cut at 512 tokens. A model that expects mean pooling will load and
run but rank worse; wrap it with the Rust trait instead. `--dimension` must equal the model's
output width (the default is 384).

ONNX Runtime must be loadable: set `ORT_DYLIB_PATH` or install it on the library path.

## An embeddings endpoint

`--embedder http` sends batches as an OpenAI embeddings request, `{"model": ..., "input": [...]}`,
and reads `data[i].embedding` from the response:

```sh
gyr serve ./idx \
  --embedder http \
  --embed-url http://localhost:8080/v1/embeddings \
  --model-id BAAI/bge-base-en-v1.5 \
  --dimension 768
```

The model id is sent as the request's `model`. Every returned vector must have `--dimension`
components or the call fails. The command line sends no API key; for an endpoint that needs one,
put an authenticating proxy in front of it, or use `HttpEmbedder::with_api_key` from Rust.

Every chunk and every query becomes a network call, so ingest and query latency follow the
endpoint's.

## The Rust `Embedder` trait

```rust
pub trait Embedder: Send + Sync {
    /// One vector of `dimension()` components per text.
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn model_id(&self) -> &str;
    fn dimension(&self) -> u32;
    /// Bytes of text per ingest call, every text counted at the batch's longest.
    fn batch_bytes(&self) -> usize { 32 * 1024 }
}
```

`embed` is called from several ingest threads at once; it must be thread-safe. During ingest,
texts arrive sorted by length in batches of at most 32 texts and `batch_bytes` bytes. A padded
transformer computes a batch at its longest text, so return a small `batch_bytes` (the ONNX
embedder uses 8 KB) to bound memory; a model without padding can keep the default.

A complete program with a toy embedder in place of a real model:

```rust
use std::path::Path;

use gyrfalcon::{
    Document, Embedder, Fusion, Index, IndexConfig, Quantization, RetrieveOptions, ANALYZER_ID,
};

/// Letter-trigram hashing: a stand-in for your model. Replace the body of `embed` with a call
/// into your runtime or service.
struct TrigramEmbedder {
    dimension: u32,
}

impl Embedder for TrigramEmbedder {
    fn embed(&self, texts: &[String]) -> gyrfalcon::Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|text| {
                let width = self.dimension as usize;
                let mut vector = vec![0f32; width];
                let chars: Vec<char> = text.to_lowercase().chars().collect();
                for window in chars.windows(3) {
                    let hash = window.iter().fold(0u32, |h, c| h.wrapping_mul(31).wrapping_add(*c as u32));
                    vector[hash as usize % width] += 1.0;
                }
                vector
            })
            .collect())
    }

    fn model_id(&self) -> &str {
        "example/trigram-v1"
    }

    fn dimension(&self) -> u32 {
        self.dimension
    }
}

fn main() -> gyrfalcon::Result<()> {
    let embedder = TrigramEmbedder { dimension: 256 };
    let config = IndexConfig {
        model_id: embedder.model_id().to_string(),
        dimension: embedder.dimension(),
        quantization: Quantization::I8,
        analyzer_id: ANALYZER_ID.to_string(),
    };
    let dir = Path::new("trigram-index");
    let mut index = if dir.join("manifest.json").exists() {
        Index::open(dir, &config)?
    } else {
        Index::create(dir, &config)?
    };
    index.set_embedder(Box::new(embedder));
    index.fusion = Fusion::Rescore { dense_weight: 0.4 };

    index.index_documents(&[Document::body("a", "Gyrfalcons nest on Arctic cliffs.")])?;
    let hits = index.retrieve("arctic cliff nesting", RetrieveOptions { rerank: false, limit: 5 })?;
    println!("{:?}", hits.hits.iter().map(|h| (&h.doc_id, h.score)).collect::<Vec<_>>());
    Ok(())
}
```

`Index::open` checks the manifest against `config` before anything else, so running this
program against a collection built with another model fails cleanly instead of mixing vectors.
Set the embedder before the first write: an index opened without one embeds with a model-free
hashing stand-in meant for tests.

The library also exports the embedders the tiers and the CLI use, so they can be composed
directly: `StaticEmbedder::from_dir`, `OrtEmbedder::from_dir`, `HttpEmbedder::new`, and
`HashEmbedder` for tests.

---

<sub>Previous: [Indexing large corpora](indexing-large-corpora.md) · Next: [Persistence](persistence.md)</sub>
