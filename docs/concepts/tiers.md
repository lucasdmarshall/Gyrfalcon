<sub>[Docs](../README.md) / Concepts</sub>

# Tiers

A tier is a named bundle of the choices that must stay fixed for the life of a collection: which
model embeds the text, where its files come from, how wide its vectors are, and how the two
retrieval legs are fused. Gyrfalcon has two, `lite` and `full`. This page explains what each
contains and how the engine keeps a collection from being opened with the wrong one. For which
one to pick, see [Choosing a tier](../guides/choosing-a-tier.md).

## What a tier fixes

| | `lite` | `full` |
|---|---|---|
| Model id recorded in the collection | `minishlab/potion-retrieval-32M` | `Xenova/bge-small-en-v1.5-q8` |
| Model files | `config.json`, `model.safetensors`, `tokenizer.json` | `model.onnx`, `tokenizer.json` |
| Model directory name | `potion-retrieval-32M` | `bge-small-en-v1.5-q8` |
| Vector width | 512 | 384 |
| Embedder | static table, mean of token rows | int8 transformer, first-token output |
| Runtime | none | ONNX Runtime, loaded at run time |
| Fusion | `rescore-max`, dense weight 0.45 | `rescore`, dense weight 0.65 |
| Stored vectors | int8 | int8 |
| Analyzer | `gyr-text-v2` | `gyr-text-v2` |

The tier definitions live in one place, `gyrfalcon::Tier`. The Python package, the command line,
and Rust programs all open collections through it, so a collection built by any of them opens
in the others.

## The lite tier: a static embedding table

[potion-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M) is a Model2Vec
model: a table with one 512-dimensional row for each of 63,091 tokens, distilled from a sentence
transformer and trained further for retrieval. Embedding a text means:

1. Tokenize with the model's WordPiece vocabulary (Gyrfalcon has its own fast path for this,
   checked token for token against the reference tokenizer on 6,731 texts).
2. Look up each token's row.
3. Average the rows and L2-normalize the result.

There is no neural network to run. A text costs a tokenizer pass and a few hundred additions per
token, and each text's vector is independent of what else is in its batch, so results are
identical under any batching or thread count.

The 129 MB table is memory-mapped rather than read, so a process pays only for the rows of the
tokens it meets. A one-shot `gyr query` on a lite collection takes 33 ms and 27 MB of memory.

What it gives up: word order and context. "Dog bites man" and "man bites dog" embed the same,
and a word's vector does not change with its neighbors. That is why the lite tier leans on BM25
for precision and uses the dense leg mainly to recover vocabulary mismatches.

## The full tier: a small transformer

[bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) is a 33M-parameter BERT
encoder. Gyrfalcon runs the int8-quantized ONNX export through ONNX Runtime and takes the first
token's output (CLS pooling), L2-normalized, as the BGE models expect. Inputs are cut at 512
tokens.

Its vectors depend on context, which makes its dense leg much stronger than the lite tier's
(SciFact dense-only nDCG@10 0.711 against 0.632). The cost is a transformer pass per text: about
25 short documents per second on 4 cores during ingest, and a few milliseconds per query.

Two details keep that cost bounded:

- **Length-sorted batches capped at 8 KB.** A transformer pads each batch to its longest text.
  Sorting by length wastes no padding, and the cap bounds memory. On SciFact this took ingest
  from 482 s and 2.1 GB to 195 s and 0.5 GB.
- **Thread count.** ONNX Runtime gets `min(4, cores)` intra-op threads unless `GYR_ORT_THREADS`
  says otherwise.

One caveat: the int8 model quantizes activations per batch, so the same text can get a slightly
different vector in a different batch. Across batching configurations SciFact's nDCG@10 moved
between 0.742 and 0.748. The lite tier has no such effect.

## The identity check

A collection's `manifest.json` records the identity it was created with:

```json
{
  "format_version": 7,
  "model_id": "minishlab/potion-retrieval-32M",
  "dimension": 512,
  "quantization": "i8",
  "analyzer_id": "gyr-text-v2"
}
```

Every open compares these five fields with what the caller asks for, before mapping any segment
file and before loading any model. A mismatch is refused with the field and both values:

```text
incompatible index: model_id is minishlab/potion-retrieval-32M in the manifest, Xenova/bge-small-en-v1.5-q8 requested
```

Each field guards a different way to get silently wrong results:

| Field | Mixing it would mean |
|---|---|
| `model_id` | comparing query vectors from one model with chunk vectors from another |
| `dimension` | the same, or reading vectors at the wrong stride |
| `quantization` | reading stored vectors in the wrong number format |
| `analyzer_id` | looking up query terms that were never indexed in that form (`safety` against the stem `safeti`) |
| `format_version` | reading files laid out differently |

The model's own output width is checked against `dimension` as well, so pointing a tier at the
wrong model files fails at open rather than at the first query.

## Beyond the two tiers

The tiers are presets, not limits. The command line and the Rust API accept any Model2Vec
model, any CLS-pooled ONNX encoder, an OpenAI-compatible embeddings endpoint, or your own
`Embedder` implementation; the identity check applies to them in the same way. See
[Custom embedders](../guides/custom-embedders.md). Why the product exposes two named tiers rather
than free configuration is recorded in [design record 0002](../design/0002-tier-system.md).

---

<sub>Previous: [Hybrid retrieval](hybrid-retrieval.md) · Next: [Planner](planner.md)</sub>
