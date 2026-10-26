# Corpus distillation (v2 beta, parked)

Status: **design only.** Nothing below is implemented or measured. Numbers marked *estimate*
are arithmetic from v1 measurements, not results. Parked on 2026-10-26: the v1 marketing promise
is met, and this feature changes what a collection is and could fit the benchmarks instead of
the user's data, so it waits for its own staged evaluation.

## Why

The lite tier embeds with a static table (potion-retrieval-32M, 63,091 tokens × 512 dims): a
query or chunk vector is the mean of its tokens' rows. It needs no model runtime and embeds at
memory speed, but its dense leg is the lite tier's limit. On the v1 protocol the lite dense leg
alone scores well below the full tier's (SciFact 0.632 vs 0.715, ArguAna 0.452 vs 0.592,
NFCorpus 0.308 vs 0.339), and the 2026-10-26 sweep found that fusion and BM25 changes no longer
move held-out quality by more than about 0.01 (`bench/results/2026-10-26-quality-sweep`).

The full tier's bge-small-en-v1.5 is the better embedder but costs a transformer pass per text:
SciFact's 5,183 documents take 193 s on 4 cores, against under 1 s for the lite tier.

Corpus distillation tries to have both: spend teacher (bge) compute **once, on a sample of the
user's own corpus**, to fit a static table whose mean-pooled vectors approximate bge's on that
corpus. Queries and every later write then embed statically at lite speed.

## What it is

```
sample of the collection's chunks ──► bge-small (ORT, int8)  ──► teacher vectors T (S × 384)
                     │
                     └──► token counts ──► pooling matrix A (S × V, sparse, SIF-weighted)

initial table E0 (V × 384)  ──►  ridge fit:  min ||A E − T||² + λ ||E − E0||²  ──► table E
                                                                                     │
re-embed every chunk with E (static, seconds) ──► new vectors.bin / codes.bin per segment
                                                                                     │
publish a manifest whose model id names the distilled table ──► queries embed with E
```

### 1. Vocabulary and tokenizer

- Base: bge-small's WordPiece vocabulary (30,522 tokens), so the teacher and the table share a
  tokenizer and the WordPiece fast path (`wordpiece.rs`) applies unchanged.
- Optional extension (stage D0 decides): frequent corpus words that WordPiece splits into
  several pieces (for example a domain term) become whole tokens, as Model2Vec's vocabulary
  extension does. Their initial rows come from the teacher embedding the word alone. The
  tokenizer JSON gains these entries; longest-match WordPiece then prefers them.

### 2. Initial table E0

Two candidates, compared in stage D0:

- **bge token-level distillation** (Model2Vec's base method): run every vocabulary token through
  the teacher as `[CLS] token [SEP]` and take its output embedding; 30,522 sequences of 3 tokens,
  about 92k tokens, comparable to embedding 300 documents (*estimate*: about 10 s on 4 cores).
- **potion mapped into bge space:** potion's rows (512 dims, trained for retrieval on a large
  general corpus) times a linear map W (512 × 384) fitted by least squares on the sample's
  pooled vectors against T. Keeps potion's training; needs potion's vocabulary instead of bge's,
  so tokens map through potion's tokenizer.

### 3. Weights

Mean pooling over raw counts lets frequent tokens dominate. Each token gets a SIF weight
`w_t = a / (a + p_t)`, with `p_t` its probability in the collection (known exactly from the
analyzer pass; `a` about 1e-3), so the pooling matrix row for chunk i is
`A_i,t = w_t · count_i,t / Σ_t' w_t' · count_i,t'`. Query pooling uses the same weights, so
query and chunk vectors stay in one space. The weights are stored with the table.

### 4. Fit

Ridge regression toward the teacher, anchored to E0:

```
E = argmin ||A E − T||_F² + λ ||E − E0||_F²      ⇔      (Aᵀ A + λ I) E = Aᵀ T + λ E0
```

- T: the teacher's L2-normalized document embeddings of the sample (no query instruction).
- Solved with conjugate gradients on all 384 columns at once; each iteration is two sparse
  products with A (nnz about S × 150 distinct tokens per chunk). *Estimate*: S = 10,000 gives
  1.5M nonzeros, about 1.2 GFLOP per iteration, seconds for 50 iterations, no autograd and no
  Python needed.
- Tokens absent from the sample get no gradient and keep their E0 row: the anchor is what keeps a
  small corpus from producing garbage rows for rare tokens.
- Optional PCA to 256 dims after the fit (stage D0 decides): smaller vectors, faster dense leg.
- Least squares on normalized targets approximates the cosine objective Model2Vec's tokenlearn
  trains with; if D0 shows a gap, a few epochs of cosine-loss SGD on E (same sparse products)
  replace it.

### 5. Teacher cost

The expensive step. From the full-tier measurement (27 documents of about 1.5 KB per second on
4 cores), *estimates* for a sample of 2 KB chunks:

| sample S | 4 cores | 1 core |
|---|---|---|
| 2,000 | about 1.5 min | about 6 min |
| 10,000 | about 6 min | about 25 min |
| 30,000 | about 19 min | about 75 min |

Distillation is therefore opt-in, runs in the background with progress, checkpoints teacher
vectors so an interrupted run resumes, and never runs implicitly during `index`.

## How a collection carries it

- **Identity.** The manifest's model id becomes `gyr-distill/bge-small-en-v1.5-q8/<sha256 of the
  table>` with the table's dimension; the open gate (`check_compatible`) then refuses to mix
  vectors from different tables, as it does for any model change.
- **Storage.** The table, its tokenizer, and its SIF weights live inside the collection:
  `models/<generation>/model.safetensors`, `tokenizer.json`, `weights.bin`. A collection made on
  one machine opens on another without downloading anything. The table is memory-mapped like
  potion's (`static_embed.rs`, `Table::Mapped`).
- **Re-vectoring.** After the fit every live chunk is re-embedded with E (static, seconds even
  for 500k chunks) and every segment is rewritten with new `vectors.bin` and `codes.bin`; the
  lexical files and docs are unchanged and can be hard-linked. The write buffer is flushed first.
  One manifest publish switches table and segments together, so a reader never sees a mix. Old
  segment directories go through the reaper as after a merge.
- **Concurrency.** Distillation holds the writer lock for the re-vector and publish steps; the
  teacher pass (the long part) runs on a snapshot, and chunks written meanwhile are re-embedded
  in the final step.
- **Growth.** The manifest records the sample size and the chunk count at distillation. New
  writes embed with the distilled table. When the collection has grown past a factor (2× by
  default) a warning suggests re-distilling; nothing re-distills on its own.
- **Revert.** `gyr distill --revert` (and the Python equivalent) re-embeds with potion and
  publishes the lite tier's identity again.

## Interfaces (proposed)

```sh
gyr distill ./my-index --sample 10000          # needs the full tier's model files and ORT
gyr distill ./my-index --revert
```

```python
col.distill(sample=10_000)        # blocks; progress on stderr
col.distill(revert=True)
```

The full tier's files (`bge-small-en-v1.5-q8`, ONNX Runtime) are needed only while distilling.

## Risks

1. **It may not beat potion.** potion-retrieval-32M was trained on a large corpus for retrieval;
   a table fitted to one corpus's documents may embed that corpus better and its queries worse.
   This is the main risk and the reason for stage D0 before any engine work.
2. **Overfitting to the corpus,** most for small collections, where many tokens appear a few
   times. Mitigations: the anchor λ, a minimum collection size (refuse below about 2,000
   chunks), and the within-corpus check in D0.
3. **Query and document asymmetry.** The fit sees documents only; short queries pool few tokens.
   bge itself embeds queries with an instruction prefix, which the static table cannot imitate.
4. **Benchmark contamination.** Distilling on a BEIR corpus uses that corpus's documents (never
   its queries or judgments), which is what a user's collection would do. Every reported number
   must still say the table was distilled on the evaluated corpus, and D0 must also report
   cross-corpus results (distil on one set, evaluate on another).
5. **Cost on small machines.** About 25 minutes for 10,000 chunks on one core (*estimate*). It
   must stay opt-in and resumable.
6. **Collection format.** New manifest fields and an in-collection model directory: a format
   version bump, and v1 readers must refuse such collections cleanly.

## Staged plan, gates, and kill criteria

Every stage reports on the v2 protocol (`docs/v2-beta/README.md`): tuning sets' test splits and
the four held-out BEIR sets, paired bootstrap per dataset, speed and memory beside quality.

**D0, offline prototype (Python, no engine change).** model2vec distillation plus NumPy/SciPy
ridge on each BEIR corpus; vectors scored with the same fusion (`Fusion::RescoreMax`) through
the trace tools. Matrix: E0 ∈ {bge token distill, potion mapped}; SIF on/off; λ ∈ three values;
S ∈ {2k, 5k, 10k, 30k}; dims ∈ {384, 256}; vocabulary extension on/off. Choose on tuning
validation splits only.
Checks: reconstruction cosine on held-back chunks of the same corpus (fit on 80%, test on 20%);
a cross-corpus matrix (distil on A, evaluate on B).
- Gate to D1: held-out mean nDCG@10 at least +0.010 over v1 lite (0.4993), no dataset below its
  v1 value by more than 0.010, tuning test mean not lower, and the chosen S fitting the cost
  table above.
- Kill: the best variant gains less than +0.005 on the held-out mean, or wins only where it was
  distilled on the evaluated corpus and loses cross-corpus.

**D1, engine.** Rust: SIF pooling in `StaticEmbedder`, the CG ridge fit, `gyr distill`,
segment re-vectoring, manifest identity, revert. Gate: reproduces D0's numbers within ±0.003;
query p50 and p99 within v1 lite's; re-vectoring 500k chunks under a minute on 4 cores; crash
tests for the publish step.

**D2, release as beta.** Python API, docs, an explicit beta flag; a v1 lite collection never
changes unless `distill` is called.

## Prior art

- Model2Vec (MinishLab): static tables distilled from sentence transformers by token-level
  embedding, PCA, and Zipf/SIF weighting; `tokenlearn` fits tables to teacher embeddings of a
  corpus. potion models are built this way.
- SIF (Arora et al., 2017): `a / (a + p(w))` weighting for averaged word vectors.
- sentence-transformers `StaticEmbedding` (static-retrieval-mrl-en-v1): static tables trained
  contrastively for retrieval; measured against potion in the spec ("Lite model candidates").
