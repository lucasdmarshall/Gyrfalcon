<sub>[Docs](../README.md) / Benchmarks</sub>

# Reproducing the benchmarks

Every number in [Results](results.md) can be regenerated from this repository. This page gives
the one-command reruns first, then the individual commands behind each table.

## One command

```sh
scripts/reproduce.sh lite          # Gyrfalcon lite tier, eight BEIR sets, five runs
scripts/reproduce.sh compare       # head-to-head against LlamaIndex, LangChain, Haystack, 4 cores
```

Each builds `gyr`, downloads whatever models and datasets are missing, runs, and writes every
raw report plus a `summary.md` to `bench/reports/reproduce-<mode>-<timestamp>/`.

| Mode | Produces | Time on a 4-vCPU VM | Downloads |
|---|---|---|---|
| `lite [RUNS]` | the [variance table](results.md#run-to-run-variance): nDCG@10, index time, p50 and p99 per dataset, median and range over `RUNS` runs | about 2.5 minutes per run after the first | models (about 400 MB with ONNX Runtime), eight BEIR sets (about 1.3 GB unpacked) |
| `compare [CORES]` | the [head-to-head tables](results.md#head-to-head) for the lite embedder on SciFact, NFCorpus, ArguAna, and FiQA | several hours; the Python frameworks' FiQA cells dominate | the above, plus a Python environment from `bench/compare/requirements.txt` |

Requirements: a Rust toolchain, `python3`, `pip`, and `curl`; `compare` also needs `taskset`
(Linux).

## What to expect

- **nDCG@10 for the lite tier is deterministic.** The static embedder computes each text alone
  and every stage is order-independent, so a rerun gives the same value to every digit on any
  machine.
- **nDCG@10 for the full tier moves slightly.** The int8 ONNX model quantizes activations per
  batch, so batch composition, and with it thread scheduling, changes vectors a little. Expect
  about ±0.003.
- **Times depend on the machine.** Compare ratios between systems measured on the same machine,
  not absolute milliseconds across machines. The published numbers name their CPU in each
  report's `environment` block.

## The commands behind each result

### Gyrfalcon on BEIR

```sh
scripts/fetch-models.sh
scripts/fetch-beir.sh .data/beir scifact nfcorpus arguana fiqa trec-covid scidocs quora webis-touche2020
cargo build --release

target/release/gyr bench beir \
  --dataset .data/beir/scifact .data/beir/nfcorpus .data/beir/arguana .data/beir/fiqa \
            .data/beir/trec-covid .data/beir/scidocs .data/beir/quora .data/beir/webis-touche2020 \
  --fusions configured,lexical,dense \
  --out lite.json
```

`--fusions` scores several rules on one index: `configured` (the tier's), `lexical`, `dense`,
`rrf`, `score:W`, `rescore:W`, and `rescoremax:W`. Add `--tier full` for the full tier (needs
`ORT_DYLIB_PATH`), and `--run-dir DIR` to also write each ranked list as a TSV for another
evaluator.

### The head-to-head

```sh
python3 -m venv .venv-compare
.venv-compare/bin/pip install -r bench/compare/requirements.txt
.venv-compare/bin/python bench/compare/compare.py out/ --cores 4 --datasets scifact,nfcorpus,arguana,fiqa
.venv-compare/bin/python bench/compare/compare.py out/ --cores 4 --datasets scifact,nfcorpus --embedders bge-small
.venv-compare/bin/python bench/compare/compare.py out/ --cores 1 --datasets scifact,nfcorpus
.venv-compare/bin/python bench/compare/summarize.py out/results.jsonl
```

`compare.py` skips cells already in `out/results.jsonl`, so an interrupted run resumes.
`--systems gyrfalcon` reruns only Gyrfalcon's cells. The harness and the framework adapters
are in [`bench/compare/`](../../bench/compare/).

### Fusion weights and the planner

```sh
target/release/gyr bench trace --dataset .data/beir/scifact .data/beir/nfcorpus \
  .data/beir/arguana .data/beir/fiqa --out-dir bench/traces
target/release/gyr bench replay --traces bench/traces/minishlab_potion-retrieval-32M/*.jsonl \
  --split validation --policies lexical,dense,rrf,rescore:0.3,rescore:0.4,rescore:0.5 \
  --reference rescore:0.4
```

Run `replay` with `--split validation` to choose and with `--split test` once to confirm.
Results are in `bench/results/2026-10-26-full-tier-v2/` and `2026-10-25-cloud-4core-h2h/`
(`replay-*.json`).

### Low-CPU runs

```sh
scripts/bench-low-cpu.sh            # lite benches pinned to 1, 2, and 4 cores
```

### Other benches

| Command | Measures |
|---|---|
| `gyr bench retrieve --segments 1` and `--segments 50` | single-flight and concurrent query latency on synthetic text |
| `gyr bench ingest` | documents per second with merges included |
| `gyr bench ram --vectors 1000000` | file bytes and RSS of a million-vector collection |
| `gyr bench embed` | embedder throughput and single-query latency |
| `gyr bench accuracy` | the Open RAG Bench text-only slice |
| `bench/pdf/` | PDF text extractors on arXiv papers ([results](../../bench/results/2026-10-26-pdf-extractors/)) |

## Published results

Each directory under [`bench/results/`](../../bench/results/) holds the raw JSON and a README
naming the commit, the command, and the machine.

| Directory | Contents |
|---|---|
| `2026-10-25-cloud-4core-h2h` | Lite embedder head-to-head, 4 cores, with replay files for the lite weight |
| `2026-10-25-cloud-1core-h2h` | The same on 1 core |
| `2026-10-26-lite-refresh` | Gyrfalcon's lite cells after the memory and batching changes |
| `2026-10-26-cloud-4core-bge-h2h` | bge-small head-to-head, 4 cores |
| `2026-10-26-full-tier-v2` | Full tier fusion weight, validation and test |
| `2026-10-26-quality-sweep` | Lite fusion rule and query term counts, tuning and held-out sets |
| `2026-10-26-lite-variance` | Five repeated lite runs over eight BEIR sets |
| `2026-10-26-pdf-extractors` | Ten PDF extractors on 100 arXiv papers |
| `baseline` | The first reference numbers, before the validation and test split existed |

---

<sub>Previous: [Methodology](methodology.md)</sub>
