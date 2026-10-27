<sub>[Docs](../README.md) / Contributing</sub>

# Development setup

Everything needed to build, test, and benchmark Gyrfalcon from a clone.

## Requirements

| Tool | Version | For |
|---|---|---|
| Rust | stable (developed on 1.97) | everything |
| Python | 3.9 or later (developed on 3.13) | the Python package, benchmark harnesses |
| `maturin` | 1.7 or later | building the Python package |
| `curl`, `pip` | any | `scripts/fetch-models.sh` |
| `taskset` | util-linux | pinned benchmarks (Linux) |

No C or C++ toolchain is needed: ONNX Runtime and PDFium are loaded at run time, not linked.

## Clone and build

```sh
git clone https://github.com/lucasdmarshall/gyrfalcon
cd gyrfalcon
cargo build --release        # target/release/gyr
```

The workspace has four crates:

```text
crates/
├── gyrfalcon/          the engine (library)
├── gyrfalcon-ingest/   files to text
├── gyr-cli/            the gyr binary, HTTP server, and benchmark harness
└── gyrfalcon-py/       the Python package (built with maturin, not part of `cargo build`)
```

Dependencies are compiled with optimizations even in debug builds (`[profile.dev.package."*"]`
in `Cargo.toml`), because tests build segments of 100,000 chunks.

## Models and runtime

```sh
scripts/fetch-models.sh               # into .models/
eval "$(scripts/fetch-models.sh)"     # also export ORT_DYLIB_PATH, GYR_MODELS, ...
```

This downloads both tiers' models, the fp32 bge-small model, the MiniLM cross-encoder, and an
ONNX Runtime build from PyPI. `.models/` is ignored by Git.

For PDFs, download a PDFium build for your platform from
[bblanchon/pdfium-binaries](https://github.com/bblanchon/pdfium-binaries) and set `GYR_PDFIUM`
to the library, or `pip install pypdfium2` and point `GYR_PDFIUM` at the `libpdfium` inside it.

## The Python package

```sh
cd crates/gyrfalcon-py
python -m venv .venv && . .venv/bin/activate
pip install maturin pytest
maturin develop --release
pytest -q tests
```

`maturin develop` builds the native module and installs the package into the active
environment in editable mode: Python changes apply at once, Rust changes need another
`maturin develop`.

## Benchmark data

```sh
scripts/fetch-beir.sh                                        # scifact nfcorpus fiqa arguana into .data/beir
scripts/fetch-beir.sh .data/beir trec-covid scidocs quora webis-touche2020
```

`.data/` is ignored by Git. The head-to-head harness needs its own virtual environment; see
[Reproducing](../benchmarks/reproducing.md).

## Useful commands

| Task | Command |
|---|---|
| Format | `cargo fmt --all` |
| Lint | `cargo clippy --workspace --all-targets -- -D warnings` |
| All Rust tests | `cargo test --release` |
| One test file | `cargo test --release -p gyrfalcon --test w1_buffer` |
| Engine docs | `cargo doc -p gyrfalcon --no-deps --open` |
| Profile ingest | `cargo run --release -p gyrfalcon --example profile_ingest -- .data/beir/fiqa/corpus.jsonl .models/potion-retrieval-32M` |
| Profile the BM25 leg | `cargo run --release -p gyrfalcon --example profile_query -- .data/beir/scifact` |
| Profile the dense leg | `cargo run --release -p gyrfalcon --example profile_dense -- .data/beir/fiqa .models/potion-retrieval-32M` |
| Profile small writes | `cargo run --release -p gyrfalcon --example profile_incremental -- .data/beir/fiqa/corpus.jsonl .models/potion-retrieval-32M 100 1000` |
| Quick BEIR check | `target/release/gyr bench beir --dataset .data/beir/scifact` |

## Cross-checking other platforms

CI builds and tests on Linux. Release wheels are built for macOS and Windows by the `wheels`
workflow. To type-check another target locally:

```sh
rustup target add aarch64-unknown-linux-gnu x86_64-pc-windows-msvc
cargo check --target x86_64-pc-windows-msvc -p gyrfalcon -p gyrfalcon-ingest -p gyr-cli
```

Some dependencies compile C code for their target, so a full cross-check may need a C
cross-compiler (for example `zig cc` as `CC_<target>`).

## Editor setup

Any editor with rust-analyzer works. The repository's `rustfmt.toml` sets 120 columns and
`use_small_heuristics = "Max"`; format on save keeps diffs small.

---

<sub>Previous: [Contributing](CONTRIBUTING.md) · Next: [Testing](testing.md)</sub>
