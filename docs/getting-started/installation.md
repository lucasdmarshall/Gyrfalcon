<sub>[Docs](../README.md) / Getting started</sub>

# Installation

Gyrfalcon ships as a Python package (`gyrfalcon`), a Rust crate (`gyrfalcon`), and a command
line (`gyr`). All three use the same engine and the same on-disk format, so a collection built
by one opens in the others.

| Component | Needed for | Required |
|---|---|---|
| The package, crate, or binary | everything | yes |
| Lite tier model (potion-retrieval-32M, 129 MB) | the default tier | yes, downloaded on first use from Python |
| Full tier model (int8 bge-small-en-v1.5) | the `full` tier | only for `full` |
| ONNX Runtime 1.20 or later | the `full` tier and the reranker | only for `full` |
| PDFium | reading PDF files | only for PDFs |

The lite tier needs no model runtime at all. Most installs need nothing beyond the first row.

## Python

Python 3.9 or later. Wheels are built for Linux (x86_64, aarch64), macOS (arm64, x86_64), and
Windows (x64), one abi3 wheel per platform.

```sh
pip install gyrfalcon              # lite tier, text, Markdown, and HTML
pip install "gyrfalcon[pdf]"       # adds PDF support through pypdfium2's PDFium
pip install "gyrfalcon[full]"      # adds the full tier through onnxruntime
pip install "gyrfalcon[full,pdf]"
```

> [!NOTE]
> The PyPI release is published by the `wheels` workflow when a `v*` tag is pushed. Until the
> first tag, install from source as shown below.

The extras do not add Python code paths. They install a library the engine loads at run time,
and the package finds it on its own: `pypdfium2` ships PDFium, `onnxruntime` ships ONNX
Runtime. If you already have either library, point the engine at it instead:

```sh
export GYR_PDFIUM=/path/to/libpdfium.so          # the library or its directory
export ORT_DYLIB_PATH=/path/to/libonnxruntime.so
```

### From source

Needs a Rust toolchain (stable) and `maturin`.

```sh
git clone https://github.com/lucasdmarshall/gyrfalcon
cd gyrfalcon/crates/gyrfalcon-py
python -m venv .venv && . .venv/bin/activate
pip install maturin
maturin develop --release
```

`maturin build --release` produces a wheel in `target/wheels/` instead of installing into the
active environment.

### Models

The package downloads a tier's model files from Hugging Face the first time a collection of
that tier opens. They go to `$GYR_MODELS`, or to `~/.cache/gyrfalcon/models` when it is unset
(`$XDG_CACHE_HOME/gyrfalcon/models` if that is set). Downloads go through `urllib`, so
`HTTPS_PROXY` and similar variables apply.

To prepare a machine without network access, fetch the files on a connected machine and copy
the directory:

```python
import gyrfalcon
gyrfalcon.ensure_model("lite", "/opt/gyrfalcon-models")   # returns the model directory
gyrfalcon.ensure_model("full", "/opt/gyrfalcon-models")
```

Then set `GYR_MODELS=/opt/gyrfalcon-models` on the offline machine, or pass
`models="/opt/gyrfalcon-models"` to `Collection`.

## Rust

The crate is not on crates.io yet. Depend on it from Git:

```sh
cargo add gyrfalcon --git https://github.com/lucasdmarshall/gyrfalcon
cargo add gyrfalcon-ingest --git https://github.com/lucasdmarshall/gyrfalcon   # optional: PDF and HTML
```

`gyrfalcon` is the engine. `gyrfalcon-ingest` turns PDF, HTML, and text files into text; add it
only if your program reads files.

Neither crate downloads or links a native library at build time. ONNX Runtime and PDFium are
loaded when first used:

- ONNX Runtime: `ORT_DYLIB_PATH`, else the system library path.
- PDFium: `GYR_PDFIUM` (the library or its directory), else the directory of the executable,
  else the system library path. Prebuilt PDFium libraries are published at
  [bblanchon/pdfium-binaries](https://github.com/bblanchon/pdfium-binaries).

The Rust crate does not download models. Fetch them with `scripts/fetch-models.sh` (below) or
with the Python `ensure_model`, and pass the model directory to `Tier::open`.

## The `gyr` command line

```sh
cargo install --git https://github.com/lucasdmarshall/gyrfalcon gyr-cli
```

This installs `gyr`. From a clone, `cargo build --release` builds it at `target/release/gyr`.

`gyr` looks for models under `--models` (or `GYR_MODELS`), which defaults to `.models` in the
current directory. `scripts/fetch-models.sh` fills that directory with both tiers' models, the
cross-encoder reranker, and ONNX Runtime, then prints the environment to export:

```sh
scripts/fetch-models.sh            # into .models
eval "$(scripts/fetch-models.sh)"  # and export ORT_DYLIB_PATH, GYR_MODELS, ...
```

Set `GYR_MODELS` to the same directory for Python and the CLI and they share one copy.

## Check the install

```sh
python -c "import gyrfalcon; print(gyrfalcon.__version__, gyrfalcon.supported_extensions())"
gyr --version
```

## Platform notes

- Linux x86_64 is where the test suite and every benchmark in these docs run.
- The dense kernels are compiled for AVX2 with popcnt, popcnt alone, and a portable baseline;
  the engine picks one at run time, so one binary runs on any x86_64 CPU. Other architectures
  use the portable kernels.
- On Windows, a collection's files stay mapped while a reader holds them. Deleting a retired
  segment can fail with a sharing violation; the engine keeps it queued and retries, so this
  needs no action.

---

<sub>Next: [Python quickstart](quickstart-python.md) or [Rust quickstart](quickstart-rust.md)</sub>
