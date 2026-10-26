<sub>[Docs](../README.md) / Contributing</sub>

# Contributing to Gyrfalcon

Thank you for considering a contribution. This page explains what kinds of changes are welcome,
how to propose them, and what a change needs before it can be merged. Setting up a machine is
covered in [Development setup](development-setup.md), and running the tests in
[Testing](testing.md).

## What is welcome

- Bug reports with a reproduction: the commands or code, what you expected, what happened, and
  the output of `gyr --version` or `gyrfalcon.__version__`.
- Fixes, with a test that fails before the fix.
- Documentation corrections. If a page says something the code does not do, the page is wrong.
- Performance work, with measurements (see below).
- Quality work, through the evaluation protocol (see below).
- New file readers for `gyrfalcon-ingest`, new embedders behind the `Embedder` trait, and
  bindings for other languages.

For anything larger than a focused fix, open an issue first and describe the problem and the
approach. Changes to the on-disk format, the public API, or a default setting need that
discussion before code.

## How a change is judged

Gyrfalcon makes measurable claims, so changes are judged by measurements, not by argument.

### Correctness

- Every behavior change comes with a test. Integration tests live in
  `crates/gyrfalcon/tests/`, named by area (`m1_seal`, `w1_buffer`, `f1_filter`, ...).
- Crash safety is tested by injecting failures, not assumed. If a change touches publishing,
  merging, the write-ahead log, or deletion, extend the tests that stop before a rename, pause a
  merge, or fail a delete.
- BM25 scores must stay identical to Tantivy's; the differential test in
  `crates/gyrfalcon/src/lexical/reference.rs` enforces it.

### Speed and memory

A change that claims to be faster or smaller includes before and after numbers in the pull
request: the command, the machine (CPU, cores), and the best of three runs. The profiling
examples in `crates/gyrfalcon/examples/` (`profile_ingest`, `profile_query`, `profile_dense`,
`profile_incremental`) and `gyr bench` cover most paths. A change that makes something slower
needs a reason that outweighs it.

### Retrieval quality

Quality changes follow the protocol in [Methodology](../benchmarks/methodology.md):

1. Choose any setting on validation queries only.
2. Report test nDCG@10 on the tuning sets and on the four held-out sets, with paired bootstrap
   intervals against the current default.
3. Report query latency and index time beside quality.
4. A change that wins on tuning sets and loses on held-out sets is not merged.

Trace and replay ([Planner](../concepts/planner.md)) make most fusion experiments cheap: record
traces once, then try policies offline.

Negative results are useful. If an idea did not work, a short write-up of what was tried and
what it measured, added to the spec, saves the next person the same experiment.

### Format changes

A change to any file in a collection increments `FORMAT_VERSION` in
`crates/gyrfalcon/src/manifest.rs`, adds a line to its history comment, and makes older
collections fail to open with `IncompatibleIndex` on `format_version` rather than misread them.

## Pull requests

1. Fork, branch, and make the change.
2. Run the checks CI runs:

   ```sh
   cargo fmt --all --check
   cargo clippy --workspace --all-targets -- -D warnings
   cargo test --release
   ```

   and, if you touched the Python package, `maturin develop --release && pytest` in
   `crates/gyrfalcon-py`.
3. Update the documentation the change affects: these pages, the docstrings, and, for a design
   or format change, a dated section in
   [`superpowers/specs/2026-10-24-gyrfalcon-design.md`](../superpowers/specs/2026-10-24-gyrfalcon-design.md)
   that states what changed, why, and the measurement.
4. For a benchmark result, add a directory `bench/results/YYYY-MM-DD-topic/` with a `README.md`
   that names the commit and the command, and the raw JSON.
5. Open the pull request with a description of the problem, the change, and the evidence.

### Commit messages

A short summary line naming the area and the change, in the style of the existing history:

```text
lexical: block-max MaxScore search, identical to exhaustive scoring
full tier: rescore with dense weight 0.65
```

The body explains why, and includes the numbers when the change is about speed, memory, or
quality.

### Code style

- `rustfmt` with the repository's `rustfmt.toml` (120 columns).
- Clippy clean with warnings denied.
- Comments explain why, not what. A constant that was measured says where its value came from.
- Write code that reads like the code around it.

## License

Gyrfalcon is licensed under the [Apache License 2.0](../../LICENSE). By contributing you agree
that your contribution is licensed under the same terms, as section 5 of the license states.

---

<sub>Next: [Development setup](development-setup.md)</sub>
