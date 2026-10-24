#!/bin/sh
# Runs the lite-tier benches pinned to 1, 2, and 4 cores with every thread pool capped to match.
#   scripts/bench-low-cpu.sh [OUT_DIR]   (default: bench/reports/low-cpu)
# Needs `taskset`, the lite model (scripts/fetch-models.sh), and SciFact and FiQA
# (scripts/fetch-beir.sh).
set -eu
OUT=${1:-bench/reports/low-cpu}
G=${GYR:-target/release/gyr}
mkdir -p "$OUT"
for cores in 1 2 4; do
    last=$((cores - 1))
    export GYR_ORT_THREADS=$cores RAYON_NUM_THREADS=$cores
    run() { taskset -c "0-$last" "$G" "$@"; }
    run bench beir --dataset .data/beir/scifact .data/beir/fiqa --out "$OUT/beir-${cores}core.json" > /dev/null
    run bench retrieve --segments 1 --concurrency "$cores,$((cores * 4))" --out "$OUT/retrieve-${cores}core.json" > /dev/null
    run bench ingest --files 16 --chunks-per-file 2000 --out "$OUT/ingest-${cores}core.json" > /dev/null
    echo "$cores core(s) done"
done
