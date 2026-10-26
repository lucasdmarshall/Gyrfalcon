#!/bin/sh
# Reruns the benchmarks behind docs/benchmarks/results.md.
#   scripts/reproduce.sh lite [RUNS]       the lite tier on eight BEIR sets, RUNS times (default 5)
#   scripts/reproduce.sh compare [CORES]   the head-to-head against LlamaIndex, LangChain, and
#                                          Haystack on CORES pinned cores (default 4); takes hours
# Builds gyr, downloads the models and datasets it lacks, and writes every report and a
# summary.md to bench/reports/reproduce-MODE-TIMESTAMP/. Needs a Rust toolchain, curl, pip, and
# python3; compare also needs taskset.
set -eu
MODE=${1:-lite}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
DATASETS="scifact nfcorpus arguana fiqa trec-covid scidocs quora webis-touche2020"
case "$MODE" in
lite | compare) ;;
*)
    echo "usage: scripts/reproduce.sh lite [RUNS] | compare [CORES]" >&2
    exit 2
    ;;
esac
OUT=bench/reports/reproduce-$MODE-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"

cargo build --release -p gyr-cli
scripts/fetch-models.sh .models > /dev/null
# shellcheck disable=SC2086
scripts/fetch-beir.sh .data/beir $DATASETS > /dev/null

if [ "$MODE" = lite ]; then
    RUNS=${2:-5}
    set --
    for name in $DATASETS; do
        set -- "$@" ".data/beir/$name"
    done
    i=1
    while [ "$i" -le "$RUNS" ]; do
        GYR_MODELS=.models target/release/gyr bench beir --dataset "$@" --out "$OUT/run$i.json" > "$OUT/run$i.log" 2>&1
        echo "run $i of $RUNS done"
        i=$((i + 1))
    done
    python3 bench/summarize_runs.py "$OUT"/run*.json | tee "$OUT/summary.md"
else
    CORES=${2:-4}
    python3 -m venv .venv-compare
    .venv-compare/bin/pip install --quiet -r bench/compare/requirements.txt
    .venv-compare/bin/python bench/compare/compare.py "$OUT" --cores "$CORES" \
        --datasets scifact,nfcorpus,arguana,fiqa
    .venv-compare/bin/python bench/compare/summarize.py "$OUT/results.jsonl" | tee "$OUT/summary.md"
fi
echo "results in $OUT"
