#!/bin/sh
# Downloads BEIR datasets (corpus.jsonl, queries.jsonl, qrels/test.tsv) for `gyr bench beir`.
#   scripts/fetch-beir.sh [DIR] [DATASET...]   (default: .data/beir scifact nfcorpus fiqa arguana)
set -eu
DIR=${1:-.data/beir}
[ $# -gt 0 ] && shift
DATASETS=${*:-scifact nfcorpus fiqa arguana}
mkdir -p "$DIR"
for name in $DATASETS; do
    if [ ! -f "$DIR/$name/corpus.jsonl" ]; then
        curl -fsSL -o "$DIR/$name.zip" "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/$name.zip"
        python3 -m zipfile -e "$DIR/$name.zip" "$DIR"
        rm "$DIR/$name.zip"
    fi
    echo "$DIR/$name"
done
