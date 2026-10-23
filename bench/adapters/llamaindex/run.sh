#!/bin/sh
# llamaindex adapter for `gyr bench accuracy`.
#
# Reads the fixture written by `gyr bench accuracy --prepare` from $BENCH_FIXTURE, prints the
# fixture chunk count, and execs $BENCH_ADAPTER_CMD, the runner that indexes those chunks
# unchanged (no re-splitting, no PDF parsing) and writes bench/reports/llamaindex.json in the
# AccuracyReport schema. Without a runner it exits 2 and writes no report.
set -eu
SYSTEM=llamaindex
if [ -z "${BENCH_FIXTURE:-}" ]; then
    echo "BENCH_FIXTURE is not set" >&2
    exit 2
fi
CHUNKS=$(grep -c . "$BENCH_FIXTURE/chunks.jsonl")
echo "chunks $CHUNKS"
if [ -z "${BENCH_ADAPTER_CMD:-}" ]; then
    echo "BENCH_ADAPTER_CMD is not set; no $SYSTEM runner is configured" >&2
    exit 2
fi
export BENCH_SYSTEM="$SYSTEM"
exec $BENCH_ADAPTER_CMD
