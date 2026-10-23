#!/bin/sh
# Downloads ONNX Runtime and the two default models, then prints the environment to use them.
#   scripts/fetch-models.sh [DIR]   (default: .models)
# Needs network access to pypi.org, files.pythonhosted.org, and huggingface.co.
set -eu
DIR=${1:-.models}
ORT_VERSION=${ORT_VERSION:-1.31.0}
mkdir -p "$DIR/ort" "$DIR/bge-small-en-v1.5" "$DIR/ms-marco-MiniLM-L-6-v2"
DIR=$(cd "$DIR" && pwd)

if [ ! -f "$DIR/ort/libonnxruntime.so" ]; then
    pip download --quiet --no-deps --only-binary=:all: -d "$DIR/ort" "onnxruntime==$ORT_VERSION"
    python3 -m zipfile -e "$DIR"/ort/onnxruntime-*.whl "$DIR/ort/wheel"
    cp "$DIR"/ort/wheel/onnxruntime/capi/libonnxruntime.so.* "$DIR/ort/libonnxruntime.so"
    rm -rf "$DIR/ort/wheel" "$DIR"/ort/onnxruntime-*.whl
fi

fetch() {
    repo=$1
    out=$2
    for file in tokenizer.json onnx/model.onnx; do
        target="$out/$(basename "$file")"
        [ -f "$target" ] || curl -fsSL -o "$target" "https://huggingface.co/$repo/resolve/main/$file"
    done
}
fetch BAAI/bge-small-en-v1.5 "$DIR/bge-small-en-v1.5"
fetch Xenova/ms-marco-MiniLM-L-6-v2 "$DIR/ms-marco-MiniLM-L-6-v2"

echo "export ORT_DYLIB_PATH=$DIR/ort/libonnxruntime.so"
echo "export GYR_MODEL_DIR=$DIR/bge-small-en-v1.5"
echo "export GYR_RERANKER_DIR=$DIR/ms-marco-MiniLM-L-6-v2"
