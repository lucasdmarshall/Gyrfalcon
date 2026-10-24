#!/bin/sh
# Downloads ONNX Runtime and the models of both tiers, then prints the environment to use them.
#   lite: minishlab/potion-retrieval-32M (static, no runtime needed)
#   full: bge-small-en-v1.5 int8 ONNX, plus the fp32 model and the MiniLM reranker
#   scripts/fetch-models.sh [DIR]   (default: .models)
# Needs network access to pypi.org, files.pythonhosted.org, and huggingface.co.
set -eu
DIR=${1:-.models}
ORT_VERSION=${ORT_VERSION:-1.31.0}
mkdir -p "$DIR/ort" "$DIR/bge-small-en-v1.5" "$DIR/bge-small-en-v1.5-q8" "$DIR/ms-marco-MiniLM-L-6-v2" "$DIR/potion-retrieval-32M"
DIR=$(cd "$DIR" && pwd)

if [ ! -f "$DIR/ort/libonnxruntime.so" ]; then
    pip download --quiet --no-deps --only-binary=:all: -d "$DIR/ort" "onnxruntime==$ORT_VERSION"
    python3 -m zipfile -e "$DIR"/ort/onnxruntime-*.whl "$DIR/ort/wheel"
    cp "$DIR"/ort/wheel/onnxruntime/capi/libonnxruntime.so.* "$DIR/ort/libonnxruntime.so"
    rm -rf "$DIR/ort/wheel" "$DIR"/ort/onnxruntime-*.whl
fi

# get REPO FILE TARGET
get() {
    [ -f "$3" ] || curl -fsSL -o "$3" "https://huggingface.co/$1/resolve/main/$2"
}
get BAAI/bge-small-en-v1.5 tokenizer.json "$DIR/bge-small-en-v1.5/tokenizer.json"
get BAAI/bge-small-en-v1.5 onnx/model.onnx "$DIR/bge-small-en-v1.5/model.onnx"
get Xenova/bge-small-en-v1.5 tokenizer.json "$DIR/bge-small-en-v1.5-q8/tokenizer.json"
get Xenova/bge-small-en-v1.5 onnx/model_quantized.onnx "$DIR/bge-small-en-v1.5-q8/model.onnx"
get Xenova/ms-marco-MiniLM-L-6-v2 tokenizer.json "$DIR/ms-marco-MiniLM-L-6-v2/tokenizer.json"
get Xenova/ms-marco-MiniLM-L-6-v2 onnx/model.onnx "$DIR/ms-marco-MiniLM-L-6-v2/model.onnx"
for file in config.json model.safetensors tokenizer.json; do
    get minishlab/potion-retrieval-32M "$file" "$DIR/potion-retrieval-32M/$file"
done

echo "export ORT_DYLIB_PATH=$DIR/ort/libonnxruntime.so"
echo "export GYR_MODEL_DIR=$DIR/bge-small-en-v1.5"
echo "export GYR_RERANKER_DIR=$DIR/ms-marco-MiniLM-L-6-v2"
echo "export GYR_MODELS=$DIR"
