# Writes a tiny BERT-shaped embedder and cross-encoder for the ort round-trip test:
#   python3 make_ort_fixture.py OUT_DIR
#   GYR_ORT_FIXTURE=OUT_DIR ORT_DYLIB_PATH=/path/to/libonnxruntime.so cargo test -p gyrfalcon --test m3_ort
# Needs the onnx, numpy, and tokenizers Python packages.
import sys, numpy as np, onnx
from onnx import helper, TensorProto, numpy_helper
from tokenizers import Tokenizer, models, pre_tokenizers, processors
out = sys.argv[1]
vocab = {"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3}
for w in "memory safety in rust bread cooking gardening e0308 alpha beta".split(): vocab[w]=len(vocab)
tok = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
from tokenizers import normalizers
tok.normalizer = normalizers.Lowercase()
tok.pre_tokenizer = pre_tokenizers.Whitespace()
tok.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]",2),("[SEP]",3)])
import os
def model(kind, path):
    os.makedirs(path, exist_ok=True)
    tok.save(os.path.join(path, "tokenizer.json"))
    rng = np.random.default_rng(1)
    dim = 384 if kind == "embed" else 8
    table = rng.standard_normal((len(vocab), dim)).astype(np.float32)
    nodes = [helper.make_node("Gather", ["table", "input_ids"], ["emb"]),
             helper.make_node("Cast", ["attention_mask"], ["maskf"], to=TensorProto.FLOAT),
             helper.make_node("Unsqueeze", ["maskf", "axes"], ["mask3"]),
             helper.make_node("Mul", ["emb", "mask3"], ["masked"])]
    inits = [numpy_helper.from_array(table, "table"), numpy_helper.from_array(np.array([2], dtype=np.int64), "axes")]
    if kind == "embed":
        # Every position carries the masked token sum, so CLS pooling sees the whole text.
        nodes.append(helper.make_node("ReduceSum", ["masked", "axes1"], ["summed"], keepdims=1))
        nodes.append(helper.make_node("Shape", ["masked"], ["shape"]))
        nodes.append(helper.make_node("Expand", ["summed", "shape"], ["last_hidden_state"]))
        inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "axes1"))
        output = helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["b", "s", dim])
    else:
        nodes.append(helper.make_node("ReduceSum", ["masked", "axes12"], ["logits"], keepdims=0))
        nodes.append(helper.make_node("Unsqueeze", ["logits", "axes1b"], ["out"]))
        inits.append(numpy_helper.from_array(np.array([1, 2], dtype=np.int64), "axes12"))
        inits.append(numpy_helper.from_array(np.array([1], dtype=np.int64), "axes1b"))
        output = helper.make_tensor_value_info("out", TensorProto.FLOAT, ["b", 1])
    inputs = [helper.make_tensor_value_info(n, TensorProto.INT64, ["b", "s"]) for n in ["input_ids", "attention_mask", "token_type_ids"]]
    graph = helper.make_graph(nodes, kind, inputs, [output], inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    onnx.save(m, os.path.join(path, "model.onnx"))
model("embed", os.path.join(out, "embed"))
model("rerank", os.path.join(out, "rerank"))
