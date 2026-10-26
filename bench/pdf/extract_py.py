"""Python PDF extractors, one at a time, single-threaded: the libraries RAG frameworks call.

    python extract_py.py EXTRACTOR PDF_DIR OUT_DIR

EXTRACTOR is pypdf (LlamaIndex, LangChain, and Haystack default), pypdfium2, or pymupdf.
Prints one JSON line per file like the Rust harness.
"""
import json
import os
import sys
import time

extractor, pdf_dir, out_dir = sys.argv[1:4]
out = os.path.join(out_dir, extractor)
os.makedirs(out, exist_ok=True)


def extract(path):
    if extractor == "pypdf":
        from pypdf import PdfReader

        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if extractor == "pypdfium2":
        import pypdfium2

        pdf = pypdfium2.PdfDocument(path)
        return "\n".join(pdf[i].get_textpage().get_text_range() for i in range(len(pdf)))
    if extractor == "pymupdf":
        import pymupdf

        with pymupdf.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    raise ValueError(extractor)


for name in sorted(f for f in os.listdir(pdf_dir) if f.endswith(".pdf")):
    path = os.path.join(pdf_dir, name)
    started = time.perf_counter()
    try:
        text, error = extract(path), None
    except Exception as err:  # noqa: BLE001 - a failure is a result here
        text, error = "", str(err)[:200]
    ms = (time.perf_counter() - started) * 1000
    if error is None:
        open(os.path.join(out, name[:-4] + ".txt"), "w").write(text)
    print(json.dumps({"extractor": extractor, "file": name[:-4], "bytes": os.path.getsize(path), "ms": ms,
                      "chars": len(text), "error": error}), flush=True)
