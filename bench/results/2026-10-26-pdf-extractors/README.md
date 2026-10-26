# PDF text extractors, 2026-10-26

Which extractor `gyrfalcon-ingest` should use for PDFs, measured on the documents a RAG
framework is asked to read: 100 arXiv papers from Open RAG Bench (2,747 pages, 232 MB), whose
reference text and 334 queries come with the benchmark. Cloud VM, 4 vCPU Intel Xeon @ 2.10 GHz.
Harness: `bench/pdf` (`src/main.rs`, `extract_py.py`, `score.py`, `orb_beir.py`).

## Text quality and cost

`score.py` compares each extractor's text with the benchmark's reference rendering of the same
paper, after removing the reference's LaTeX math, which no text extractor reproduces. Word F1
is the multiset overlap of prose words; trigram recall is the share of the reference's word
trigrams the extraction contains, which drops when columns interleave or words split.

CPU seconds are user + system time for all 100 papers in one process (`getrusage`), measured
while another benchmark used part of the machine; wall-clock rates would be noisier, CPU time
much less so.

| extractor | language | word F1 | trigram recall | CPU s, 100 papers | CPU ms / page | peak RSS |
|---|---|---|---|---|---|---|
| **PDFium + layout layer** (`gyrfalcon-ingest`) | C++ / Rust | **0.9046** | **0.8994** | 9.7 | 3.5 | 110 MB |
| PDFium, U+0002 removed | C++ | 0.9017 | 0.8981 | 9.3 | 3.4 | 110 MB |
| PDFium as is | C++ | 0.8926 | 0.8767 | 9.3 | 3.4 | 110 MB |
| pypdfium2 (PDFium) | Python / C++ | 0.8933 | 0.8786 | 9.6 | 3.5 | 73 MB |
| unpdf 0.50 | Rust | 0.9011 | 0.8947 | 38.4 (34.3 sequential) | 12.5 | 216 MB |
| pdf_oxide 0.3 | Rust | 0.8994 | 0.8744 | 29.7 | 10.8 | 220 MB |
| pdf-extract 0.12 | Rust | 0.8857 | 0.8589 | 34.9 | 12.7 | 215 MB |
| PyMuPDF 2.0 | Python / C | 0.8834 | 0.8566 | 11.7 | 4.3 | 185 MB |
| oxidize-pdf 5.5 | Rust | 0.8769 | 0.8369 | 12.5 | 4.6 | 156 MB |
| pypdf 6.2 | Python | 0.8666 | 0.8254 | 136.7 | 49.8 | 140 MB |
| lopdf 0.46 | Rust | 0.6621 | 0.5965 | ~5.8 | ~2.1 | — |

No extractor failed on any paper. unpdf runs pages on its own thread pool, so its wall time
(21.8 s) beats its CPU time; sequential, it is CPU-bound in parsing (rendering costs nothing)
with a flat profile: SipHash 15%, malloc/free 18%, memcmp 6%. PDFium needs one fix: where it
joins a word split by a soft hyphen it leaves U+0002 (`cov\x02erage`), which splits the word for
any tokenizer. Its output also carries C0 controls and Private Use glyphs from math fonts with
no Unicode mapping, including form feeds that are glyphs, not page breaks.

The layout layer (`crates/gyrfalcon-ingest/src/layout.rs`) costs 0.15 ms per page.

## Retrieval on the papers' queries

`orb_beir.py` cuts each extractor's text into passages with the chunker's rule (2,048-byte
windows cut at a blank line, a newline, or a sentence end, back half of the window first; no
overlap) and judges a passage relevant to a query when it shares at least half of the word
trigrams of the smaller of itself and the query's reference section. `gyr bench beir --tier lite`
then ranks the passages of all 100 papers for each of the 334 queries
(`retrieval-lite.json`).

| text | passages | nDCG@10 | MRR@10 | recall@100 | lexical nDCG@10 | dense nDCG@10 |
|---|---|---|---|---|---|---|
| benchmark's reference rendering | 5,257 | 0.6213 | 0.6616 | 0.8950 | 0.6501 | 0.4303 |
| **PDFium + layout layer** | 4,291 | 0.6553 | **0.6775** | **0.9362** | 0.6529 | 0.4740 |
| PDFium, U+0002 removed | 4,011 | **0.6620** | 0.6770 | 0.9259 | 0.6643 | 0.4636 |
| pypdfium2 | 4,033 | 0.6396 | 0.6500 | 0.9145 | 0.6454 | 0.4525 |
| unpdf | 4,368 | 0.6474 | 0.6776 | 0.9169 | 0.6542 | 0.4697 |
| pdf_oxide | 4,441 | 0.6503 | 0.6770 | 0.9303 | 0.6522 | 0.4690 |
| pdf-extract | 4,336 | 0.6438 | 0.6624 | 0.9177 | 0.6513 | 0.4745 |
| oxidize-pdf | 3,907 | 0.6503 | 0.6499 | 0.9166 | 0.6579 | 0.4667 |
| PyMuPDF | 4,243 | 0.6358 | 0.6494 | 0.9319 | 0.6454 | 0.4562 |
| pypdf | 4,019 | 0.6341 | 0.6338 | 0.9104 | 0.6392 | 0.4413 |
| lopdf | 4,395 | 0.5034 | 0.5226 | 0.7184 | 0.5085 | 0.3809 |

Every extractor but lopdf lands within about 0.03 nDCG@10, and paired bootstrap intervals on
334 queries are about ±0.03 wide, so retrieval does not separate them: cost does. Against plain
PDFium text the layout layer was not significantly different on any metric (nDCG@10 interval
-0.033 to +0.019), and neither was dropping running headers (they are often the paper's title,
which might have helped find the paper; it did not matter). It is kept for what retrieval
metrics do not see: paragraphs an answering model reads, no page furniture in citations, and
page offsets. The texts' passages are judged against different cuts, so compare rows loosely.

Two changes came out of this run:

- **PDFium** is the PDF backend: the best text after the U+0002 fix and 3.6× less CPU per page
  than the best pure-Rust extractor. It loads at run time like ONNX Runtime.
- **The chunker cuts in the window's back half first** (`aa76e2f`): a blank line early in a
  window no longer cuts off a fragment. On this set nDCG@10 rose 0.641 → 0.655 with layout
  paragraphs and 0.654 → 0.662 without; BEIR lite was unchanged to four decimals.

End to end, `gyr index` read, cleaned, chunked, embedded (lite tier), and indexed the 100
papers on one pinned core in 12.1 s wall, 11.3 s CPU: 4.1 ms per page.

## Reproduce

```
# PDFs: Open RAG Bench pdf_urls.json, first 100 ids in .data/pdf/ids.txt
cd bench/pdf && CARGO_TARGET_DIR=../../target/pdf-bench cargo build --release
PDFIUM_LIB=/path/to/libpdfium.so ../../target/pdf-bench/release/pdf-bench pdfium .data/pdf OUT > OUT/t-pdfium.jsonl
python extract_py.py pypdfium2 .data/pdf OUT > OUT/t-pypdfium2.jsonl
cargo run --release -p gyrfalcon-ingest --example clean_pages -- OUT/pdfium OUT/pdfium-layout
python score.py ORB/pdf/arxiv/corpus OUT OUT/t-*.jsonl      # needs OUT/pages from `pdf-bench pages`
python orb_beir.py ORB/pdf/arxiv .data/pdf/ids.txt OUT/pdfium-layout BEIR/pdfium-layout
gyr bench beir --tier lite --dataset BEIR/* --fusions configured,lexical,dense
```
