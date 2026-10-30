# One stemmed BM25 setup beside the library defaults

Lite tier, potion-retrieval-32M, depth 100, 4 vCPU, the four tuning sets. The default cells are
the 2026-10-25 run (Python 3.13) and were not repeated. The stemmed cells are this machine
(Python 3.12.3), 29 Oct 2026, 19:02–19:38 UTC (01:32–02:08 Asia/Rangoon on 30 Oct). nDCG is the
comparison. The clocks are not comparable to the 2026-10-25 table.

`langchain-stem` replaces LangChain's whitespace split with lowercase, the `\b\w\w+\b` word
pattern, and an English Snowball stemmer (`PyStemmer`). `haystack-stem` keeps Haystack's own
BM25 regex and its lowercase, and stems those tokens. Dense stores and fusion are unchanged.
Nothing here was chosen by looking at a score. LlamaIndex's default already stems
(`BM25Retriever`, bm25s, `language="en"`, `skip_stemming=False`), so it has no second row.

Held-out sets were not run. Quora and Touché-2020 are the ones that swapped in the default run.

| Dataset | Gyrfalcon configured | LlamaIndex default | LangChain default | LangChain stem | Haystack default | Haystack stem |
|---|---:|---:|---:|---:|---:|---:|
| SciFact | 0.7088 | 0.6945 | 0.6040 | 0.6905 | 0.6739 | 0.6908 |
| NFCorpus | 0.3435 | 0.3081 | 0.2543 | 0.3073 | 0.3265 | 0.3385 |
| ArguAna | 0.4912 | 0.4725 | 0.3867 | 0.4260 | 0.4414 | 0.4260 |
| FiQA | 0.2659 | 0.2420 | 0.1883 | 0.2373 | 0.2388 | 0.2446 |

LangChain's gap was mostly the tokenizer. Stemming raises it by 0.086 on SciFact, 0.053 on
NFCorpus, 0.039 on ArguAna, and 0.049 on FiQA. It is then level with LlamaIndex on NFCorpus
(0.3073 vs 0.3081) and still short of LlamaIndex on ArguAna (0.4260 vs 0.4725) and of
Gyrfalcon configured on every set.

Haystack already lowercases. Stemming adds 0.017 on SciFact, 0.012 on NFCorpus, and 0.006 on
FiQA. On ArguAna it drops 0.015 (0.4414 to 0.4260). That cell got worse.

Neither stemmed row passes Gyrfalcon configured. The closest is Haystack stem on NFCorpus,
0.3385 against 0.3435.

Query latency for the stemmed cells, this machine only: LangChain p50 65 / 38 / 253 / 809 ms
and Haystack 102 / 71 / 278 / 1,229 ms on SciFact, NFCorpus, ArguAna, FiQA. These are not the
2026-10-25 clocks.
