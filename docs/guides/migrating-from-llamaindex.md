<sub>[Docs](../README.md) / Guides</sub>

# Migrating from LlamaIndex

LlamaIndex builds hybrid retrieval from a reader, a node parser, an embedding model, a
`VectorStoreIndex`, a `BM25Retriever`, and a `QueryFusionRetriever` that joins the two.
Gyrfalcon replaces those with one `Collection` and fits back into LlamaIndex as a retriever, so
query engines, response synthesizers, and chat engines keep working.

## What maps to what

| LlamaIndex | Gyrfalcon |
|---|---|
| `SimpleDirectoryReader("docs/").load_data()` | `Collection.add_path("docs/")` |
| `SentenceSplitter`, `TokenTextSplitter` | built in: 2 KB chunks with 256 bytes of overlap, cut at paragraphs and sentences |
| `Settings.embed_model` | the tier's model, chosen at creation |
| `VectorStoreIndex(nodes)` | the collection's dense index |
| `BM25Retriever.from_defaults(nodes=...)` | the collection's BM25 index |
| `QueryFusionRetriever(mode="reciprocal_rerank")` | fusion inside `search`, weights tuned per tier |
| `index.insert(doc)`, `index.delete_ref_doc(id)` | `add`, `delete`; `update` to replace |
| `index.storage_context.persist()` / `load_index_from_storage` | nothing: writes are durable, `Collection(path)` reopens |
| `MetadataFilters([ExactMatchFilter(...)])` | `filter={"key": "value"}` |
| `index.as_retriever(similarity_top_k=k)` | the `GyrfalconRetriever` class below |
| `TextNode` with `ref_doc_id` | a hit: chunk text plus `doc_id` and byte offsets |

## Before

```python
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever

Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

documents = SimpleDirectoryReader("docs/").load_data()
nodes = SentenceSplitter(chunk_size=512, chunk_overlap=50).get_nodes_from_documents(documents)
index = VectorStoreIndex(nodes)
index.storage_context.persist("storage")

retriever = QueryFusionRetriever(
    [index.as_retriever(similarity_top_k=4),
     BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=4)],
    similarity_top_k=4,
    num_queries=1,               # no LLM query rewriting
    mode="reciprocal_rerank",
    use_async=False,
)
```

## After

```python
from gyrfalcon import Collection

col = Collection("./kb")                 # tier="full" for bge-small-en-v1.5
col.add_path("docs/")
retriever = GyrfalconRetriever(col, k=4)
```

## The retriever adapter

Gyrfalcon does not depend on LlamaIndex. This class makes a collection a LlamaIndex retriever:

```python
from typing import Any, Optional

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

from gyrfalcon import Collection


class GyrfalconRetriever(BaseRetriever):
    """A Gyrfalcon collection as a LlamaIndex retriever."""

    def __init__(self, collection: Collection, k: int = 4,
                 filter: Optional[dict[str, Any]] = None) -> None:
        self._collection = collection
        self._k = k
        self._filter = filter
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        hits = self._collection.search(query_bundle.query_str, k=self._k, filter=self._filter)
        return [
            NodeWithScore(
                node=TextNode(
                    id_=f"{hit.doc_id}#{hit.chunk_id}",
                    text=hit.text,
                    metadata={**hit.metadata, "doc_id": hit.doc_id,
                              "start": hit.start, "end": hit.end},
                ),
                score=hit.score,
            )
            for hit in hits
        ]
```

With a query engine:

```python
from llama_index.core.query_engine import RetrieverQueryEngine

engine = RetrieverQueryEngine.from_args(retriever, llm=llm)
print(engine.query("How many vacation days do new employees get?"))
```

`llm` is the LlamaIndex LLM you already use.

## Differences to plan for

**Nodes and documents.** LlamaIndex stores nodes, each with a `ref_doc_id`. Gyrfalcon stores
documents and returns chunks; `delete` and `update` take the document id and affect all of its
chunks at once. `update` is atomic: a query never sees half of the old and half of the new
document.

**Chunking.** The chunker is fixed at 2 KB windows with 256 bytes of overlap. LlamaIndex's
`SentenceSplitter(chunk_size=512)` counts tokens, so its chunks are roughly the same size in
English text. For structural splits (one node per section), split yourself and add each piece as
a document with an id like `"guide.md#install"`.

**Metadata filters.** Exact matches only: every key must match, and a list of values matches
any of them. There are no range or `NOT` operators. LlamaIndex filters with operators such as
`>=` need a coarser key (`"year_bucket": "2020s"`) or a post-filter on the hits.

**Excluded metadata keys.** LlamaIndex can embed metadata with the text
(`excluded_embed_metadata_keys`). Gyrfalcon never embeds metadata; to make a title searchable,
put it in the text.

**Async and concurrency.** `aretrieve` falls back to the synchronous method. Gyrfalcon releases
the GIL during a search, so threads querying one collection run in parallel.

**Persistence.** There is no storage context. The collection directory is the index, and it is
complete after every write; see [Persistence](persistence.md).

## What changes in the numbers

Same embedding model on both sides, LlamaIndex's `VectorStoreIndex` plus `BM25Retriever` (bm25s
with English stemming) plus `QueryFusionRetriever`, 4-vCPU VM
([results](../benchmarks/results.md)):

| | LlamaIndex | Gyrfalcon lite |
|---|---|---|
| nDCG@10, SciFact | 0.695 | 0.709 |
| nDCG@10, FiQA | 0.242 | 0.266 |
| Index FiQA (57,638 docs) | 32.8 s | 2.0 s |
| Query p50, FiQA | 1,436 ms | 2.7 ms |
| Peak memory, FiQA | 2,116 MB | 414 MB |

With bge-small-en-v1.5 on SciFact, LlamaIndex scored 0.727 and Gyrfalcon's full tier 0.743.
With the same fusion rule (reciprocal rank) on both sides, the two score the same on SciFact
(0.727 each); the difference comes from Gyrfalcon's tuned fusion rule, which is disclosed in
[Methodology](../benchmarks/methodology.md#disclosures).

---

<sub>Previous: [Migrating from LangChain](migrating-from-langchain.md) · Next: [Architecture](../concepts/architecture.md)</sub>
