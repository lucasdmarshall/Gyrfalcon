<sub>[Docs](../README.md) / Guides</sub>

# Migrating from LangChain

A typical LangChain hybrid retriever is assembled from five parts: a loader, a text splitter, an
embedding model, a vector store, and a BM25 retriever joined to it by an `EnsembleRetriever`.
Gyrfalcon replaces all five with one `Collection`, and plugs back into the rest of a LangChain
pipeline (prompts, models, chains) as an ordinary retriever.

## What maps to what

| LangChain | Gyrfalcon |
|---|---|
| `PyPDFLoader`, `DirectoryLoader`, `BSHTMLLoader` | `Collection.add_path`, `gyrfalcon.read_file` |
| `RecursiveCharacterTextSplitter` | built in: 2 KB chunks with 256 bytes of overlap, cut at paragraphs and sentences |
| `HuggingFaceEmbeddings`, `OpenAIEmbeddings` | the tier's model, chosen at creation (`tier="lite"` or `"full"`) |
| `FAISS`, `Chroma`, `InMemoryVectorStore` | the collection's dense index |
| `BM25Retriever` (rank_bm25) | the collection's BM25 index, with stemming and stopwords |
| `EnsembleRetriever(weights=[0.5, 0.5])` | fusion inside `search`, weights tuned per tier |
| `vectorstore.add_documents(docs, ids=...)` | `add`, `add_many` |
| `vectorstore.similarity_search(q, k, filter=...)` | `search(q, k, filter=...)` |
| `vectorstore.delete(ids)` | `delete(doc_id)`; `update(doc_id, text)` to replace |
| `FAISS.save_local` / `load_local` | nothing: writes are durable, `Collection(path)` reopens |
| `vectorstore.as_retriever()` | the `GyrfalconRetriever` class below |

## Before

```python
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = DirectoryLoader("docs/", glob="**/*.md", loader_cls=TextLoader).load()
chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss-index")

bm25 = BM25Retriever.from_documents(chunks, k=4)    # rebuilt from the chunks on every start
retriever = EnsembleRetriever(
    retrievers=[bm25, vectorstore.as_retriever(search_kwargs={"k": 4})],
    weights=[0.5, 0.5],
)
```

## After

```python
from gyrfalcon import Collection

col = Collection("./kb")                 # tier="full" for bge-small-en-v1.5
col.add_path("docs/")                    # loads, splits, embeds, and indexes both legs
retriever = GyrfalconRetriever(collection=col, k=4)
```

Both indexes persist in `./kb`, including BM25, which LangChain's `BM25Retriever` keeps only in
memory.

## The retriever adapter

Gyrfalcon does not depend on LangChain. This class, about twenty lines, makes a collection a
LangChain retriever, so it works anywhere a retriever does: `retriever.invoke(query)`, LCEL
chains, and agents.

```python
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from gyrfalcon import Collection


class GyrfalconRetriever(BaseRetriever):
    """A Gyrfalcon collection as a LangChain retriever."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    collection: Collection
    k: int = 4
    filter: Optional[dict[str, Any]] = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        hits = self.collection.search(query, k=self.k, filter=self.filter)
        return [
            Document(
                id=f"{hit.doc_id}#{hit.chunk_id}",
                page_content=hit.text,
                metadata={**hit.metadata, "doc_id": hit.doc_id, "score": hit.score,
                          "start": hit.start, "end": hit.end},
            )
            for hit in hits
        ]
```

Used in a chain:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

prompt = ChatPromptTemplate.from_template(
    "Answer from the context.\n\nContext:\n{context}\n\nQuestion: {question}"
)

def join(docs):
    return "\n\n".join(d.page_content for d in docs)

chain = {"context": retriever | join, "question": RunnablePassthrough()} | prompt | llm
chain.invoke("How many vacation days do new employees get?")
```

`llm` is whatever chat model you already use.

## Differences to plan for

**Documents and chunks.** In LangChain you split first and store chunks; each chunk is a
`Document`. In Gyrfalcon you add whole documents and the collection splits them; each hit is a
chunk and carries its parent `doc_id` and byte offsets. Delete and update work on the whole
document.

**Chunk size.** The chunker cuts 2 KB windows with 256 bytes of overlap and is not
configurable from Python. If you need other boundaries, for example one chunk per FAQ entry,
split the text yourself and add each piece as its own document (`"faq.md#12"`). A piece shorter
than 2 KB stays one chunk.

**Metadata.** LangChain metadata can hold any JSON. Gyrfalcon metadata is for filtering: keys
map to strings, numbers, or booleans (or lists of them), stored as text, and filters are exact
matches. Range filters such as `year > 2020` are not supported; store a coarser key such as
`decade` instead.

**Scores.** `EnsembleRetriever` returns documents without scores. Gyrfalcon returns a fused score
per hit, comparable within one query. The adapter puts it in `metadata["score"]`.

**Embeddings.** There is no embeddings object to pass around: the collection owns its model. A
collection built with one tier cannot be queried with another. To use a model that is not one of
the two tiers, see [Custom embedders](custom-embedders.md).

**Async.** `BaseRetriever` runs the synchronous method in a thread pool for `ainvoke`. Gyrfalcon
releases the GIL during a search, so concurrent searches from threads run in parallel.

## What changes in the numbers

On the BEIR sets of the [head-to-head](../benchmarks/results.md), with the same embedding model
on both sides and LangChain's standard `BM25Retriever` plus `InMemoryVectorStore` plus
`EnsembleRetriever`, the lite tier measured:

| | LangChain | Gyrfalcon lite |
|---|---|---|
| nDCG@10, SciFact | 0.604 | 0.709 |
| nDCG@10, FiQA | 0.188 | 0.266 |
| Index FiQA (57,638 docs) | 20.6 s | 2.0 s |
| Query p50, FiQA | 2,398 ms | 2.7 ms |
| Peak memory, FiQA | 2,791 MB | 414 MB |

LangChain's `BM25Retriever` tokenizes with a plain whitespace split by default, without
lowercasing or stemming. That likely accounts for much of the quality gap, but the legs were not
measured separately per framework, so treat it as a likely cause, not a measured one
([Methodology](../benchmarks/methodology.md#disclosures)). A LangChain pipeline with a better BM25
tokenizer would close part of the gap; the speed and memory differences come from the engine.

---

<sub>Previous: [Persistence](persistence.md) · Next: [Migrating from LlamaIndex](migrating-from-llamaindex.md)</sub>
