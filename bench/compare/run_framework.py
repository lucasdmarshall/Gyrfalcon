"""One framework, one dataset, one embedder: build the framework's standard hybrid retriever
(BM25 + dense, fused), time it, run every test query, and write a run file plus a JSON report.

    python run_framework.py FRAMEWORK BEIR_DIR EMBEDDER THREADS OUT_PREFIX [MAX_QUERIES] [SEED]

FRAMEWORK is llamaindex, langchain, or haystack; EMBEDDER is bge-small or potion.
MAX_QUERIES and SEED are optional. When MAX_QUERIES is smaller than the judged set, only that
fixed-seed sample is indexed against and scored. The corpus is always the full dataset.
"""
import json
import sys
import time

from common import Embedder, evaluate, load_beir, subsample_queries, timed_queries, write_run

DEPTH = 100


def llamaindex(docs, embedder):
    from llama_index.core import VectorStoreIndex
    from llama_index.core.embeddings import BaseEmbedding
    from llama_index.core.llms import MockLLM
    from llama_index.core.retrievers import QueryFusionRetriever
    from llama_index.core.schema import TextNode
    from llama_index.retrievers.bm25 import BM25Retriever

    class Adapter(BaseEmbedding):
        def _get_query_embedding(self, query):
            return embedder.embed([query])[0].tolist()

        async def _aget_query_embedding(self, query):
            return self._get_query_embedding(query)

        def _get_text_embedding(self, text):
            return embedder.embed([text])[0].tolist()

        def _get_text_embeddings(self, texts):
            return embedder.embed(texts).tolist()

    nodes = [TextNode(text=text, id_=doc_id) for doc_id, text in docs]
    index = VectorStoreIndex(nodes, embed_model=Adapter(embed_batch_size=256))
    bm25 = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=DEPTH)
    fused = QueryFusionRetriever(
        [index.as_retriever(similarity_top_k=DEPTH), bm25],
        similarity_top_k=DEPTH,
        num_queries=1,
        mode="reciprocal_rerank",
        use_async=False,
        llm=MockLLM(),
    )
    return lambda text: [hit.node.node_id for hit in fused.retrieve(text)]


def langchain(docs, embedder):
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import InMemoryVectorStore

    class Adapter(Embeddings):
        def embed_documents(self, texts):
            return embedder.embed(texts).tolist()

        def embed_query(self, text):
            return embedder.embed([text])[0].tolist()

    documents = [Document(page_content=text, metadata={"id": doc_id}) for doc_id, text in docs]
    store = InMemoryVectorStore.from_documents(documents, embedding=Adapter())
    bm25 = BM25Retriever.from_documents(documents, k=DEPTH)
    fused = EnsembleRetriever(retrievers=[bm25, store.as_retriever(search_kwargs={"k": DEPTH})], weights=[0.5, 0.5])
    return lambda text: [d.metadata["id"] for d in fused.invoke(text)][:DEPTH]


def haystack(docs, embedder):
    from haystack import Document, Pipeline, component
    from haystack.components.joiners import DocumentJoiner
    from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
    from haystack.document_stores.in_memory import InMemoryDocumentStore

    @component
    class QueryEmbedder:
        @component.output_types(embedding=list[float])
        def run(self, text: str):
            return {"embedding": embedder.embed([text])[0].tolist()}

    store = InMemoryDocumentStore()
    vectors = embedder.embed([text for _, text in docs])
    store.write_documents(
        [Document(id=doc_id, content=text, embedding=v.tolist()) for (doc_id, text), v in zip(docs, vectors)]
    )
    pipeline = Pipeline()
    pipeline.add_component("embedder", QueryEmbedder())
    pipeline.add_component("bm25", InMemoryBM25Retriever(document_store=store, top_k=DEPTH))
    pipeline.add_component("dense", InMemoryEmbeddingRetriever(document_store=store, top_k=DEPTH))
    pipeline.add_component("join", DocumentJoiner(join_mode="reciprocal_rank_fusion", top_k=DEPTH))
    pipeline.connect("embedder.embedding", "dense.query_embedding")
    pipeline.connect("bm25", "join")
    pipeline.connect("dense", "join")

    def search(text):
        out = pipeline.run({"embedder": {"text": text}, "bm25": {"query": text}})
        return [d.id for d in out["join"]["documents"]]

    return search


def main():
    framework, dataset, embedder_name, threads, out = sys.argv[1:6]
    # Optional: MAX_QUERIES SEED. Absent, zero, or larger than the judged set keeps every query.
    max_queries = int(sys.argv[6]) if len(sys.argv) > 6 else 0
    seed = int(sys.argv[7]) if len(sys.argv) > 7 else 20261029
    docs, queries, qrels = load_beir(dataset)
    queries, sample = subsample_queries(queries, max_queries, seed)
    embedder = Embedder(embedder_name, int(threads))
    embedder.embed(["warm up the model"])
    started = time.perf_counter()
    search = {"llamaindex": llamaindex, "langchain": langchain, "haystack": haystack}[framework](docs, embedder)
    index_seconds = time.perf_counter() - started
    run, latency = timed_queries(queries, search)
    write_run(out + ".tsv", run)
    report = {
        "system": framework,
        "dataset": dataset.rstrip("/").split("/")[-1],
        "embedder": embedder_name,
        "threads": int(threads),
        "docs": len(docs),
        "queries": len(queries),
        "index_seconds": index_seconds,
        **latency,
        **evaluate(run, qrels, queries),
    }
    if sample is not None:
        report["subsample"] = sample
    json.dump(report, open(out + ".json", "w"), indent=1)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
