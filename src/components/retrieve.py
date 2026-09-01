"""Retrieval + Fusion + Rerank component: for each sub_query, dense+BM25
retrieve into one shared candidate pool, RRF-fuse everything together, then
Cohere-rerank the fused pool against rerank_query (normally the ORIGINAL
question, not any individual sub_query).

Standalone: `python retrieve.py "some question"` prints the top chunks.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clients import client, collection, bm25_retriever, cohere_client
import bm25s

RERANK_MODEL = "rerank-v4.0-fast"
K_RRF = 60


def reciprocal_rank_fusion(dense_ids, bm25_ids, k=K_RRF):
    """Merge two ranked id lists into one score per id: sum(1 / (k + rank))."""
    scores = {}
    for rank, doc_id in enumerate(dense_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, doc_id in enumerate(bm25_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def search_reranked(query, k, candidate_k, fused_ids, fused_texts):
    """Cross-encoder rerank the top `candidate_k` RRF results down to the final top `k`."""
    candidate_ids = fused_ids[:candidate_k]
    candidate_texts = fused_texts[:candidate_k]

    response = cohere_client.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=candidate_texts,
        top_n=k,
    )
    # response.results is already ordered by relevance; .index points back
    # into candidate_texts/candidate_ids, not into the full fused list
    return [candidate_ids[result.index] for result in response.results]


def retrieve(rerank_query, sub_queries, top_k, candidate_k):
    """Returns (top_ids, chunk_lookup) where chunk_lookup maps id ->
    {text, source_file} for every candidate considered, not just the
    reranked top_k."""
    chunk_lookup = {}
    all_dense_ids = []
    all_bm25_ids = []

    for sub_query in sub_queries:
        query_embedding = client.embeddings.create(
            input=[sub_query], model="text-embedding-3-small"
        ).data[0].embedding
        results_dense = collection.query(query_embeddings=[query_embedding], n_results=candidate_k)
        all_dense_ids.extend(results_dense["ids"][0])
        for id_, doc, meta in zip(
            results_dense["ids"][0], results_dense["documents"][0], results_dense["metadatas"][0]
        ):
            chunk_lookup.setdefault(id_, {"text": doc, "source_file": meta["source_file"]})

        query_tokens_bm25 = bm25s.tokenize([sub_query], stopwords="en")
        bm25_hits, _ = bm25_retriever.retrieve(query_tokens_bm25, k=candidate_k)
        bm25_hits = bm25_hits[0]
        all_bm25_ids.extend(hit["id"] for hit in bm25_hits)
        for hit in bm25_hits:
            chunk_lookup.setdefault(hit["id"], {"text": hit["text"], "source_file": hit["source_file"]})

    fused = reciprocal_rank_fusion(all_dense_ids, all_bm25_ids, k=K_RRF)
    fused_ids = [doc_id for doc_id, _ in fused]
    fused_texts = [chunk_lookup[doc_id]["text"] for doc_id in fused_ids]

    top_ids = search_reranked(rerank_query, top_k, candidate_k, fused_ids, fused_texts)
    return top_ids, chunk_lookup


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else input("Question: ")
    top_ids, chunk_lookup = retrieve(question, [question], top_k=6, candidate_k=30)
    print(json.dumps({"top_ids": top_ids, "chunks": {i: chunk_lookup[i] for i in top_ids}}, indent=2))
