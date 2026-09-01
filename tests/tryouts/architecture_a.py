"""Architecture A (tryout, not in active use): the original single
monolithic call -- one LLM call sees the whole reranked context and both
judges relevance and drafts the answer in one shot. Self-contained: does NOT
share retrieval/generation logic with src/components (architecture C, the
current best) -- only the underlying API clients and prompts.yaml are
shared, via src/clients.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from clients import client, collection, bm25_retriever, cohere_client, PROMPTS
import bm25s

ACTIVE_PROMPT_VERSION = "system_prompt_v8"
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
    candidate_ids = fused_ids[:candidate_k]
    candidate_texts = fused_texts[:candidate_k]
    response = cohere_client.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=candidate_texts,
        top_n=k,
    )
    return [candidate_ids[result.index] for result in response.results]


def retrieve_fuse_rerank(rerank_query, sub_queries, top_k, candidate_k):
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


def ask(question, top_k=10, candidate_k=30, sub_queries=None):
    if sub_queries is None:
        sub_queries = [question]

    top_ids, chunk_lookup = retrieve_fuse_rerank(question, sub_queries, top_k, candidate_k)

    context_blocks = [
        f"[Source: {chunk_lookup[doc_id]['source_file']}]\n{chunk_lookup[doc_id]['text']}"
        for doc_id in top_ids
    ]
    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = PROMPTS[ACTIVE_PROMPT_VERSION]
    user_prompt = f"Retrieved Context:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=1000,
        frequency_penalty=0.4,
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else input("Question: ")
    print(ask(q))
