"""Architecture B (tryout, not in active use): decompose -> retrieve+rerank+
verify each sub-query independently -> synthesize the verified sub-answers
into one response. Unlike architecture A, no single LLM call ever has to
scan the whole candidate pool and judge relevance for the full original
question at once. Self-contained: does NOT share retrieval/generation logic
with src/components (architecture C, the current best) -- only the
underlying API clients and prompts.yaml are shared, via src/clients.py.
"""
import sys
import time
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from clients import client, collection, bm25_retriever, cohere_client, PROMPTS
import bm25s

QUERY_DECOMPOSITION_PROMPT = "query_decomposition_prompt_v3"
SUBQUERY_VERIFY_PROMPT = "subquery_verify_prompt_v1"
SUBQUERY_SYNTHESIS_PROMPT = "subquery_synthesis_prompt_v1"
RERANK_MODEL = "rerank-v4.0-fast"
K_RRF = 60


def reciprocal_rank_fusion(dense_ids, bm25_ids, k=K_RRF):
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


def decompose_query(question):
    system_prompt = PROMPTS[QUERY_DECOMPOSITION_PROMPT]
    user_prompt = f"Question: {question}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=300,
    )
    try:
        sub_queries = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        sub_queries = [question]
    return sub_queries


def retrieve_and_rerank(query, top_k, candidate_k):
    """Dense+BM25 retrieve, RRF-fuse, and Cohere-rerank for a single query
    end to end, against that same query throughout."""
    chunk_lookup = {}

    query_embedding = client.embeddings.create(
        input=[query], model="text-embedding-3-small"
    ).data[0].embedding
    results_dense = collection.query(query_embeddings=[query_embedding], n_results=candidate_k)
    dense_ids = results_dense["ids"][0]
    for id_, doc, meta in zip(
        results_dense["ids"][0], results_dense["documents"][0], results_dense["metadatas"][0]
    ):
        chunk_lookup.setdefault(id_, {"text": doc, "source_file": meta["source_file"]})

    query_tokens_bm25 = bm25s.tokenize([query], stopwords="en")
    bm25_hits, _ = bm25_retriever.retrieve(query_tokens_bm25, k=candidate_k)
    bm25_hits = bm25_hits[0]
    bm25_ids = [hit["id"] for hit in bm25_hits]
    for hit in bm25_hits:
        chunk_lookup.setdefault(hit["id"], {"text": hit["text"], "source_file": hit["source_file"]})

    fused = reciprocal_rank_fusion(dense_ids, bm25_ids, k=K_RRF)
    fused_ids = [doc_id for doc_id, _ in fused]
    fused_texts = [chunk_lookup[doc_id]["text"] for doc_id in fused_ids]

    top_ids = search_reranked(query, top_k, candidate_k, fused_ids, fused_texts)
    return top_ids, chunk_lookup


def verify_sub_query(sub_query, top_ids, chunk_lookup):
    """Returns {"answered": bool, "content": str|None}."""
    context_blocks = [
        f"[Source: {chunk_lookup[doc_id]['source_file']}]\n{chunk_lookup[doc_id]['text']}"
        for doc_id in top_ids
    ]
    context = "\n\n---\n\n".join(context_blocks)

    system_prompt = PROMPTS[SUBQUERY_VERIFY_PROMPT]
    user_prompt = f"Sub-question: {sub_query}\n\nContext:\n{context}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=600,
        frequency_penalty=0.4,
        response_format={"type": "json_object"},
    )
    try:
        result = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        result = {}
    return {"answered": bool(result.get("answered")), "content": result.get("content")}


def synthesize_answer(question, sub_results):
    parts = []
    for r in sub_results:
        finding = r["content"] if r["answered"] and r["content"] else "UNANSWERED - not covered by the knowledge base"
        parts.append(f"Sub-question: {r['sub_query']}\nVerified finding: {finding}")
    sub_answers_block = "\n\n".join(parts)

    system_prompt = PROMPTS[SUBQUERY_SYNTHESIS_PROMPT]
    user_prompt = f"Original developer question: {question}\n\nVerified sub-answers:\n{sub_answers_block}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=1000,
        frequency_penalty=0.4,
    )
    return response.choices[0].message.content


def ask_v2(question, sub_query_top_k=6, candidate_k=20, sub_queries=None):
    if sub_queries is None:
        sub_queries = decompose_query(question)

    sub_results = []
    for i, sub_query in enumerate(sub_queries):
        top_ids, chunk_lookup = retrieve_and_rerank(sub_query, sub_query_top_k, candidate_k)
        verified = verify_sub_query(sub_query, top_ids, chunk_lookup)
        sub_results.append({"sub_query": sub_query, **verified})

        ##################################################################
        # TEMP: Cohere trial key caps rerank calls at 10/min, and this
        # architecture makes one rerank call per sub-query instead of one
        # per question. Delete this block once on a paid key.
        if i < len(sub_queries) - 1:
            time.sleep(6.5)
        ##################################################################

    return synthesize_answer(question, sub_results)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else input("Question: ")
    print(ask_v2(q))
