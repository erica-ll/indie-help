"""Architecture C2 (tryout, not in active use): a simplified scan step --
one batched call judging all reranked chunks at once with a much smaller
output (just relevant/reason per snippet, no content extraction, no anchor,
no topics) -- followed by a draft step that gets both the verdicts and the
full real chunk text, to cross-check the stated reason against the actual
snippet before relying on it.

Tested and found less reliable than architecture C (src/components/scan.py +
draft.py) on the hardest test cases -- see tests/rubric.txt history / the
Numbers scoreboard for the comparison (C2 clean average ~7.1 vs C's ~8.0).
Kept here for reference, not in active use. Self-contained: does NOT share
retrieval/generation logic with src/components -- only the underlying API
clients and prompts.yaml are shared, via src/clients.py.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from clients import client, collection, bm25_retriever, cohere_client, PROMPTS
import bm25s

SCAN_VERIFY_PROMPT = "scan_verify_prompt_v1"
DRAFT_FROM_CHUNKS_PROMPT = "draft_from_chunks_prompt_v1"
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


def verify_chunks_simple(question, top_ids, chunk_lookup, n_votes=3, vote_temperature=0.5, model="gpt-4o-mini"):
    """Returns a list of {source_file, text, relevant, reason, votes} in the
    same order as top_ids; `text` is always the chunk's real, unmodified
    text from chunk_lookup, never anything the model produced."""
    snippets_block = "\n\n".join(
        f"Snippet {i}:\n{chunk_lookup[doc_id]['text']}"
        for i, doc_id in enumerate(top_ids, start=1)
    )
    system_prompt = PROMPTS[SCAN_VERIFY_PROMPT]
    user_prompt = f"Question: {question}\n\nSnippets:\n{snippets_block}"

    votes_by_id = []
    for _ in range(n_votes):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=vote_temperature if n_votes > 1 else 0,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        try:
            evaluations = json.loads(response.choices[0].message.content).get("evaluations", [])
        except json.JSONDecodeError:
            evaluations = []
        votes_by_id.append({e.get("id"): e for e in evaluations if isinstance(e, dict)})

    from collections import Counter
    results = []
    for i, doc_id in enumerate(top_ids, start=1):
        chunk = chunk_lookup[doc_id]
        votes = [by_id.get(i, {}) for by_id in votes_by_id]
        relevant_votes = [bool(v.get("relevant")) for v in votes]
        relevant = Counter(relevant_votes).most_common(1)[0][0]
        reason = next(
            (v.get("reason") for v in votes if bool(v.get("relevant")) == relevant),
            None,
        )
        results.append({
            "source_file": chunk["source_file"],
            "text": chunk["text"],
            "relevant": relevant,
            "reason": reason,
            "votes": relevant_votes,
        })
    return results


def draft_from_chunks(question, chunk_results):
    relevant = [c for c in chunk_results if c["relevant"]]
    snippets_block = (
        "\n\n---\n\n".join(
            f"[Source: {c['source_file']}]\n"
            f"Earlier verdict: relevant, reason=\"{c['reason']}\"\n"
            f"Full text:\n{c['text']}"
            for c in relevant
        )
        if relevant else "(no relevant snippets)"
    )

    system_prompt = PROMPTS[DRAFT_FROM_CHUNKS_PROMPT]
    user_prompt = f"Developer's question: {question}\n\nSnippets:\n{snippets_block}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=1000,
    )
    return response.choices[0].message.content


def ask_c2(question, top_k=6, candidate_k=30, sub_queries=None):
    if sub_queries is None:
        sub_queries = [question]
    top_ids, chunk_lookup = retrieve_fuse_rerank(question, sub_queries, top_k, candidate_k)
    chunk_results = verify_chunks_simple(question, top_ids, chunk_lookup)
    return draft_from_chunks(question, chunk_results)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else input("Question: ")
    print(ask_c2(q))
