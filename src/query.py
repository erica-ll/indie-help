import os
import re
import time
import chromadb
import yaml
import bm25s
import cohere
from openai import OpenAI

from config import CHROMA_DIR, TESTS_DIR, PROMPTS_PATH, BM25_DIR

ACTIVE_PROMPT_VERSION = "system_prompt_v8"
RERANK_MODEL = "rerank-v4.0-fast"
K_RRF = 60

client = OpenAI()
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma_client.get_collection("postmortems")
bm25_retriever = bm25s.BM25.load(str(BM25_DIR), load_corpus=True)
cohere_client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
with open(PROMPTS_PATH) as f:
    PROMPTS = yaml.safe_load(f)


def reciprocal_rank_fusion(dense_ids, bm25_ids, k: int = K_RRF):
    """Merge two ranked id lists into one score per id: sum(1 / (k + rank))."""
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, doc_id in enumerate(bm25_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)

def search_reranked(question, k, candidate_k, fused_ids, fused_texts):
    """Cross-encoder rerank the top `candidate_k` RRF results down to the final top `k`."""
    candidate_ids = fused_ids[:candidate_k]
    candidate_texts = fused_texts[:candidate_k]

    response = cohere_client.rerank(
        model=RERANK_MODEL,
        query=question,
        documents=candidate_texts,
        top_n=k,
    )
    # response.results is already ordered by relevance; .index points back
    # into candidate_texts/candidate_ids, not into the full fused list
    return [candidate_ids[result.index] for result in response.results]


def ask(question, top_k=10, candidate_k=30):
    # Dense retrieval: ChromaDB returns a wide candidate pool of ids + text/metadata
    query_embedding = client.embeddings.create(
        input=[question], model="text-embedding-3-small"
    ).data[0].embedding
    results_dense = collection.query(query_embeddings=[query_embedding], n_results=candidate_k)
    dense_ids = results_dense["ids"][0]

    # Sparse retrieval: bm25s returns a matching-size candidate pool of corpus dicts
    query_tokens_bm25 = bm25s.tokenize([question], stopwords="en")
    bm25_hits, _ = bm25_retriever.retrieve(query_tokens_bm25, k=candidate_k)
    bm25_hits = bm25_hits[0]
    bm25_ids = [hit["id"] for hit in bm25_hits]

    # RRF only needs the two ranked id lists, not the text. Build a lookup
    # from both result sets so we can go from an id back to its text
    chunk_lookup = {
        id_: {"text": doc, "source_file": meta["source_file"]}
        for id_, doc, meta in zip(
            results_dense["ids"][0], results_dense["documents"][0], results_dense["metadatas"][0]
        )
    }
    for hit in bm25_hits:
        chunk_lookup.setdefault(hit["id"], {"text": hit["text"], "source_file": hit["source_file"]})

    fused = reciprocal_rank_fusion(dense_ids, bm25_ids, k=K_RRF)
    fused_ids = [doc_id for doc_id, _ in fused]
    fused_texts = [chunk_lookup[doc_id]["text"] for doc_id in fused_ids]

    # Cross-encoder rerank: narrow the fused candidate pool down to the final top_k
    top_ids = search_reranked(question, top_k, candidate_k, fused_ids, fused_texts)

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


def parse_prompts(text):
    entries = re.split(r"\n(?=\d+\.\s)", text.strip())
    prompts = {}
    for entry in entries:
        match = re.match(r"(\d+)\.\s*(.*)", entry, re.DOTALL)
        if match:
            idx, question = match.groups()
            prompts[int(idx)] = question.strip()
    return prompts


def run_tests():
    prompts_path = TESTS_DIR / "test_prompts.txt"
    answers_path = TESTS_DIR / "answers.txt"

    prompts = parse_prompts(prompts_path.read_text())
    ids = sorted(prompts)

    with open(answers_path, "w") as f:
        for i, idx in enumerate(ids):
            question = prompts[idx]
            print(f"[{idx}] {question}")
            answer = ask(question)

            # Write + flush per question so a crash mid-run doesn't lose earlier answers
            f.write(f"===== Q{idx} =====\nQuestion: {question}\nAnswer:\n{answer}\n\n")
            f.flush()

            ##################################################################
            # TEMP: Cohere trial key caps rerank calls at 10/min. Delete this
            # block once on a paid key with a higher rate limit.
            if i < len(ids) - 1:
                time.sleep(6.5)
            ##################################################################

    print(f"\nWrote {len(ids)} answers to {answers_path}")


if __name__ == "__main__":
    run_tests()
