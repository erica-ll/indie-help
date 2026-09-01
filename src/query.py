import os
import re
import json
import time
import random
from collections import Counter
import chromadb
import yaml
import bm25s
import cohere
from openai import OpenAI

from config import CHROMA_DIR, TESTS_DIR, PROMPTS_PATH, BM25_DIR

ACTIVE_PROMPT_VERSION = "system_prompt_v8"
QUERY_DECOMPOSITION_PROMPT = "query_decomposition_prompt_v3"  # v1/v2 were lost to an accidental `git restore .` and could not be recovered (never committed); v3 was the improved, committed-to-memory version and is a safe, strictly-better substitute since decompose_query() is never actually called in current testing (precomputed sub_queries are always passed in)
SUBQUERY_VERIFY_PROMPT = "subquery_verify_prompt_v1"
SUBQUERY_SYNTHESIS_PROMPT = "subquery_synthesis_prompt_v1"
SCAN_VERIFY_PROMPT = "scan_verify_prompt_v1"       # used by verify_chunks_simple (architecture C2)
SCAN_VERIFY_PROMPT_V2 = "scan_verify_prompt_v2"    # used by verify_chunks (architecture C1) -- different output schema (findings/chunk_id/anchor/topics), do not point both at the same constant
DRAFT_FROM_FINDINGS_PROMPT = "draft_from_findings_prompt_v1"
DRAFT_FROM_CHUNKS_PROMPT = "draft_from_chunks_prompt_v1"
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


def retrieve_fuse_rerank(rerank_query, sub_queries, top_k, candidate_k):
    """Shared retrieval helper: for each sub_query, do dense+BM25 retrieval
    into one shared candidate pool, then RRF-fuse everything together, then
    Cohere-rerank the fused pool against rerank_query (the ORIGINAL question,
    not any individual sub_query). Used by both ask() and ask_v3() so their
    retrieval/fusion/rerank behavior is identical -- only what happens to the
    reranked chunks afterward differs between architectures.
    Returns (top_ids, chunk_lookup) where chunk_lookup maps id -> {text, source_file}
    for every candidate considered, not just the reranked top_k."""
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
    """Architecture A: single monolithic call. Retrieval/fusion/rerank shared
    with ask_v3() via retrieve_fuse_rerank; otherwise behaviorally unchanged
    from the original single-query version."""
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


def decompose_query(question):
    """Break a possibly multi-part, conversational question into 1-3 atomic,
    independently-searchable sub-queries (see query_decomposition_prompt_v3's
    [OUTPUT FORMAT]: a JSON array of strings)."""
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
    """Dense+BM25 retrieve, RRF-fuse, and Cohere-rerank for a single query end
    to end. Unlike search_reranked (which reranks a candidate pool assembled
    from OTHER queries), this both builds and reranks the pool against the
    same query throughout.
    Returns (top_ids, chunk_lookup) where chunk_lookup maps id -> {text, source_file}
    for every candidate considered, not just the reranked top_k."""
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
    """Ask a narrow, dedicated LLM call whether THIS sub-query's own retrieved
    evidence actually answers it. Returns {"answered": bool, "content": str|None}."""
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
    """Combine already-verified sub-answers into one final response to the
    ORIGINAL question. Does no new evidence extraction of its own."""
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
    """Architecture B: decompose -> retrieve+rerank+verify each sub-query
    independently -> synthesize the verified sub-answers into one response.
    Unlike ask(), no single LLM call ever has to scan the whole candidate
    pool and judge relevance for the full original question at once."""
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


def _normalize_for_match(text):
    """Lowercase, collapse whitespace, and fold curly quotes to straight
    ones so verbatim-anchor checks aren't defeated by PDF-extraction
    artifacts (smart quotes, odd line breaks)."""
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip().lower()


def _short_tags(n):
    """Generate n short, distinctive, non-sequential tags (e.g. 'k3q7') to
    label chunks with -- harder to lose track of than a plain 0..n-1 index
    when several near-duplicate chunks sit close together, but much cheaper
    and more reliable for a model to echo back exactly than a full chunk_id
    string (that was tried and failed: the model sometimes copied the
    surrounding brackets, invented a plausible-looking but nonexistent id,
    or destabilized the JSON output entirely)."""
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"  # no 0/o/1/l/i, avoids visual confusion
    return ["".join(random.choices(alphabet, k=4)) for _ in range(n)]


def verify_chunks(question, top_ids, chunk_lookup, n_votes=3, vote_temperature=0.5, model="gpt-4o-mini", prompt_key=SCAN_VERIFY_PROMPT_V2):
    """One dedicated LLM call that scans the reranked top chunks against the
    ORIGINAL question and extracts only what's genuinely relevant -- this is
    system_prompt_v8's old Step 1 (Scan) pulled out into its own call,
    separate from drafting. Chunks are labeled with short random tags
    (not a bare positional index), since a distinctive label is harder to
    lose track of than a plain sequence number when several near-duplicate
    chunks from the same document sit close together. Returns a list of
    {source_file, relevant, content, anchor, topics, grounding_rejected,
    votes} in the same order as top_ids.

    n_votes/vote_temperature/model mirror verify_chunks_simple: when
    n_votes > 1, the scan call is repeated and each chunk's relevant/not
    verdict is decided by majority vote (grounding is checked per vote
    first, so a vote whose anchor doesn't actually appear in that chunk
    counts as a "not relevant" vote). Tags are generated once and reused
    across all votes so a given chunk's tag -- and therefore its identity
    across votes -- stays fixed."""
    tags = _short_tags(len(top_ids))
    chunks_block = "\n\n".join(
        f"[{tag}] [Source: {chunk_lookup[doc_id]['source_file']}]\n{chunk_lookup[doc_id]['text']}"
        for tag, doc_id in zip(tags, top_ids)
    )
    system_prompt = PROMPTS[prompt_key]
    user_prompt = f"Question: {question}\n\nChunks:\n{chunks_block}"

    votes_by_tag = []
    for _ in range(n_votes):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=vote_temperature if n_votes > 1 else 0,
            max_tokens=2500,
            response_format={"type": "json_object"},
        )
        try:
            findings = json.loads(response.choices[0].message.content).get("findings", [])
        except json.JSONDecodeError:
            findings = []
        by_tag = {f.get("chunk_id"): f for f in findings if isinstance(f, dict)}

        graded = {}
        for tag, doc_id in zip(tags, top_ids):
            f = by_tag.get(tag, {})
            relevant = bool(f.get("relevant"))
            content = f.get("content")
            anchor = f.get("anchor")

            # Grounding check: the model's claimed anchor must actually
            # appear in THIS chunk's real text, or this vote is discarded
            # regardless of what the model claims -- catches chunk mix-ups
            # (content that genuinely exists, but in a different chunk than
            # claimed) before they can reach the drafting step.
            grounding_rejected = False
            if relevant and (not anchor or _normalize_for_match(anchor) not in _normalize_for_match(chunk_lookup[doc_id]["text"])):
                relevant = False
                content = None
                grounding_rejected = True

            graded[tag] = {
                "relevant": relevant,
                "content": content,
                "anchor": anchor,
                "topics": f.get("topics"),
                "grounding_rejected": grounding_rejected,
            }
        votes_by_tag.append(graded)

    verified = []
    for tag, doc_id in zip(tags, top_ids):
        votes = [v[tag] for v in votes_by_tag]
        relevant_votes = [v["relevant"] for v in votes]
        relevant = Counter(relevant_votes).most_common(1)[0][0]
        winning = next((v for v in votes if v["relevant"] == relevant), votes[0])

        verified.append({
            "source_file": chunk_lookup[doc_id]["source_file"],
            "relevant": relevant,
            "content": winning["content"],
            "anchor": winning["anchor"],
            "topics": winning["topics"],
            "grounding_rejected": winning["grounding_rejected"],
            "votes": relevant_votes,
        })
    return verified


def draft_from_findings(question, verified_chunks):
    """Compose the final answer from chunks already verified by
    verify_chunks -- this call does no extraction or relevance judgment of
    its own, only composition. The [Source: ...] tag is attached here in
    code from the reliably-known source_file, rather than trusting the scan
    step to have embedded a citation correctly inside its extracted text."""
    relevant = [v for v in verified_chunks if v["relevant"] and v["content"]]
    findings_block = (
        "\n\n".join(f"- [Source: {v['source_file']}] {v['content']}" for v in relevant)
        if relevant else "(no relevant findings)"
    )

    system_prompt = PROMPTS[DRAFT_FROM_FINDINGS_PROMPT]
    user_prompt = f"Developer's question: {question}\n\nVerified findings:\n{findings_block}"

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


def verify_chunks_simple(question, top_ids, chunk_lookup, n_votes=3, vote_temperature=0.5, model="gpt-4o-mini"):
    """One batched call judging all reranked chunks at once, like
    verify_chunks, but with a much smaller output: just relevant/reason per
    snippet, numbered by plain position (1..N) -- no content extraction, no
    anchor, no topics. Returns a list of {source_file, text, relevant,
    reason, votes} in the same order as top_ids; `text` is always the
    chunk's real, unmodified text from chunk_lookup, never anything the
    model produced.

    Single-shot classification of these entity-collision/analogical-evidence
    cases has shown real instability (a one-sentence prompt edit flipped a
    known test case from 3/3 correct to 3/3 wrong). When n_votes > 1, the
    scan call is repeated at vote_temperature and each snippet's verdict is
    decided by majority vote across the runs, to smooth over that noise
    instead of chasing it with prompt wording. n_votes=1 reproduces the old
    single-shot, temperature=0 behavior exactly."""
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
    """Compose the final answer from chunks verify_chunks_simple marked
    relevant. Each one is shown with BOTH its earlier verdict (reason) and
    its full real text, so the drafting model can cross-check the stated
    reason against the actual snippet before relying on it, instead of
    trusting the scan step's judgment blindly."""
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


def ask_v3(question, top_k=10, candidate_k=30, sub_queries=None):
    """Architecture C: identical retrieval/fusion/rerank to ask() (one
    shared candidate pool, one rerank against the ORIGINAL question) -- the
    only change is splitting the single monolithic generation call into a
    dedicated scan/verify call followed by a drafting call that only
    compiles already-verified findings. Isolates whether separating
    extraction from drafting (with retrieval untouched) fixes the omission
    pattern found in ask()'s outputs."""
    if sub_queries is None:
        sub_queries = decompose_query(question)

    top_ids, chunk_lookup = retrieve_fuse_rerank(question, sub_queries, top_k, candidate_k)
    verified_chunks = verify_chunks(question, top_ids, chunk_lookup)
    return draft_from_findings(question, verified_chunks)


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
