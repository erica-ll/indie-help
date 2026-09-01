"""Scan/Verify component (architecture C): one dedicated LLM call that scans
the reranked top chunks against the ORIGINAL question and extracts only
what's genuinely relevant. Chunks are labeled with short random tags (harder
to lose track of than a plain index when several near-duplicate chunks from
the same document sit close together, but much cheaper and more reliable for
a model to echo back exactly than a full chunk_id string).

n_votes/vote_temperature/model: single-shot classification of
entity-collision/analogical-evidence cases has shown real instability. When
n_votes > 1, the scan call is repeated and each chunk's relevant/not verdict
is decided by majority vote (grounding is checked per vote first, so a vote
whose anchor doesn't actually appear in that chunk counts as a "not
relevant" vote). n_votes=1 reproduces plain single-shot, temperature=0
behavior.

Standalone: `python scan.py` runs a scan against retrieve.py's output for a
question passed as an argument.
"""
import sys
import re
import json
import random
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clients import client, PROMPTS

PROMPT_KEY = "scan_verify_prompt_v2"


def _normalize_for_match(text):
    """Lowercase, collapse whitespace, and fold curly quotes to straight
    ones so verbatim-anchor checks aren't defeated by PDF-extraction
    artifacts (smart quotes, odd line breaks)."""
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", text).strip().lower()


def _short_tags(n):
    """Generate n short, distinctive, non-sequential tags (e.g. 'k3q7')."""
    alphabet = "abcdefghjkmnpqrstuvwxyz23456789"  # no 0/o/1/l/i, avoids visual confusion
    return ["".join(random.choices(alphabet, k=4)) for _ in range(n)]


def scan(question, top_ids, chunk_lookup, n_votes=3, vote_temperature=0.5, model="gpt-4o-mini", prompt_key=PROMPT_KEY):
    """Returns a list of {source_file, relevant, content, anchor, topics,
    grounding_rejected, votes} in the same order as top_ids."""
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
            # before they can reach the drafting step.
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


if __name__ == "__main__":
    from retrieve import retrieve
    question = sys.argv[1] if len(sys.argv) > 1 else input("Question: ")
    top_ids, chunk_lookup = retrieve(question, [question], top_k=6, candidate_k=30)
    print(json.dumps(scan(question, top_ids, chunk_lookup, n_votes=1, model="gpt-4o"), indent=2))
