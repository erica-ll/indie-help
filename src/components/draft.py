"""Draft component (architecture C): compose the final answer from chunks
already verified by scan() -- this call does no extraction or relevance
judgment of its own, only composition. The [Source: ...] tag is attached
here in code from the reliably-known source_file, rather than trusting the
scan step to have embedded a citation correctly inside its extracted text.

Standalone: `python draft.py` runs scan.py's output for a question passed as
an argument through drafting.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clients import client, PROMPTS

PROMPT_KEY = "draft_from_findings_prompt_v1"


def draft(question, verified_chunks, prompt_key=PROMPT_KEY):
    relevant = [v for v in verified_chunks if v["relevant"] and v["content"]]
    findings_block = (
        "\n\n".join(f"- [Source: {v['source_file']}] {v['content']}" for v in relevant)
        if relevant else "(no relevant findings)"
    )

    system_prompt = PROMPTS[prompt_key]
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


if __name__ == "__main__":
    from retrieve import retrieve
    from scan import scan
    question = sys.argv[1] if len(sys.argv) > 1 else input("Question: ")
    top_ids, chunk_lookup = retrieve(question, [question], top_k=6, candidate_k=30)
    verified = scan(question, top_ids, chunk_lookup, n_votes=1, model="gpt-4o")
    print(draft(question, verified))
