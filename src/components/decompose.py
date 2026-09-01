"""Query Decomposition component: break a possibly multi-part, conversational
question into 1-3 atomic, independently-searchable sub-queries.

Standalone: `python decompose.py "some question"` prints the sub_queries.
"""
import sys
import re
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from clients import client, PROMPTS

PROMPT_KEY = "query_decomposition_prompt_v3"


def decompose(question, prompt_key=PROMPT_KEY):
    """Returns a list of 1-3 sub-query strings (see [OUTPUT FORMAT] in the
    prompt: a JSON array of strings). Also accepts a prompt_key override, so
    this same function is what you reach for to compare decomposition
    prompt revisions against each other -- not a separate copy."""
    system_prompt = PROMPTS[prompt_key]
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
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        sub_queries = json.loads(raw)
    except json.JSONDecodeError:
        sub_queries = None

    if not sub_queries or not isinstance(sub_queries, list):
        return [question]
    return sub_queries


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else input("Question: ")
    print(json.dumps(decompose(question), indent=2))
