"""Run only query decomposition (no retrieval/rerank/generation) against a
given prompt version, for comparing decomposition prompt revisions."""
import re
import json
import sys
from pathlib import Path

import yaml
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import TESTS_DIR, PROMPTS_PATH

PROMPT_VERSION = "query_decomposition_prompt_v3"
OUTPUT_PATH = TESTS_DIR / "query_decompositions_v3.json"

client = OpenAI()
with open(PROMPTS_PATH) as f:
    PROMPTS = yaml.safe_load(f)


def decompose_query(question, prompt_version):
    decompose_prompt = PROMPTS[prompt_version]
    user_prompt = f"Question: {question}"
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": decompose_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=1000,
        frequency_penalty=0.4,
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


def parse_prompts(text):
    entries = re.split(r"\n(?=\d+\.\s)", text.strip())
    prompts = {}
    for entry in entries:
        match = re.match(r"(\d+)\.\s*(.*)", entry, re.DOTALL)
        if match:
            idx, question = match.groups()
            prompts[int(idx)] = question.strip()
    return prompts


def main():
    prompts_path = TESTS_DIR / "test_prompts.txt"
    prompts = parse_prompts(prompts_path.read_text())
    ids = sorted(prompts)

    decompositions = {}
    for idx in ids:
        question = prompts[idx]
        print(f"[{idx}] {question}")
        sub_queries = decompose_query(question, PROMPT_VERSION)
        decompositions[idx] = {"question": question, "sub_queries": sub_queries}

    OUTPUT_PATH.write_text(json.dumps(decompositions, indent=2, ensure_ascii=False))
    print(f"\nWrote {len(ids)} decompositions to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
