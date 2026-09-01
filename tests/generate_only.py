"""Run retrieval/rerank/generation only, using a precomputed decomposition
file (e.g. from decompose_only.py) so tuning the generation prompt doesn't
require re-calling the decomposition API for unchanged sub_queries."""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import TESTS_DIR
from query import ask

DECOMPOSITIONS_PATH = TESTS_DIR / "query_decompositions_v3.json"
ANSWERS_PATH = TESTS_DIR / "answers_v3.txt"


def main():
    decompositions = json.loads(DECOMPOSITIONS_PATH.read_text())
    ids = sorted(decompositions, key=int)

    with open(ANSWERS_PATH, "w") as f:
        for i, idx in enumerate(ids):
            entry = decompositions[idx]
            question = entry["question"]
            sub_queries = entry["sub_queries"]
            print(f"[{idx}] {question}")

            answer = ask(question, sub_queries=sub_queries)

            # Write + flush per question so a crash mid-run doesn't lose earlier answers
            f.write(f"===== Q{idx} =====\nQuestion: {question}\nAnswer:\n{answer}\n\n")
            f.flush()

            ##################################################################
            # TEMP: Cohere trial key caps rerank calls at 10/min. Delete this
            # block once on a paid key with a higher rate limit.
            if i < len(ids) - 1:
                time.sleep(6.5)
            ##################################################################

    print(f"\nWrote {len(ids)} answers to {ANSWERS_PATH}")


if __name__ == "__main__":
    main()
