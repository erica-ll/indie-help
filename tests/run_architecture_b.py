"""Run architecture B (decompose -> per-sub-query retrieve+rerank+verify ->
synthesize) using a precomputed decomposition file, so this doesn't re-pay
for decomposition. Mirrors generate_only.py but calls ask_v2 instead of ask."""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import TESTS_DIR
from query import ask_v2

DECOMPOSITIONS_PATH = TESTS_DIR / "query_decompositions_v3.json"
ANSWERS_PATH = TESTS_DIR / "answers_b.txt"


def main():
    decompositions = json.loads(DECOMPOSITIONS_PATH.read_text())
    ids = sorted(decompositions, key=int)

    with open(ANSWERS_PATH, "w") as f:
        for i, idx in enumerate(ids):
            entry = decompositions[idx]
            question = entry["question"]
            sub_queries = entry["sub_queries"]
            print(f"[{idx}] {question}")

            answer = ask_v2(question, sub_queries=sub_queries)

            # Write + flush per question so a crash mid-run doesn't lose earlier answers
            f.write(f"===== Q{idx} =====\nQuestion: {question}\nAnswer:\n{answer}\n\n")
            f.flush()

            ##################################################################
            # TEMP: Cohere trial key caps rerank calls at 10/min. ask_v2
            # already sleeps between its own per-sub-query rerank calls, but
            # a gap is still needed between questions too.
            if i < len(ids) - 1:
                time.sleep(6.5)
            ##################################################################

    print(f"\nWrote {len(ids)} answers to {ANSWERS_PATH}")


if __name__ == "__main__":
    main()
