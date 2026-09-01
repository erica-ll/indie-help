"""Run architecture C1 (the elaborate scan step: verify_chunks with anchor
grounding + topics enumeration, then draft_from_findings) with the scan
step's judge swapped to gpt-4o, single-shot (n_votes=1). Uses a precomputed
decomposition file, same as run_architecture_c.py. TOP_K/CANDIDATE_K match
run_architecture_c.py's C2 run so the two are directly comparable."""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import TESTS_DIR
from query import retrieve_fuse_rerank, verify_chunks, draft_from_findings

DECOMPOSITIONS_PATH = TESTS_DIR / "query_decompositions_v3.json"
ANSWERS_PATH = TESTS_DIR / "answers_c1_4o_clean.txt"
TOP_K = 6
CANDIDATE_K = 30


def main():
    decompositions = json.loads(DECOMPOSITIONS_PATH.read_text())
    ids = sorted(decompositions, key=int)

    with open(ANSWERS_PATH, "w") as f:
        for i, idx in enumerate(ids):
            entry = decompositions[idx]
            question = entry["question"]
            sub_queries = entry["sub_queries"]
            print(f"[{idx}] {question}")

            top_ids, chunk_lookup = retrieve_fuse_rerank(question, sub_queries, TOP_K, CANDIDATE_K)
            verified = verify_chunks(question, top_ids, chunk_lookup, n_votes=1, model="gpt-4o")
            answer = draft_from_findings(question, verified)

            findings_block = "\n".join(
                f"  [{j}] relevant={v['relevant']} | grounding_rejected={v['grounding_rejected']} | source={v['source_file']} | chunk_id={doc_id}\n"
                f"      topics: {v['topics']!r}\n"
                f"      content: {(v['content'] or '')[:150]!r}"
                for j, (doc_id, v) in enumerate(zip(top_ids, verified), start=1)
            )

            f.write(
                f"===== Q{idx} =====\n"
                f"Question: {question}\n"
                f"Verified findings (scan step, top {TOP_K} reranked chunks):\n{findings_block}\n"
                f"Answer:\n{answer}\n\n"
            )
            f.flush()

            ##################################################################
            # TEMP: Cohere trial key caps rerank calls at 10/min. One rerank
            # call per question, same as run_architecture_c.py.
            if i < len(ids) - 1:
                time.sleep(6.5)
            ##################################################################

    print(f"\nWrote {len(ids)} answers to {ANSWERS_PATH}")


if __name__ == "__main__":
    main()
