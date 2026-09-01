"""Run architecture C with the simplified scan step (verify_chunks_simple +
draft_from_chunks): one batched call judges all reranked chunks with a
minimal relevant/reason verdict per snippet, then the draft step gets both
that verdict AND each snippet's real text so it can cross-check the reason
against the actual content before using it. Uses a precomputed
decomposition file, same as generate_only.py."""
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import TESTS_DIR
from query import retrieve_fuse_rerank, verify_chunks_simple, draft_from_chunks

DECOMPOSITIONS_PATH = TESTS_DIR / "query_decompositions_v3.json"
ANSWERS_PATH = TESTS_DIR / "answers_c_short_pp.txt"
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
            chunk_results = verify_chunks_simple(question, top_ids, chunk_lookup)
            answer = draft_from_chunks(question, chunk_results)

            findings_block = "\n".join(
                f"  [{j}] relevant={c['relevant']} | votes={c['votes']} | source={c['source_file']} | chunk_id={doc_id}\n"
                f"      reason: {c['reason']!r}"
                for j, (doc_id, c) in enumerate(zip(top_ids, chunk_results), start=1)
            )

            # Write + flush per question so a crash mid-run doesn't lose earlier answers
            f.write(
                f"===== Q{idx} =====\n"
                f"Question: {question}\n"
                f"Verified findings (scan step, top {TOP_K} reranked chunks):\n{findings_block}\n"
                f"Answer:\n{answer}\n\n"
            )
            f.flush()

            ##################################################################
            # TEMP: Cohere trial key caps rerank calls at 10/min. This
            # architecture makes exactly one rerank call per question, same
            # as ask().
            if i < len(ids) - 1:
                time.sleep(6.5)
            ##################################################################

    print(f"\nWrote {len(ids)} answers to {ANSWERS_PATH}")


if __name__ == "__main__":
    main()
