"""Orchestrates the full architecture-C pipeline: decompose -> retrieve ->
scan -> draft. Each stage is also independently runnable on its own --
see components/decompose.py, retrieve.py, scan.py, draft.py.

With save_intermediate=True, each stage's output is written to disk under
out_dir (01_decompose.json, 02_retrieve.json, 03_scan.json, 04_draft.txt) so
a run can be inspected, diffed against a previous run, or debugged
stage-by-stage without rerunning the whole pipeline.
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from components.decompose import decompose
from components.retrieve import retrieve
from components.scan import scan
from components.draft import draft


def run(question, top_k=6, candidate_k=30, sub_queries=None, save_intermediate=False, out_dir=None):
    if save_intermediate:
        if out_dir is None:
            raise ValueError("out_dir is required when save_intermediate=True")
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    if sub_queries is None:
        sub_queries = decompose(question)
    if save_intermediate:
        (out_dir / "01_decompose.json").write_text(
            json.dumps({"question": question, "sub_queries": sub_queries}, indent=2)
        )

    top_ids, chunk_lookup = retrieve(question, sub_queries, top_k, candidate_k)
    if save_intermediate:
        (out_dir / "02_retrieve.json").write_text(json.dumps({
            "top_ids": top_ids,
            "chunks": {doc_id: chunk_lookup[doc_id] for doc_id in top_ids},
        }, indent=2))

    verified = scan(question, top_ids, chunk_lookup)
    if save_intermediate:
        (out_dir / "03_scan.json").write_text(json.dumps(verified, indent=2))

    answer = draft(question, verified)
    if save_intermediate:
        (out_dir / "04_draft.txt").write_text(answer)

    return answer


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else input("Question: ")
    print(run(q, save_intermediate=True, out_dir="pipeline_run_output"))
