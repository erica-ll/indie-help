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
import time
from pathlib import Path
from unittest import result

sys.path.insert(0, str(Path(__file__).resolve().parent))
from components.decompose import decompose
from components.retrieve import retrieve
from components.scan import scan
from components.draft import draft

from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class PipelineState(TypedDict):
    question: str
    top_k: int
    candidate_k: int
    sub_queries: list[str]
    top_ids: list[str]
    chunk_lookup: dict # what is this
    verified: list[dict]
    answer: str

def decompose_node(state: PipelineState) -> dict:
    return {"sub_queries": decompose(state["question"])}

def retrieve_node(state: PipelineState) -> dict:
    top_ids, chunk_lookup = retrieve(
        state["question"], state["sub_queries"], state["top_k"], state["candidate_k"]
    )
    return {"top_ids": top_ids, "chunk_lookup": chunk_lookup}

def scan_node(state: PipelineState) -> dict:
    return {"verified": scan(state["question"], state["top_ids"], state["chunk_lookup"], n_votes=1, model="gpt-4o")}

def draft_node(state: PipelineState) -> dict:
    return {"answer": draft(state["question"], state["verified"])}


def run_sequential(question, top_k=6, candidate_k=30, sub_queries=None, save_intermediate=False, out_dir=None):
    """Pre-LangGraph plain function orchestration, kept for direct comparison
    against the graph-based run() -- same components, no graph. scan() is
    called with n_votes=1, model="gpt-4o" to match tests/run_architecture_c1.py,
    the script that produced the graded ~8.14 baseline."""
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

    verified = scan(question, top_ids, chunk_lookup, n_votes=1, model="gpt-4o")
    if save_intermediate:
        (out_dir / "03_scan.json").write_text(json.dumps(verified, indent=2))

    answer = draft(question, verified)
    if save_intermediate:
        (out_dir / "04_draft.txt").write_text(answer)

    return {"sub_queries": sub_queries, "top_ids": top_ids, "verified": verified, "answer": answer}

def run(question, top_k=6, candidate_k=30, sub_queries=None, save_intermediate=False, out_dir=None):
    # Add checkpoint later
    graph = StateGraph(PipelineState)
    graph.add_node("decompose", decompose_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("scan", scan_node)
    graph.add_node("draft", draft_node)

    graph.add_edge(START, "decompose")
    graph.add_edge("decompose", "retrieve")
    graph.add_edge("retrieve", "scan")
    graph.add_edge("scan", "draft")
    graph.add_edge("draft", END)

    app = graph.compile()

    result = app.invoke({"question": question, "top_k": top_k, "candidate_k": candidate_k})
    return result


def _parse_prompts(text):
    """Parse the plain '1. question text' / '2. question text' format used
    by tests/test_prompts.txt. End-to-end call."""
    import re
    entries = re.split(r"\n(?=\d+\.\s)", text.strip())
    prompts = {}
    for entry in entries:
        match = re.match(r"(\d+)\.\s*(.*)", entry, re.DOTALL)
        if match:
            idx, question = match.groups()
            prompts[int(idx)] = question.strip()
    return prompts


def run_batch(questions_path, output_path, top_k=6, candidate_k=30, evaluate=False, gt_path=None, eval_out_path=None):
    """Run every question in a plain question-list file (tests/test_prompts.txt
    format) through the full graph end to end, including a live decompose_node call for
    each one. Writes each result to output_path in the same format the earlier function-call-based test
    runner used.

    If evaluate=True, runs the Ragas eval step (components/evaluate.py) as
    the pipeline's last stage once every answer has been written, scoring
    output_path against gt_path (defaults to tests/test_gt.txt next to
    questions_path). Requires the content field above to be untruncated,
    since Faithfulness needs the full chunk text as evidence, not a
    150-char preview."""
    prompts = _parse_prompts(Path(questions_path).read_text())
    ids = sorted(prompts)

    with open(output_path, "w") as f:
        for i, idx in enumerate(ids):
            question = prompts[idx]
            print(f"[{idx}] {question}")

            result = run(question, top_k=top_k, candidate_k=candidate_k)
            verified = result["verified"]
            top_ids = result["top_ids"]
            answer = result["answer"]

            findings_block = "\n".join(
                f"  [{j}] relevant={v['relevant']} | grounding_rejected={v['grounding_rejected']} | source={v['source_file']} | chunk_id={doc_id}\n"
                f"      topics: {v['topics']!r}\n"
                f"      content: {(v['content'] or '')!r}"
                for j, (doc_id, v) in enumerate(zip(top_ids, verified), start=1)
            )

            f.write(
                f"===== Q{idx} =====\n"
                f"Question: {question}\n"
                f"Verified findings (scan step, top {top_k} reranked chunks):\n{findings_block}\n"
                f"Answer:\n{answer}\n\n"
            )
            f.flush()

            ##################################################################
            # TEMP: Cohere trial key caps rerank calls at 10/min. One rerank
            # call per question.
            if i < len(ids) - 1:
                time.sleep(6.5)
            ##################################################################

    print(f"\nWrote {len(ids)} answers to {output_path}")

    if evaluate:
        import asyncio
        from components.evaluate import run_eval

        output_path = Path(output_path)
        gt_path = Path(gt_path) if gt_path else output_path.parent / "test_gt.txt"
        eval_out_path = Path(eval_out_path) if eval_out_path else output_path.with_suffix(".eval.json")
        return asyncio.run(run_eval(gt_path, output_path, eval_out_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?", help="Run a single question directly instead of the batch file")
    parser.add_argument("--evaluate", action="store_true", help="Run the Ragas eval step after the batch")
    args = parser.parse_args()

    if args.question:
        print(run(args.question)["answer"])
    else:
        run_batch(
            Path(__file__).resolve().parent.parent / "tests" / "test_prompts.txt",
            Path(__file__).resolve().parent.parent / "tests" / "answers_langgraph_ragas.txt",
            evaluate=args.evaluate,
        )
