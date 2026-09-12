"""Offline batch evaluation: scores a run_pipeline.py
run-output file (the "===== Q{n} =====" / "Answer:" format from run_batch)
against tests/test_gt.txt using Ragas' AnswerCorrectness and Faithfulness.

Faithfulness needs the retrieved contexts as evidence, which are pulled
straight out of the run-output file's scan-step findings block (every chunk
marked relevant=True).

Standalone: `python evaluate.py [run_output_path]` scores a run-output file against tests/test_gt.txt.

`python evaluate.py --diagnose 3 [input_path]` prints the statement-level breakdown
behind Q3's scores, straight from ragas' internal NLI/classification calls
"""
import re
import ast
import json
import asyncio
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI
from ragas.llms import llm_factory
from ragas.embeddings import embedding_factory
from ragas.metrics.collections import AnswerCorrectness, Faithfulness

TESTS_DIR = Path(__file__).resolve().parent.parent.parent / "tests"


def _parse_numbered(text):
    entries = re.split(r"\n(?=\d+\.\s)", text.strip())
    parsed = {}
    for entry in entries:
        match = re.match(r"(\d+)\.\s*(.*)", entry, re.DOTALL)
        if match:
            idx, body = match.groups()
            parsed[int(idx)] = body.strip()
    return parsed


def _parse_run_output(text):
    blocks = re.split(r"\n(?=={5} Q\d+ ={5})", text.strip())
    parsed = {}
    for block in blocks:
        header = re.match(r"===== Q(\d+) =====\n", block)
        if not header:
            continue
        idx = int(header.group(1))

        qmatch = re.search(r"^Question: (.*)$", block, re.MULTILINE)
        question = qmatch.group(1).strip() if qmatch else ""

        findings_section, _, answer_section = block.partition("Answer:\n")

        contexts = []
        for fmatch in re.finditer(
            r"^\s*\[\d+\] relevant=(True|False).*\n"
            r"\s*topics:.*\n"
            r"\s*content: (.*)$",
            findings_section,
            re.MULTILINE,
        ):
            relevant, content_repr = fmatch.groups()
            if relevant != "True":
                continue
            try:
                content = ast.literal_eval(content_repr.strip())
            except (ValueError, SyntaxError):
                content = content_repr.strip()
            if content:
                contexts.append(content)

        parsed[idx] = {
            "question": question,
            "answer": answer_section.strip(),
            "contexts": contexts,
        }
    return parsed


def load_samples(gt_path, run_output_path):
    references = _parse_numbered(Path(gt_path).read_text())
    runs = _parse_run_output(Path(run_output_path).read_text())

    ids = sorted(set(references) & set(runs))
    missing = (set(references) | set(runs)) - set(ids)
    if missing:
        print(f"Skipping ids missing from one of the two files: {sorted(missing)}")

    return [
        {
            "id": i,
            "question": runs[i]["question"],
            "response": runs[i]["answer"],
            "reference": references[i],
            "retrieved_contexts": runs[i]["contexts"],
        }
        for i in ids
    ]


async def run_eval(
    gt_path,
    run_output_path,
    out_path=None,
    model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
):
    """Scores every aligned sample with AnswerCorrectness (always) and
    Faithfulness (only for samples that have retrieved_contexts -- e.g. the
    scan step found zero relevant chunks). Writes a per-question + averages
    report to out_path if given, and returns the same summary dict."""
    samples = load_samples(gt_path, run_output_path)

    client = AsyncOpenAI()
    # llm_factory defaults to max_tokens=1024 for structured output, which
    # truncates the statement-level classification JSON for longer GT/RAG
    # answers (see ragas/llms/base.py) -- raised to avoid IncompleteOutputException.
    llm = llm_factory(model, client=client, max_tokens=4096)
    embeddings = embedding_factory(
        "openai", model=embedding_model, client=client, interface="modern"
    )

    answer_correctness = AnswerCorrectness(llm=llm, embeddings=embeddings)
    faithfulness = Faithfulness(llm=llm)

    results = []
    for sample in samples:
        ac_result = await answer_correctness.ascore(
            user_input=sample["question"],
            response=sample["response"],
            reference=sample["reference"],
        )
        row = {
            "id": sample["id"],
            "question": sample["question"],
            "answer_correctness": ac_result.value,
            "faithfulness": None,
        }

        if sample["retrieved_contexts"]:
            f_result = await faithfulness.ascore(
                user_input=sample["question"],
                response=sample["response"],
                retrieved_contexts=sample["retrieved_contexts"],
            )
            row["faithfulness"] = f_result.value

        f_display = "n/a" if row["faithfulness"] is None else f"{row['faithfulness']:.3f}"
        print(f"[{sample['id']}] answer_correctness={row['answer_correctness']:.3f} faithfulness={f_display}")
        results.append(row)

    def _avg(key):
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "n": len(results),
        "avg_answer_correctness": _avg("answer_correctness"),
        "avg_faithfulness": _avg("faithfulness"),
        "results": results,
    }

    if out_path:
        Path(out_path).write_text(json.dumps(summary, indent=2))
        print(f"\nWrote eval report to {out_path}")

    ac_avg = summary["avg_answer_correctness"]
    f_avg = summary["avg_faithfulness"]
    print(f"\naverage answer_correctness: {ac_avg:.3f}" if ac_avg is not None else "\naverage answer_correctness: n/a")
    print(f"average faithfulness:       {f_avg:.3f}" if f_avg is not None else "average faithfulness:       n/a (no retrieved_contexts for any sample)")

    return summary


async def diagnose(
    question_id,
    gt_path,
    run_output_path,
    model="gpt-4o-mini",
    embedding_model="text-embedding-3-small",
):
    """Prints the statement-level breakdown behind one question's scores:
    which statements matched/didn't and why, straight from ragas' internal
    NLI/classification calls (AnswerCorrectness._classify_statements and
    Faithfulness._create_verdicts)."""
    samples = {s["id"]: s for s in load_samples(gt_path, run_output_path)}
    if question_id not in samples:
        print(f"Q{question_id} not found in {run_output_path} / {gt_path}")
        return
    sample = samples[question_id]

    client = AsyncOpenAI()
    llm = llm_factory(model, client=client, max_tokens=4096)
    embeddings = embedding_factory("openai", model=embedding_model, client=client, interface="modern")

    ac = AnswerCorrectness(llm=llm, embeddings=embeddings)
    response_statements = await ac._generate_statements(sample["question"], sample["response"])
    reference_statements = await ac._generate_statements(sample["question"], sample["reference"])
    classification = await ac._classify_statements(sample["question"], response_statements, reference_statements)
    factuality = ac._compute_f1_score(classification)
    similarity = await ac._calculate_similarity(sample["response"], sample["reference"])
    weighted = (factuality * ac.weights[0] + similarity * ac.weights[1]) / sum(ac.weights)

    print(f"===== Q{question_id} diagnosis =====")
    print(f"\n--- AnswerCorrectness: factuality_f1={factuality:.3f} similarity={similarity:.3f} weighted={weighted:.3f} ---")
    print(f"\nTP ({len(classification.TP)}) -- response statements supported by the reference:")
    for s in classification.TP:
        print(f"  + {s.statement}\n    reason: {s.reason}")
    print(f"\nFP ({len(classification.FP)}) -- response statements NOT supported by the reference:")
    for s in classification.FP:
        print(f"  - {s.statement}\n    reason: {s.reason}")
    print(f"\nFN ({len(classification.FN)}) -- reference statements missing from the response:")
    for s in classification.FN:
        print(f"  ? {s.statement}\n    reason: {s.reason}")

    if sample["retrieved_contexts"]:
        faithfulness = Faithfulness(llm=llm)
        statements = await faithfulness._create_statements(sample["question"], sample["response"])
        verdicts = await faithfulness._create_verdicts(statements, "\n".join(sample["retrieved_contexts"]))
        score = faithfulness._compute_score(verdicts)
        print(f"\n--- Faithfulness: score={score:.3f} ---\n")
        for s in verdicts.statements:
            mark = "supported    " if s.verdict else "NOT supported"
            print(f"  [{mark}] {s.statement}\n    reason: {s.reason}")
    else:
        print("\n--- Faithfulness: skipped, no retrieved_contexts for this sample ---")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("run_output_path", nargs="?", default=str(TESTS_DIR / "answers_langgraph_ragas.txt"))
    parser.add_argument("--diagnose", type=int, metavar="ID", help="Print the statement-level breakdown for one question instead of running the full batch")
    args = parser.parse_args()

    gt_path = TESTS_DIR / "test_gt.txt"
    run_output_path = Path(args.run_output_path)

    if args.diagnose is not None:
        asyncio.run(diagnose(args.diagnose, gt_path, run_output_path))
    else:
        out_path = run_output_path.with_suffix(".eval.json")
        asyncio.run(run_eval(gt_path, run_output_path, out_path))
