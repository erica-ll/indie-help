"""Write one full test run (answer text, score, reasoning) into the Numbers
scoreboard. Reusable for every future prompt/architecture iteration: pass a
new `version_label` and it appends a fresh 'RAG {label} Result Answer' /
'Score {label}' / 'Reasoning {label}' column triplet, matching the layout
used since v1.0-2.1 (columns 2.2-3.3 dropped the answer-text column; this
restores it going forward). Re-running with a label that already has a
header overwrites that same triplet in place instead of duplicating it.

Grading is NOT automated here. Claude reads each answer against the
Ground Truth column (col 3) and rubric.txt and supplies `results`; this
script only does the mechanical spreadsheet write. numbers_parser resets a
cell's style to a bare default on table.write() unless an explicit style=
object is passed, and reusing a style loaded via doc.styles[name] after a
document reload corrupts it (loses bg_color) -- so every style used below is
captured live from an existing cell in this same session, never refetched
by name after a reload.
"""
import re
from pathlib import Path
from numbers_parser import Document

SCOREBOARD_PATH = "/Users/erica/Desktop/Indie Help/DB/Test Records/RAG_Evaluation_Scoreboard.numbers"


def load_answers_file(path):
    """Parse a test-runner output file (the '===== Q{n} =====' / 'Answer:'
    format used by run_architecture_c.py, run_architecture_b.py, etc.) into
    {question_number: answer_text}. Only pulls the final answer text -- not
    the question or the scan-step findings block -- since that's the one
    piece `write_run` needs and hand-transcribing 14 answers into a Python
    dict is slow and error-prone."""
    text = Path(path).read_text()
    blocks = re.split(r"===== Q(\d+) =====\n", text)[1:]
    answers = {}
    for qnum_str, block in zip(blocks[0::2], blocks[1::2]):
        m = re.search(r"Answer:\n(.*)", block, re.DOTALL)
        if m:
            answers[int(qnum_str)] = m.group(1).strip()
    return answers


def read_ground_truth(scoreboard_path=SCOREBOARD_PATH):
    """Read {question_number: {topic, question, gt}} from the Scorecard
    sheet, for grading a fresh run against. Rows 1..14 = Q1..Q14."""
    doc = Document(scoreboard_path)
    t = doc.sheets["Scorecard"].tables["Table 1"]
    return {
        r: {
            "topic": t.cell(r, 1).value,
            "question": t.cell(r, 2).value,
            "gt": t.cell(r, 3).value,
        }
        for r in range(1, 15)
    }


def write_run(version_label, results, scoreboard_path=SCOREBOARD_PATH):
    """results: {question_number (1..14): (answer_text, score, reasoning)}.
    Returns the computed average score."""
    doc = Document(scoreboard_path)
    t = doc.sheets["Scorecard"].tables["Table 1"]

    header_style = t.cell(0, 34).style
    answer_style = t.cell(1, 4).style
    score_style = t.cell(1, 34).style
    reasoning_style = t.cell(1, 35).style
    avg_style = t.cell(15, 20).style

    answer_header = f"RAG {version_label} Result Answer"
    score_header = f"Score {version_label}"
    reasoning_header = f"Reasoning {version_label}"

    existing = {t.cell(0, c).value: c for c in range(t.num_cols)}
    if answer_header in existing:
        answer_col = existing[answer_header]
    else:
        answer_col = t.num_cols
        t.add_column(3)
        t.write(0, answer_col, answer_header, style=header_style)
        t.write(0, answer_col + 1, score_header, style=header_style)
        t.write(0, answer_col + 2, reasoning_header, style=header_style)
    score_col = answer_col + 1
    reasoning_col = answer_col + 2

    for qnum, (answer_text, score, reasoning) in results.items():
        t.write(qnum, answer_col, answer_text, style=answer_style)
        t.write(qnum, score_col, float(score), style=score_style)
        t.write(qnum, reasoning_col, reasoning, style=reasoning_style)

    avg = sum(s for _, s, _ in results.values()) / len(results)
    t.write(15, score_col, round(avg, 2), style=avg_style)

    doc.save(scoreboard_path)
    return round(avg, 2)
