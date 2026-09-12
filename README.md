# Indie Help

A LangGraph-orchestrated retrieval-augmented question-answering system that
gives indie game developers grounded advice from industry professionals.
Everything is drawn directly from free GDC talk transcripts and developer
post-mortems with proper citations, no hallucinations.




Ask something like *"How does Hades' meta-progression actually break down?"*
or *"Is there a real example of a game handling arachnophobia accessibility?"*
and get an answer built only from retrieved evidence, with every claim traced
back to a source file. If the knowledge base does not cover this specific topic, it will tell you explicit "not in the knowledge base". We are continuing expanding our knowledge.




Built as a from-scratch exploration of production-grade RAG: hybrid retrieval,
cross-encoder reranking, a dedicated grounding/verification stage, and offline
evaluation against a hand-written 27-question golden set. See
[docs/DESIGN_LOG.md](docs/DESIGN_LOG.md) for the full history of what was
tried, measured, and thrown out along the way.




## Pipeline




```
question
   │
   ▼
┌─────────────┐   1-3 atomic, independently-searchable sub-queries
│  decompose  │   (src/components/decompose.py)
└─────────────┘
   │
   ▼
┌─────────────┐   dense (OpenAI embeddings) + BM25 per sub-query
│  retrieve   │◄──┐  → Reciprocal Rank Fusion → Cohere cross-encoder rerank
└─────────────┘   │  (src/components/retrieve.py)
   │              │
   ▼              │ nothing relevant found → retry once,
┌─────────────┐   │  top_k doubled (capped at candidate_k)
│    scan     │───┘
└─────────────┘   per-chunk relevance + verbatim extraction, with a
   │              hard grounding check against the chunk's real text
   │ found evidence, or already retried    (src/components/scan.py)
   ▼
┌─────────────┐   composes the final answer from verified findings only,
│    draft    │   one [Source: file] citation per claim
└─────────────┘   (src/components/draft.py)
   │
   ▼
  answer + citations
```

`src/run_pipeline.py`'s `run()` wires these into a **LangGraph** `StateGraph`
with a conditional edge: if `scan` comes back with
zero relevant chunks, the graph loops back through `retrieve` once with
`top_k` doubled (capped at `candidate_k`) before falling through to `draft`.
This targets a chunk that made it into the candidate pool but got cut by
the narrow final rerank cutoff before ever reaching `scan` (see
[DESIGN_LOG § Self-Correcting Retrieval Loop](docs/DESIGN_LOG.md#self-correcting-retrieval-loop-a-langgraph-retry-edge)).
A plain-function baseline (`run_sequential()`) is kept alongside it with no
retry, for comparison.

Each component is also independently runnable, e.g. `python
src/components/retrieve.py "some question"`.




## Results




**Rubric-graded single-question testing** (manual grading against
[tests/rubric.txt](tests/rubric.txt), tracked across ~20 prompt/architecture
iterations in a local scoreboard, not all committed — see
[DESIGN_LOG § Architecture Comparison](docs/DESIGN_LOG.md#architecture-comparison-a-vs-b-vs-c2-vs-c)):




| Architecture | Avg. score | Notes |
|---|---|---|
| A — single monolithic judge+draft call | — | superseded early, no hybrid retrieval |
| B — decompose → per-sub-query verify → synthesize | — | one rerank call per sub-query, higher latency |
| C2 — batched relevance-only scan (no extraction/anchor) | ~7.1 | verdicts not cross-checked against real chunk text |
| **C — scan with anchor grounding + topics, gpt-4o judge** | **~8.14** | current architecture (`src/components/`) |




**Offline Ragas evaluation** on the 27-question golden set
(`tests/test_gt.txt`, run via `python src/run_pipeline.py --evaluate`):




| Metric | Average |
|---|---|
| Answer Correctness | 0.677 |
| Faithfulness | 0.818 |




Faithfulness is `null` for 2/27 questions where the scan step found zero
chunks that passed grounding.
// NEED REFINEMENT HERE




## Evaluation




- **Golden set**: 27 hand-written question/answer pairs
(`tests/test_gt.txt` + `tests/test_prompts.txt`), designed around six
specific failure modes rather than picked at random. Full mapping and
reasoning in [DESIGN_LOG § Golden Set Design](docs/DESIGN_LOG.md#golden-set-design-six-failure-modes):
   - **Entity Trap** — a specific proper noun the answer hinges on
     (*"Who is Alex Karpenko?"*), to stress dense retrieval's weak spot on
     exact names.
   - **Cross-Contamination** — a query word that collides with an
     unrelated document (*"How does the color theory in Disco Elysium help
     narrate the story?"* — the corpus has a color-*psychology* article
     that shares the word but not the topic), to stress the anchor check.
   - **Multi-hop Reasoning** — an answer that only exists by combining two
     separate articles (*"Is there a real argument against chasing
     TikTok-style retention mechanics for a story-driven RPG?"*).
   - **Conflicting Opinions** — two sources disagreeing inside the same
     corpus (*"will better leadership actually fix our burnout?"* — one
     speaker says yes, another directly rebuts her in the same talk).
   - **Hypothetical Scenario Generalization** — a rambling, no-named-entity
     question describing the developer's own project (*"my gold economy is
     completely broken... is there an actual tool for this?"*), to stress
     whether the system over-applies entity-matching rules when there's no
     entity to match.
   - **Rejection/Abstention** — a question the corpus genuinely has nothing
     on (*"How does the composer behind Sword of the Sea decide which
     projects to say yes to?"*), where the correct behavior is a plain
     refusal rather than falling back on pretrained knowledge.


- **Rubric** (`tests/rubric.txt`): a 0–10 scale for answerable questions and
 a separate 6-value scale for abstention questions. Fabricated entities/quotes/figures are
 hard-capped at 0 regardless of fluency; a citation attached to a claim the
 cited chunk doesn't actually support is capped at 6 ("decorative
 citation"). See [DESIGN_LOG § Rubric design](docs/DESIGN_LOG.md#evaluation-rubric-design)
 for socring details and why abstention scoring needed its own scale.
- **Automated scoring**: `src/components/evaluate.py` runs Ragas'
 `AnswerCorrectness` and `Faithfulness` against a pipeline run's output, and
 a `--diagnose <question_id>` mode that prints Ragas' internal
 statement-level TP/FP/FN breakdown for one question, useful for figuring
 out *why* a score is low instead of just knowing that it is.




## How to run




```bash
pip install -r requirements.txt
```




Requires a `.env` with `OPENAI_API_KEY` and `COHERE_API_KEY`. `DB_PATH`
(defaults to `../DB` relative to the repo) is where raw documents, chunk
embeddings, the Chroma index, and the BM25 index live (kept outside the repo).




```bash
# 1. Ingest: pull GDC talk transcripts into DB_PATH/raw
python src/scrape_gdc_transcripts.py --max 150




# 2. Chunk + embed every raw document
python src/text_processor.py




# 3. Build the indexes
python src/load_to_chroma.py
python src/load_to_bm25.py




# 4. Ask a question
python src/run_pipeline.py "How does Hades' meta-progression work?"




# 5. Run the full golden set + Ragas evaluation
python src/run_pipeline.py --evaluate
```




## Tech stack




OpenAI (`text-embedding-3-small`, `gpt-4o` / `gpt-4o-mini`) · ChromaDB ·
`bm25s` · Cohere Rerank (`rerank-v4.0-fast`) · LangGraph · Ragas ·
`youtube-transcript-api`




## Repo structure




```
src/
 components/       decompose, retrieve, scan, draft — each independently runnable
 run_pipeline.py   LangGraph + plain-function orchestration, batch runner
 clients.py        shared OpenAI/Chroma/BM25/Cohere clients (constructed once)
 config.py         paths, all driven by DB_PATH
 prompts.yaml       every prompt version, version-controlled
 scrape_gdc_transcripts.py, text_processor.py, load_to_chroma.py, load_to_bm25.py
tests/
 tryouts/          architectures A, B, C2 — kept for comparison, not in active use
 rubric.txt, test_prompts.txt, test_gt.txt   canonical golden set (tracked)
docs/
 DESIGN_LOG.md     full rationale, rejected alternatives, prompt history
```




Most of `tests/` is intentionally untracked (`.gitignore`) — it holds
generated answer dumps and local scoring artifacts from ~20 experiment
iterations. Only the canonical golden set and the tryout architectures are
committed.




## Limitations & known gaps




- Bulk corpus growth is scoped to GDC's official YouTube channel, plus 11
 hand-downloaded GameDeveloper.com articles from initial testing — it misses
 anything only published in GDC Vault's paywalled archive or in other
 text-only post-mortems, a scope decision explained in
 [DESIGN_LOG](docs/DESIGN_LOG.md#data-sourcing-and-scraping-considerations).
- Ragas' automated metrics can't fully verify abstentions: `AnswerCorrectness`
rewards a correct refusal by text similarity to the reference, and
`Faithfulness` can't run with zero grounded chunk. Manual
rubric grading is what actually
confirms a refusal was earned with correct reasoning.
- `scan.py`'s majority-vote judging (`n_votes>1`) exists to address
single-shot instability on entity-collision cases, but was never
benchmarked with a measured score. The ~8.14 baseline came from a stronger
judge model (`gpt-4o` over `gpt-4o-mini`), still single-shot. Majority
voting remains implemented but unvalidated. See
[DESIGN_LOG § Grounding](docs/DESIGN_LOG.md#grounding-the-anchor-check).

