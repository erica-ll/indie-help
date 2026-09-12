# Design Log

This is the "why," kept separate from the main [README](../README.md) so
that stays concise. Every section here corresponds to a specific decision
point where more than one approach was actually built and compared, not a
retroactive justification for whatever ended up in `main`.

## Data Sourcing and Scraping Considerations

The original plan was to collect post-mortems and articles from GameDeveloper.com. In Phase 1, I crafted an initial test set using 10 articles I manually downloaded from the site and 1 post-mortem from another individual site. However, after Phase 1 testing, I descovered that GameDeveloper.com has an anti-scraping policy, and
automating collection at volume is exactly what that policy exists to prevent.

I still kept the initial for benchmarking purposes (they contribute to several of the golden-set questions), since no automation involved. What didn't happen was scaling that same collection process up to dozens or
hundreds of articles

The resolution for scale: Although GDC Vault's full talk archive is archived, GDC also runs an **official public YouTube channel**
(`@Gdconf`) where it deliberately publishes a large fraction of its talks for
free. `src/scrape_gdc_transcripts.py` pulls from there via two
narrow, non-adversarial API surfaces:

- `youtube_transcript_api` reads YouTube's own **caption track** data for a
  video （the transcript the platform itself already generated/hosts）.
- `yt_dlp` is used only with `extract_flat=True`, which reads the channel's
  video-listing metadata (titles, ids) and never touches video or audio
  streams.

Both are supported, intended uses of public API surfaces rather than HTML
scraping or downloading gated content, which is why this became the ingestion
path instead of the original plan's manual PDF/Markdown collection.

## Chunking

`text_processor.py` chunks at 650 tokens with 100 tokens of overlap, sized to avoid slicing a single self-contained
description across a chunk boundary.

## Retrieval: From Dense-Only to Hybrid Search

Phase 1 was pure dense retrieval, `text-embedding-3-small` embeddings into
Chroma, cosine similarity. The known failure mode: dense embeddings
often failed in matching exact proper nouns. A query naming a specific game or tool can
surface chunks that are semantically similar in topic but concern a
*different* named entity, because the embedding space doesn't strongly
distinguish "same topic, different entity" from "same topic, same entity."
In the worst cases the correct chunk never made it into the retrieved set at all, so the scan step had
nothing to work with and the system produced a **false refusal** (rubric.txt's
Scale B, band 2) on a question the corpus could actually answer.

Phase 2 added `bm25s` for exact lexical matching, and fused BM25's ranking
with the dense ranking via **Reciprocal Rank Fusion** (`retrieve.py`,
`K_RRF=60`). RRF only needs
rank position from each list, which sidesteps having to calibrate two
unrelated scoring scales against each other. This corresponds to a raise in average score (rubric-based) from 5.43 to 6.86.

## Reranking with a Cross-Encoder

RRF fusion over two coarse rankers still leaves a noisy top-`candidate_k`
(30 by default). `retrieve.py` adds Cohere's `rerank-v4.0-fast` cross-encoder
as a final precision pass over the fused candidates.

One deliberate detail: `rerank_query` is always the **original question**,
even when retrieval ran per sub-query from decomposition, so the final
ordering reflects what the developer actually asked, not just one fragment
of a decomposed question.

## Architecture Comparison: A vs B vs C2 vs C

Four different pipeline shapes were built end-to-end and run against the
same rubric-graded question set (`tests/tryouts/`, `tests/run_architecture_c1.py`):

**A — single monolithic call** (`tryouts/architecture_a.py`). One `gpt-4o`
call, given the full reranked context, both judges relevance *and* drafts
the answer in one shot. Cheapest per question (one rerank call, one
generation call), but conflates two different jobs (deciding what's true,
and writing prose) in a single call, which gives the model more room to
blend evidence across sources or state something the context doesn't
actually support. This is the architecture the `system_prompt_v1`–`v9`
iteration (see below) was mostly written against.

**B — decompose, then verify each sub-query independently**
(`tryouts/architecture_b.py`). Retrieve + rerank + verify runs fully
separately per sub-query, then a synthesis call combines the verified
sub-answers. Cleanly isolates "is this specific sub-question answered," but
costly: N sub-queries means N separate rerank calls, which collides directly
with the 10-calls/min Cohere trial cap on any multi-part question, and N×
the retrieval latency.

**C2 — batched scan, no grounding check** (`tryouts/architecture_c2.py`).
One call judges all reranked chunks at once, but with a minimal output
(`relevant` + short `reason`, no verbatim extraction, no anchor). The draft
step is given both the verdicts and the real chunk text, so it can
cross-check the stated `reason` against what the chunk actually says, but
only once, downstream, at draft time. This made C2 especially prone to
blending chunks with high similarity from the same article. Once the model has
repeated the same title or name across several chunks, it tends to lose
track of which specific one a claim actually came from. Measured average
**~7.1** on the rubric.

**C — scan with anchor grounding + topic enumeration** (current,
`src/components/scan.py` + `draft.py`). The scan step first makes the model
enumerate every distinct topic a chunk covers (`topics`) before judging
relevance, forcing it to actually read the whole chunk instead of stopping
at the first relevant-looking sentence, then extracts a verbatim `anchor`
quote that gets checked, in code, against the chunk's real text before any
`relevant: true` verdict is allowed to stand (aiming to reduce context blending).
Measured average **~8.14 baseline** with the `gpt-4o` (`n_votes=1`) judge. Experiements do not show a significant imporvement for `n_votes>=2`.


## Query Decomposition: v1 to v3

`decompose.py` calls `query_decomposition_prompt_v3`; v1 and v2 are
kept in `prompts.yaml` for reference.

- **v1**: split-if-multi-part, strip project specifics, translate to domain
  terminology. No guard against splitting a single-hop question into
  redundant paraphrases of itself.
- **v2**: added preservation rules: proper nouns/tool names must survive
  verbatim, and a *critical* question ("what's wrong with X") must keep its
  negative framing instead of flattening into a neutral how-to query.
- **v3 (current)**: gates splitting on "does this need more than one
  distinct fact, entity, or topic?" If not, one sub-query only. Since sub-queries are verified independently
  downstream, over-splitting single-hop questions produced contradictory
  verification outcomes for what was really one fact lookup. Also preserves
  "answer-type anchors" (e.g. "the book that describes...") so decomposition
  doesn't generalize away the specific form of answer the developer asked
  for.

## Scan/Verify Prompt Evolution

`scan_verify_prompt_v1` → `v3` in `prompts.yaml`, used by `scan.py`:

- **v1**: per-snippet A/B/C classification (direct match / general-category
  match / irrelevant) with just a boolean and a short reason. No verbatim
  extraction, a downstream draft step would have had to re-read the raw
  chunk itself to actually use the finding.
- **v2**: switched to verbatim extraction (`topics`, `anchor`, `content`),
  and added an explicit rule for what the prompt calls the "Name-Sharing
  Trap" — a chunk about a game literally titled, say, "Crimson Vale" is not
  evidence about the color crimson just because of the shared word. This was
  written in response to an observed collision, not added speculatively.
- **v3 (current)**: added handling for **near-duplicate chunks**. Because
  chunking keeps a shared overlap window between consecutive chunks of the
  same document, several chunks handed to one scan call can look almost
  identical — and the model was observed correctly identifying a relevant
  passage but attaching it to the wrong neighboring `chunk_id`. v3 explicitly
  instructs re-reading each specific chunk's own text before writing its
  `topics`/`content`, rather than relying on an impression carried over from
  a similar-looking neighbor.



## Prompt Iteration

`src/prompts.yaml` keeps every version of the system prompt (`v1`–`v9`) and
the decomposition prompt (`v1`–`v3`) in version control side by side, rather
than overwriting a single prompt in place. The point is bisectability: if a
change regresses behavior, there's a known-working prior version to diff
against.

Rough arc of `system_prompt_v1` → `v9` (the prompt architecture A's single
monolithic call was built and tuned against):

- **v1–v2**: baseline zero-inference + citation rules, then a mandatory
  `<scratchpad>` reasoning block added before the final answer, the theory
  being that forcing explicit extraction before drafting reduces
  cross-source blending.
- **v3–v6**: iterated on how strictly to require an exact named-entity
  match before answering, vs. allowing a "conceptual match" when the
  developer describes a category or their own personal project rather
  than one specific named thing. `v4` added an explicit "user context
  exemption" so the model stops demanding the developer's own project
  details match verbatim text in the corpus.
- **v7–v8**: replaced "extract everything, then classify" with a mandatory
  **per-source-block** scan: go through each `[Source: ...]` chunk one at a
  time and produce exactly one verdict per block. This closed off a failure
  mode where freeform scratchpad reasoning would silently skip or merge
  sources.
- **v9**: deliberately *relaxed* the relevance bar to be more generous for
  matching an underlying mechanism or principle to the developer's
  situation even without exact keyword overlap (its own example: connecting
  "advertising effectiveness" to "eCPM").

This entire prompt line was ultimately superseded not by finding one "best"
system prompt, but by moving relevance judgment out of the drafting call
entirely and into its own verification stage — architecture C, covered
above.


## Golden Set Design: Six Failure Modes

The 27 questions weren't picked at random or generated in bulk. Each one
was written to provoke a specific known failure mode, so a low score points
at a specific cause rather than just "something's wrong." Four modes target
retrieval/reasoning failures; a fifth (Hypothetical Scenario Generalization)
is the mirror image of Entity Trap, testing over-matching instead of
under-matching; a sixth (Rejection/Abstention) targets a different axis
entirely, not whether retrieval or reasoning gets the right answer, but
whether the system correctly says nothing when there's nothing to say.

Question text for all IDs below is in `tests/test_prompts.txt` /
`tests/test_gt.txt` (also on GitHub) — not repeated here.

**Entity Trap** — dense/semantic retrieval is weakest exactly where a
question hinges on one specific proper noun (a person, a named tool, a
specific game), because a semantically similar (but wrong) entity can embed
close enough to the right one to get retrieved instead.
- Q2, Q3, Q16, Q23

**Cross-Contamination** — a query term that's genuinely ambiguous across the
corpus: the same word means different things in two different documents (a
color as psychology vs. a color as a game's literal title/theme; "tone" as
emotional register vs. "tone" as a color/hue term). This is the case a pure
top-K similarity search is most likely to get wrong by blending or
misattributing across sources, and it's the direct motivation for the
anchor-grounding check in `scan.py` (see
[Grounding](#grounding-the-anchor-check)).
- Q8, Q9
- Q12 — the sharpest version of this trap, and also a Rejection/Abstention
  case (below).

**Multi-hop Reasoning** — the answer only exists by combining evidence from
more than one chunk, sometimes more than one document, that a single
retrieval pass has to surface together.
- Q5, Q6, Q7.

**Conflicting Opinions** — the corpus itself contains two sources that
disagree, so the system has to represent the disagreement rather than
silently pick a side, blend both into one vague statement, or dodge the
question with a noncommittal "it depends" instead of using the evidence the
corpus actually has. This is what `draft_from_findings_prompt_v1`'s "strict
source isolation" rule and rubric.txt's penalty for non-committal
answers to decision questions both guard against.
- Q7, Q17.

**Hypothetical Scenario Generalization** — a long, conversational question
describing the developer's *own* project, with no named entity for the
system to match against at all. This is deliberately the mirror image of
Entity Trap: instead of testing whether the system finds a specific named
thing, it tests whether the system over-applies entity-matching when
there's nothing to match, and fails to generalize a principle to the
user's stated situation.
- Q4, Q10, Q14

**Rejection/Abstention** — questions the corpus genuinely can't answer, so
the correct behavior is a refusal rather than falling back on pretrained
knowledge. Scored on `rubric.txt`'s separate Scale B (six fixed values, not
a continuum — see [Rubric Design](#evaluation-rubric-design)), and a refusal
only counts as a real pass once the "unearned-pass" check confirms the
tempting document was actually indexed and retrievable, not just absent
from the run. This specifically is a test on the scanning step.
- Q11, Q12, Q26, Q27.

## Evaluation Rubric Design

`tests/rubric.txt` uses two separate scales, since answerable and abstention
questions fail in different ways.

**Scale A — answerable (continuous 0–10):** banded by completeness, then
deducted independently of the band.
- Fabrication caps at 0 regardless of fluency. A coincidentally-correct
  fabrication scores the same as a wrong one.
- A "decorative citation" (a citation on a claim the cited chunk doesn't
  actually support) caps at 6 even if the claim is true. Severity scales
  with how much of the claim is unsupported, not with how confidently it's
  phrased.

**Scale B — abstention (six fixed values: 0, 2, 5, 8, 9, 10):** not a
continuum, no interpolating. "8 — blunt refusal" vs. "5 — hedged refusal"
isn't a matter of degree. It shows the line between declining correctly and
starting to leak pretrained content behind a disclaimer.

`src/components/evaluate.py` adds automated scoring on top: Ragas'
`AnswerCorrectness` and `Faithfulness`, plus a `--diagnose <question_id>`
mode for the statement-level TP/FP/FN behind one score. Ragas gives a
numeric, repeatable signal that runs unattended, and unlike the custom
rubric, it's an industry-standard metric, giving external benchmarkability
the rubric alone can't.

## Keeping the Repo Clean: What's Gitignored and Why

The resolution in `.gitignore`: exclude all of `tests/` except
`tests/tryouts/` (the architecture variants themselves) and the three files
that define the canonical golden set, `rubric.txt`, `test_prompts.txt`,
`test_gt.txt`. The rubric and golden questions are the artifacts that need
to stay stable and reviewable release over release; any single run's
generated answers are fully reproducible from the current code plus that
same golden set, so they don't need to live in version control to prove the
iteration happened.
