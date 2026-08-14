# Docs RAG Assistant

A retrieval-augmented Q&A assistant, with a chat UI, citation-grounded
answers, LLM observability (Langfuse), and an eval suite (RAGAS + custom
checks). Ships in two versions in the same repo:

- **v1**: a single-source pipeline over FastAPI's documentation.
- **v2**: a LangGraph-based agent that routes each question one of four
  ways -- FastAPI docs, Node.js docs, live web search for other coding
  questions, or a hard decline for anything non-technical -- see
  [v2: Multi-source router agent](#v2-multi-source-router-agent-langgraph)
  below.

Built as a portfolio project. `ingest.py` works on any folder of
markdown files -- swap in your own notes, another library's docs, or
anything else.

## Problem statement

An LLM answers from what it memorized during training. That memory has two
failure modes: it can be **outdated** (things changed since training), and
it can be **confidently wrong** -- it doesn't say "I'm not sure," it guesses
fluently, and you can't tell the difference from the outside. For trivia
that's harmless. For a company deploying AI on its own docs, policies, or
support content, it's a real liability: the AI states the wrong return
policy, or the wrong function signature, with total confidence.

This isn't a hypothetical problem. It's the exact thing companies like
[Decagon](https://decagon.ai) and Sierra (both AI customer-support agent
platforms, together worth well over $10B) are built to solve, and their own
materials describe *grounding* -- answering from retrieved source documents
instead of memory -- as the core fix, alongside citations and hallucination
detection. It's also, more narrowly, an *open* problem for technical docs
specifically: [Quantstruct](https://www.ycombinator.com/companies/quantstruct),
a Y Combinator company, exists because generative tools still hallucinate
precise API/SDK details even with RAG in place. Grounding helps a lot; it
doesn't make correctness free.

This project builds that architecture end-to-end -- retrieval, citation-
enforced generation, and a way to actually measure groundedness instead of
assuming it -- and includes a script (`eval/compare_baseline.py`) that
answers the obvious follow-up question, "why not just ask ChatGPT," with a
side-by-side comparison instead of an argument.

## Architecture (v1)

```
fetch_docs.py  -->  data/docs/*.md  -->  ingest.py  -->  data/chroma_db/
                                                              |
                                                              v
                              api.py (FastAPI) / app_streamlit.py (UI)
                                              |
                                     rag_chain.py (retrieve + generate)
                                              |
                                    Anthropic API + Langfuse tracing
```

- **Retrieval**: markdown files are chunked with a markdown-aware
  chunker (splits on headers and paragraphs, never mid-sentence -- see
  "Corpus and chunking fixes" below for why) and embedded with a local
  `sentence-transformers` MiniLM model, pulled from Hugging Face Hub.
  (Originally used Chroma's default ONNX embedder, which downloads
  from an S3 bucket some corporate networks block -- swapped for this
  reason. No API key needed either way.)
- **Generation**: Claude answers using only the retrieved chunks, and is
  instructed to cite the source file for every claim.
- **UI**: a Streamlit chat interface (`app_streamlit.py`) and a FastAPI
  JSON endpoint (`api.py`) both sit on top of the same `rag_chain.py`
  logic, so you have both a demo-able UI and an API you could put a
  real frontend on later.
- **Observability**: every retrieval + generation call is wrapped in a
  Langfuse span/generation, so you get a trace per question with
  latency, token usage, and the retrieved sources -- inspectable in the
  Langfuse UI.
- **Evals**: `eval/golden_qa.json` is a hand-written set of 20
  questions with expected source files -- 15 general, plus 5 tagged
  `"gotcha": true` that target precise syntax where a model is likely
  to default to an older, more commonly-tutorialized pattern instead
  of the current one (e.g. the deprecated `@app.on_event` style vs.
  the current `lifespan` parameter). `eval/run_eval.py` runs the set
  through the pipeline and scores results with RAGAS (faithfulness,
  context precision) plus a free retrieval sanity check (did the
  expected doc show up in the retrieved sources at all).
- **Baseline comparison**: `eval/compare_baseline.py` answers each
  question two ways -- once with no retrieval (Claude from memory,
  the "just ask ChatGPT" case) and once through the RAG pipeline --
  and writes both side by side.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY and Langfuse keys
```

Langfuse: sign up free at https://cloud.langfuse.com (or self-host --
see their docs) and drop the keys into `.env`. Optionally set
`RAGAS_EVAL_MODEL` to a cheaper model (e.g. Haiku) so the eval judge
doesn't have to be the same model that generates answers.

## Run it

```bash
# 1. Get the example corpus (FastAPI docs)
python fetch_docs.py

# 2. Chunk + embed + build the vector store
python ingest.py

# 3a. Chat UI
streamlit run app_streamlit.py

# 3b. or the API
uvicorn api:app --reload --port 8000
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
     -d '{"question": "How do I add a path parameter?"}'
```

## Run the evals

```bash
python eval/run_eval.py
```

Prints a retrieval hit-rate (with the specific questions that missed,
and what got retrieved instead of the expected file), then RAGAS
faithfulness and context-precision scores per question and averaged,
and writes full detail to `eval/results.json`. Also pushes the mean
scores to Langfuse, attached to a trace for that eval run (see "Known
issues and fixes" for why that requires a specific setup).

Real result from a full run, after all the fixes below: retrieval hit
rate 85% (17/20 -- the 3 misses are discussed below), faithfulness
~0.95-0.97, context precision ~0.77-0.81 (these move slightly run to
run; the judge model isn't perfectly deterministic even at low
temperature -- treat single-run numbers as a range, not an exact
value).

## Run the baseline comparison

```bash
python eval/compare_baseline.py --gotcha-only
```

Prints each gotcha question answered with no retrieval vs. through the
RAG pipeline, and writes both to `eval/comparison.json`. The script
doesn't auto-judge which answer is right -- read the pairs yourself.

**Honest real result**: on an actual run, none of the 5 gotcha questions
showed a clean "baseline wrong, RAG right" split. Claude already knew
current FastAPI conventions on all 5 -- the corpus (a very popular,
heavily-documented library) turned out to be too easy a target for
this specific comparison to bite. The one genuinely interesting result
went the *other* way: on the `lifespan`/`on_event` question, the
baseline (no retrieval) gave a fuller, more correct answer than the
RAG version, because retrieval was failing to find `advanced/events.md`
at the time (see "Corpus and chunking fixes" below) and fed the model
irrelevant chunks instead, causing it to hedge with "I don't have that
information." That's a more useful finding than a clean win would have
been: it shows grounding is only as good as retrieval, with a concrete,
diagnosed example of retrieval letting a correct base model down. Worth
re-running this script after the retrieval fixes below to see whether
that specific pair now looks different.

## What's tested vs. what to verify yourself

I built and tested this end-to-end except for a few things my sandbox
couldn't reach:
- **Embedding model downloads** (from S3 originally, then from Hugging
  Face Hub after the corporate-network fix). Chunking, ID generation,
  and Chroma's add/query mechanics are verified working with a
  stand-in embedder -- the real download works fine on a normal
  internet connection.
- **Actual Claude/RAGAS API calls**, since that needs a real API key.
  Every library's exact import surface and call signature mentioned
  below was checked against the actually-installed versions, not
  written from memory, and the data-wiring in each script was dry-run
  tested with a mocked Claude client. The actual numbers, and which
  gotcha questions diverge, needed your real key to see.
- Every `expected_source_hint` in `golden_qa.json` was checked against
  the real fetched FastAPI docs (file paths and exact wording), not
  guessed from memory.

## Known issues and fixes

Several real bugs surfaced only once this ran against a live API and a
live corpus -- documenting them here since the debugging process is
arguably the most interview-relevant part of this project.

**ragas 0.4.x has a broken import** against the latest
`langchain-community` (`langchain-community` is being sunset and
dropped a submodule ragas 0.4 still imports). This repo pins the last
combo verified to import cleanly: `ragas==0.3.9`.

**`llm_factory` + the classic `ragas.metrics.Faithfulness` + `evaluate()`
flow throws `AttributeError: 'InstructorLLM' object has no attribute
'agenerate_prompt'`.** In ragas 0.3.9 that classic flow expects an
older LLM interface neither `llm_factory`'s `InstructorLLM` nor
`LangchainLLMWrapper` implement -- a real internal inconsistency in
that version. Fix: `run_eval.py` uses `ragas.metrics.collections`
(`Faithfulness`, `ContextPrecisionWithoutReference`) instead, scored
one sample at a time via `await metric.ascore(...)`, the API actually
built to pair with `llm_factory`.

**`TypeError: Cannot use agenerate() with a synchronous client`** --
ragas's collections metrics call the judge LLM via `await
llm.agenerate(...)`, and ragas checks whether the client is actually
async-capable before allowing that. Fix: build the judge with
`AsyncAnthropic()`, not `Anthropic()`.

**`400: temperature and top_p cannot both be specified for this
model`** -- Anthropic's current model generations reject requests that
set both sampling parameters at once, but ragas's `llm_factory`
defaults to setting both. Fix, applied right after constructing the
judge LLM in `run_eval.py`: drop `top_p` from `model_args` and bump
the default `max_tokens` (1024, often too tight for structured output)
to 4096.

**Occasional judge-model scoring crashes** -- the judge model
sometimes returns malformed structured output, and the `instructor`
library's own retry-on-validation-error logic has a bug that can make
every retry fail identically with an unrelated Anthropic API error
(`tool_use ids were found without tool_result blocks`) instead of
recovering. This is upstream flakiness, not a config problem, but it
can crash a run partway through on question 13 as easily as question
1. Fix: `run_eval.py` scores each question/metric pair inside its own
try/except (`score_one()`); a failure is recorded as NaN and the run
continues, with NaN excluded from the mean.

**Langfuse "Bad request" even with valid scores** -- `create_score`
requires exactly one of `trace_id`, `session_id`, or `dataset_run_id`;
a "standalone" score with none of those set is rejected outright. Fix:
`run_eval.py` wraps the whole eval run in one Langfuse span
(`eval-run`), and every question's trace nests under it, so the
aggregate scores attach to that trace's ID. Bonus: your dashboard now
shows one trace per eval run with all 20 questions nested inside it,
instead of 20 scattered top-level traces.

**Corpus and chunking fixes.** Retrieval consistently missed
`advanced/events.md` for the lifespan question across multiple runs.
Investigating turned up two separate, real problems, found by actually
measuring the corpus rather than guessing:

- One file, `release-notes.md` (FastAPI's full changelog), was **45.7%
  of the entire vector store** (1,062 of 2,323 chunks from a single
  file) -- it mentions nearly every feature at least once across years
  of "added X / fixed Y" entries, so it competed for (and often won)
  retrieval slots purely on volume, not relevance. Fix: `ingest.py`
  now excludes 10 non-instructional files (the changelog plus
  contributor lists, external-link roundups, and other meta/community
  pages) from ingestion. Corpus dropped from 2,323 chunks (one file at
  45.7%) to 1,217 (biggest file down to 3.7%).
- Separately, the original fixed-size chunker cut chunks at raw
  character offsets, regularly slicing text open mid-sentence or
  mid-word (verified on real docs -- one chunk literally started as
  `'an" will be important...'`, sliced out of mid-sentence). Fix:
  switched to markdown-aware chunking (splits on headers and
  paragraphs). Verified with a TF-IDF similarity proxy (used because
  this sandbox can't reach the real embedding model): the correct
  chunk's score for the lifespan question rose from 0.283 to 0.345,
  and the correct file went from missing entirely out of the top-4
  retrieved sources to holding 3 of 4 slots.

**Claude Sonnet 5 runs with adaptive thinking on by default**, which
can break code that assumes `message.content[0].text` is always the
answer. Confirmed directly from Anthropic's own migration docs: Sonnet
5 decides per-request whether to reason before answering, and
`max_tokens` is a hard ceiling on *thinking plus the answer combined*.
This first surfaced in the v2 router (see below) with
`AttributeError: 'ThinkingBlock' object has no attribute 'text'` --
a request with only `max_tokens=10` apparently didn't leave enough
room for even brief thinking plus a one-word answer, so the response
came back as only a thinking block, no text at all. Fix, applied
everywhere a Claude response gets read (`rag_chain.py`,
`router_chain.py`, `eval/compare_baseline.py`): a shared
`_extract_text()` helper that searches `message.content` for the
actual text block instead of assuming it's first -- the exact pattern
Anthropic's own docs recommend.

If you bump the ragas version, re-run `python -c "import ragas"` and
re-test on a couple of questions before trusting it -- this kind of
fast-moving-ecosystem breakage is exactly the sort of thing worth
mentioning in an interview.

## v2: Multi-source router agent (LangGraph)

The pipeline above (`rag_chain.py`) is a straight line: retrieve, then
generate. There's no decision in it, which is exactly why it didn't
use LangGraph -- a framework for branching graphs doesn't pay for
itself on something with no branches.

`router_chain.py` adds a genuine fork -- actually two. A question
first goes through a **router** node that decides how to handle it,
one of four ways. This is a second, additive pipeline -- `rag_chain.py`,
`api.py`, and `app_streamlit.py` (v1) are untouched and still work
exactly as before.

The tool is deliberately scoped, not a general-purpose assistant:
FastAPI and Node.js questions are answered from their own docs; other
genuine programming/technical questions (a different language,
framework, or library) fall back to live web search; anything
non-technical gets a fixed decline message instead of an attempt to be
generally helpful.

```
question
   |
   v
[route]  <- one Claude call: "fastapi", "nodejs", "coding", or "offtopic"
   |         (thinking explicitly disabled -- classification needs no reasoning)
   +-- "fastapi"  --> [retrieve_fastapi] --+
   |                                        +--> [generate] --> END
   +-- "nodejs"   --> [retrieve_nodejs]  --+
   |
   +-- "coding"   --> [web_search] -----------------> END
   |
   +-- "offtopic" --> [decline] ---------------------> END
```

The `coding` branch calls Claude with Anthropic's built-in `web_search`
tool -- a server-side tool that runs the search and writes the cited
answer in the same API call, so it skips the shared `generate` node
entirely (there's nothing left for a separate generation step to do).
The `offtopic` branch is even more deliberately minimal: `decline_node`
returns a **fixed string**, making zero Claude calls at all. That's not
a shortcut -- it's the actual design choice. A fixed message is both
the cheapest possible outcome and the most reliable one: it can't drift
into being generally helpful about something the tool is explicitly
scoped not to handle, the way an LLM-generated decline message
theoretically could.

### Setup

```bash
# Get the second corpus (Node.js's official API docs, ~70 files).
# Uses a sparse/blobless git checkout so it doesn't clone the entire
# nodejs/node source repo (which is enormous) just for the docs.
python fetch_nodejs_docs.py

# Ingest it into its own Chroma collection. Reuses the existing
# ingest.py unchanged -- the FastAPI side reuses the "docs" collection
# already built for v1, no need to re-ingest it.
python ingest.py --docs_dir data/nodejs_docs --collection nodejs_docs
```

### Run it

```bash
streamlit run app_router_streamlit.py
```

Try all four paths: a clear FastAPI question, a clear Node.js question,
a coding question about something else entirely (e.g. "how does Rust's
match statement work") to confirm it uses web search instead of
declining, and something non-technical (e.g. "what's the weather" or
just "hey, what's up") to confirm it declines cleanly instead of trying
to be generally helpful. Then try a deliberately vague FastAPI/Node.js
question (see the last two entries in `eval/router_golden.json`) to
check the router isn't just keyword-matching on the word "FastAPI" or
"Node.js" appearing in the question.

**Real bug found this way**: the first version of this router only had
two categories (a doc source, or "web search for literally anything
else"). Asking it "what can you help me with" got routed to web search
with a generic system prompt, and Claude answered by listing its own
general capabilities -- file uploads, code execution, image generation
-- none of which this app actually has. The fix wasn't a bigger prompt
patch; it was a real scope narrowing: four categories instead of two,
with "coding questions elsewhere" and "not a coding question at all"
treated as genuinely different cases, the second one never reaching an
LLM at all. Worth remembering as a general lesson: a fallback path
needs to know what it's a fallback *for*, not just be told "you're a
helpful assistant."

### Test router accuracy

```bash
python eval/run_router_eval.py
```

Runs `eval/router_golden.json` (16 clear-cut questions + 2 deliberately
ambiguous ones) through just the routing step (cheap -- 1 call per
question, no retrieval or generation) and reports accuracy plus which
questions it got wrong.

Real result: **18/18 (100%)**, including both deliberately vague
questions ("streaming data with backpressure" correctly routed to
Node.js, "validate incoming request data with type hints" correctly
routed to FastAPI, neither one naming the framework). That's real
evidence the router is reasoning about topic content, not just
pattern-matching a literal keyword.

**Caveat**: this result was measured before the router had `coding`
and `offtopic` categories -- `eval/router_golden.json` currently only
has `fastapi`/`nodejs`-labeled questions, so it doesn't test the two
newer paths at all. Worth adding a handful of `expected_source:
"coding"` and `"offtopic"` questions to actually measure accuracy on
the boundary that matters most now: telling "a coding question about
something else" apart from "not a coding question at all" (e.g. "what
is an API" is arguably borderline and worth specifically testing).

### Test full answer quality (routing + retrieval + generation)

```bash
python eval/run_router_quality_eval.py
```

`router_golden.json` also carries a `reference_answer` for each
question (verified against the real fetched docs, same standard as
`golden_qa.json`'s), so this runs the *entire* pipeline and reports
two separate things side by side: routing accuracy, and RAGAS
faithfulness/context-precision on the actual generated answer. Kept
separate from the cheap routing-only script on purpose: if something
looks wrong end-to-end, this tells you whether the cause was a bad
routing decision or a bad retrieval/generation on a correctly-routed
question -- two different bugs with two different fixes. Same
fault-tolerant scoring and Langfuse trace-wrapping as `run_eval.py`.

### Dependency note (langgraph version pin)

`requirements.txt` pins `langgraph==0.2.60`, not the current 1.x line.
langgraph 1.x requires `langchain-core>=1.4`, which directly conflicts
with the `langchain-core==0.3.86` the ragas eval stack needs -- both
can't be satisfied in one environment. 0.2.60 is the newest version
whose own requirement (`langchain-core>=0.2.43,<0.4.0`) still overlaps
with that pin. Verified by actually installing both in the same
environment and confirming `import ragas` and
`from langgraph.graph import StateGraph` both succeed together.

### What's tested vs. what to verify yourself

Same honesty note as the rest of this project: I verified the graph's
control flow end-to-end (routing decision -> correct branch -> correct
Chroma collection -> generation, in the right order, exactly once each)
using a mocked Claude client and a mocked embedding function, since my
own sandbox can't reach either the Anthropic API or Hugging Face Hub.
The graph wiring, the `langgraph`/`ragas` dependency compatibility, and
the chunking quality on Node.js's docs format were all checked against
real files and a real installed environment. Actual routing accuracy
and retrieval quality against the real embedding model needed your key
to confirm.

### Debugging story: a header-detachment chunking bug, and an honest miss

First real quality run: routing 100%, faithfulness 0.948, context
precision **0.671** -- noticeably lower than the FastAPI-only
pipeline's ~0.77-0.81. Three questions scored worst: crypto hashing
(0.25), CLI argument parsing (0.33), and streaming backpressure (0.00).

Investigating each one directly against the real files (not guessed)
turned up three different causes, not one:

- **A real, fixable bug**: `util.parseArgs()`'s documentation was
  getting shredded by the chunker. The first chunk for that section had
  the header but no real explanation (just a version-changelog block);
  every chunk after it had the real explanation but had lost the header
  entirely -- so a chunk explaining "the parsed command line arguments"
  never actually contained the word `parseArgs` anywhere in its own
  text, making it hard for retrieval to connect the two. Root cause:
  the chunker only re-attached a section's header to the *first*
  sub-chunk when a long section needed further splitting. Fixed in
  `ingest.py` (this fix lives in the shared chunker, so it applies to
  the FastAPI corpus too -- regression-checked there with no change in
  outcome, 0.345 -> 0.344 similarity, same 3-of-4 result). Verified
  directly against the real file that every fragment of that section
  now contains the header text.
- **Not a bug**: crypto hashing's low score had no findable retrieval
  cause -- the four retrieved chunks were already correctly all about
  the `Hash` class.
- **Not a bug, a corpus characteristic**: backpressure is a concept
  Node's own docs discuss inside several *other* differently-titled
  sections, with only one file (`stream_iter.md`) having a section
  literally titled "Backpressure". Retrieval favoring that dedicated
  section over scattered incidental mentions elsewhere is arguably the
  more correct behavior, not a failure.

Result after the fix and a full re-run: faithfulness 0.948 -> 0.976,
context precision 0.671 -> 0.727. Real improvement -- but an honest one,
not the clean story predicted going in. The parseArgs question
specifically targeted actually got *worse* (0.33 -> 0.25), most likely
because `parseArgs` (camelCase) and "parse command line arguments"
(plain English) don't match well under simple text-similarity methods,
and the real embedding model may share some of that limitation. The
actual gains showed up elsewhere -- crypto hashing jumped from 0.25 to
1.00, almost certainly a side effect of the same header fix helping
disambiguate `crypto.md`'s many sibling functions (`createHash`,
`createHmac`, `createCipheriv`, etc.), not something targeted directly.
Backpressure stayed at 0.00, exactly as expected once diagnosed as a
corpus characteristic rather than a bug.

Worth stating plainly: the fix was net-positive and the root cause was
real and verified, but the specific prediction of *which* question
would improve was wrong. That's a more honest -- and more interesting
-- outcome than a story where the fix works exactly as predicted:
measuring the aggregate mattered more here than trusting the
hypothesis about any single question.

### Feedback loop: thumbs up/down -> candidate golden questions

Both Streamlit apps (`app_streamlit.py` and `app_router_streamlit.py`)
show a thumbs up/down control under every answer, using Streamlit's
built-in `st.feedback("thumbs")` widget. Every rating gets appended to
`eval/feedback.jsonl` via the shared `feedback.py` module -- one JSON
record per click, including which app it came from and (for the
router) which category the question was routed to.

```bash
streamlit run app_router_streamlit.py   # or app_streamlit.py
# ask questions, click thumbs up/down under the answers
python eval/promote_feedback.py         # turn thumbs-down cases into review candidates
```

`promote_feedback.py` reads the log and writes draft candidates to
`eval/golden_qa_candidates.json` (from `app_streamlit.py` feedback) or
`eval/router_golden_candidates.json` (from `app_router_streamlit.py`
feedback). It deliberately never writes a real `reference_answer` --
just a `"TODO"` placeholder -- since every other golden question in
this project was checked against the real docs by hand, and
fabricating a "correct answer" from a bad interaction would just be a
plausible-sounding guess. The script's job is triage: collect what
went wrong in one place, a human checks the real docs and fills in the
blanks, then copies verified entries into `golden_qa.json` /
`router_golden.json` themselves. That's exactly how the two
`"offtopic"` and `"fastapi"` questions in the current `router_golden.json`
got there -- see the debugging story below.

A person can click thumbs down, then up, then down again on the exact
same answer -- each click is logged as a genuine, separate event (the
raw log stays honest), but `promote_feedback.py` groups records by
`(app, question, answer)` and only trusts the *latest* rating, so
changing your mind doesn't create duplicate candidates or leave a
stale negative rating flagged after you've decided the answer was
actually fine.

### The feedback loop found a bug in itself, and two real issues in the app

`promote_feedback.py`'s first version had a real bug, found immediately
on real usage rather than caught by my own tests: a person can click
thumbs down, then up, then down again on the exact same answer -- each
click is logged as a separate, honest event by design -- but the
original script treated every historical down-vote as a fresh issue.
Rating something down twice (with an up in between) created two
duplicate candidates for one real issue; a down followed by a final up
(changing your mind to "actually this is fine") would still have
wrongly flagged it as bad. Fixed by grouping records per (app,
question, answer) and trusting only the *latest* rating -- verified
against both the reported real sequence and the trickier down-then-up
case before shipping the fix.

Once that was fixed, real feedback surfaced two genuine, different
findings in one batch:

- **A real, verified retrieval miss**: "how easy is fastapi" got an
  answer stitching together unrelated fragments (deployment being
  "relatively easy," testing being "very easy") instead of retrieving
  `features.md` -- the actual dedicated overview page that directly
  answers this. Added as a new golden question with a reference answer
  checked against that real file.
- **A structural limitation, not a bug**: a question explicitly
  comparing FastAPI and Node.js configuration can't be answered well no
  matter how good retrieval gets, because the router can only pick one
  collection to search. The tool correctly said "I don't know" instead
  of guessing, but there's no way to write a correct `expected_source`
  for a question that genuinely needs both sources -- documented as a
  known limitation in "Deepen this later" instead of forced into the
  eval set.
- One thumbs-down turned out to be feedback on the tool's *scope*, not
  a bug: "who is the king of kabul?" was correctly declined, and the
  down-vote most likely reflected disagreement with that scope rather
  than an actual error. Kept as a golden question anyway -- as
  confirmation the decline is *correct* behavior, so a future change
  can't accidentally break it.

## Deepen this later

- **Add hybrid search (BM25 + embeddings)** to fix a real, well-evidenced
  retrieval failure: "How do I add a path parameter with a type in
  FastAPI?" has scored `context_precision=0.00` in *every* run of this
  project, before and after every other fix, including reranking.
  Measured why: 11 different files mention "path parameter" somewhere,
  because the word "parameter" is used constantly across unrelated
  FastAPI topics (path, query, body, response parameters all share the
  word). A TF-IDF proxy check showed the correct file
  (`tutorial/path-params.md`) does appear in the candidate pool, but
  buried among chunks from `body.md`, `query-params-str-validations.md`,
  and other files that only share generic vocabulary, not actual
  relevance. Adding the cross-encoder reranker (see below) was expected
  to fix exactly this kind of ambiguity -- it didn't; context precision
  stayed at 0.00 with reranking on. Embeddings compare *meaning*, which
  blurs together concepts that share a lot of vocabulary; BM25 (classic
  keyword/phrase-frequency search, no neural network involved) would
  reward an exact phrase match like "path parameter" much more heavily
  than generic embedding similarity does. Combining both -- run Chroma's
  embedding search and a BM25 keyword search in parallel, blend the two
  rankings -- is the standard fix for this specific failure mode, and a
  different kind of change from anything else fixed in this project so
  far (all previous retrieval fixes were about corpus content or
  chunking; this one is about the search method itself).
- Try query rewriting for the same underlying reason: expanding "path
  parameter" into something that disambiguates from "query parameter"/
  "body parameter" before it ever hits retrieval, rather than fixing it
  after the fact. Also relevant to a different, earlier finding: camelCase
  identifiers like `parseArgs` don't match well against plain-English
  phrasing like "parse arguments" -- the one case in this project where
  a real, verified fix didn't produce the predicted improvement.
- The router assumes every question belongs to exactly one category,
  but real usage surfaced a genuine case that breaks that assumption:
  a question explicitly comparing FastAPI and Node.js ("how do configs
  work in Node.js and how are they different to FastAPI") can only be
  routed to one collection, so the other source never gets searched at
  all. The router correctly said "I don't know" rather than making
  something up, but it's structurally unable to succeed here no matter
  how good retrieval gets. Not added as a golden question -- there's no
  single correct `expected_source` for it -- but worth designing for:
  either a router category that triggers retrieval from both
  collections for compare-style questions, or a LangGraph fan-out/
  fan-in pattern (query both sources in parallel, merge before
  generating).
- `eval/router_golden.json` still has zero `"coding"`-category
  questions, and only one `"offtopic"` one (added from real feedback,
  see the debugging story below) -- worth adding a few more of each so
  the 100% accuracy number actually covers all four categories, not
  just the original two-way fastapi/nodejs split.
- Look further into the crypto-module-style precision cost: dense,
  tightly related sibling functions (many `crypto.*` methods) seem to
  cost context precision even when the right file is retrieved.
  Reranking has now been tried and the result was mixed for this
  specific question (context precision 0.83 -> 0.75, slightly worse) --
  worth investigating per-file result diversity instead, or a
  reranker model actually tuned for code/API documentation rather than
  the general-purpose one currently used.
- Add a GitHub Actions workflow that runs `eval/run_eval.py` on every
  PR and fails the build if faithfulness drops below a threshold
- Deploy: FastAPI backend on Fly.io/Render, a proper frontend on
  Vercel, Langfuse dashboards for cost/latency/eval trends over time
