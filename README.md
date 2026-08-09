# Docs RAG Assistant

A retrieval-augmented Q&A assistant over the FastAPI documentation, with a chat
UI, citation-grounded answers, LLM observability (Langfuse), and an
eval suite (RAGAS + a retrieval sanity check).

Built as a portfolio project. The corpus is FastAPI's docs by default, but
`ingest.py` works on any folder of markdown files -- swap in your own notes,
another library's docs, or anything else.

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

## Architecture

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

- **Retrieval**: markdown files are chunked (naive fixed-size chunking,
  see `ingest.py`) and embedded with Chroma's built-in local ONNX
  MiniLM model -- no API key or GPU needed for embeddings.
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
  and writes both side by side. Run it with `--gotcha-only` to focus
  on the 5 questions most likely to actually diverge. This is the
  concrete evidence for the problem statement above, not just an
  assertion that grounding helps.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in ANTHROPIC_API_KEY and Langfuse keys
```

Langfuse: sign up free at https://cloud.langfuse.com (or self-host --
see their docs) and drop the keys into `.env`.

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

This prints a retrieval hit-rate plus RAGAS faithfulness and context
precision scores, and writes per-question detail to `eval/results.json`.
It also tries to push the mean scores to Langfuse as scores on your
traces (best-effort -- see the comment in `run_eval.py`).

## Run the baseline comparison

```bash
python eval/compare_baseline.py --gotcha-only
```

Prints each gotcha question answered with no retrieval vs. through the
RAG pipeline, and writes both to `eval/comparison.json`. Read through
the pairs yourself -- the script doesn't auto-judge which answer is
right. Look for the baseline confidently giving the older/wrong
pattern while the RAG answer cites the current doc; that pair is your
best evidence for the README and for the interview.

## What's tested vs. what to verify yourself

I built and tested this end-to-end except for two things my sandbox
couldn't reach:
- **The embedding model download** (Chroma downloads its ONNX MiniLM
  model from S3 on first use). Chunking, ID generation, and Chroma's
  add/query mechanics are verified working with a stand-in embedder --
  the real download will work fine on a normal internet connection.
- **Actual Claude/RAGAS API calls**, since that needs a real API key.
  The eval script's exact import surface (`SingleTurnSample`,
  `EvaluationDataset`, `llm_factory`, `Faithfulness`,
  `LLMContextPrecisionWithoutReference`) and the Langfuse
  `start_as_current_observation` / `.update()` / `create_score` calls
  are all verified against the actually-installed library versions,
  not just written from memory. `compare_baseline.py`'s data wiring
  (arg parsing, `--gotcha-only` filtering, output JSON shape) was
  dry-run tested with a mocked Claude client -- the actual baseline
  vs. RAG divergence still needs a real key to see. I can't predict
  which gotcha questions will actually diverge without running it;
  the `gotcha_reason` field on each question is a hypothesis to test,
  not a verified result.
- Every `expected_source_hint` in `golden_qa.json`, including the 5
  gotcha ones, was checked against the real fetched FastAPI docs
  (file paths and exact wording), not guessed from memory.

One thing worth knowing going in: **ragas 0.4.x currently has a broken
import against the latest langchain-community** (langchain-community
is being sunset and dropped a submodule ragas 0.4 still imports). This
repo pins the last combo I could verify actually imports cleanly
(`ragas==0.3.9`).

A second, subtler issue showed up during real testing (not caught in
my own sandbox, since it only surfaces with a real API key): even with
ragas importing fine, pairing `llm_factory` (the current recommended
way to set up the judge LLM) with the classic `ragas.metrics.Faithfulness`
+ `evaluate()` flow throws `AttributeError: 'InstructorLLM' object has
no attribute 'agenerate_prompt'`. In ragas 0.3.9, that classic flow
expects an older LLM interface that neither `llm_factory`'s
`InstructorLLM` nor `LangchainLLMWrapper` implement anymore -- it's a
genuine internal inconsistency in that ragas version, not a config
mistake. The fix, already applied in `run_eval.py`: use
`ragas.metrics.collections.Faithfulness` / `ContextPrecisionWithoutReference`
instead, scored one sample at a time via `await metric.ascore(...)`,
which is the API actually built to pair with `llm_factory`. Verified
end-to-end with a fake LLM before wiring in the real Anthropic call --
see the comment at the top of `run_eval.py`.

If you bump the ragas version, re-run `python -c "import ragas"` and
re-test the eval script on a couple of questions before trusting it --
this kind of fast-moving-ecosystem breakage is exactly the sort of
thing worth mentioning in an interview.

Two more issues only showed up on a real run with a real key (neither
is catchable without one):

- `TypeError: Cannot use agenerate() with a synchronous client` --
  ragas's collections metrics call the judge LLM via `await
  llm.agenerate(...)`, and ragas checks whether the client is actually
  async-capable before allowing that. Fix: build the judge with
  `AsyncAnthropic()`, not `Anthropic()`.
- `400: temperature and top_p cannot both be specified for this model`
  -- Anthropic's current model generations reject requests that set
  both sampling parameters at once, but ragas's `llm_factory` defaults
  to setting both. Fix, applied right after constructing the judge LLM
  in `run_eval.py`: drop `top_p` from `model_args` and bump the default
  `max_tokens` (1024, often too tight for structured output) to 4096.
- Occasionally the judge model returns malformed structured output
  (e.g. missing a required field), and the `instructor` library's
  built-in retry-on-validation-error logic has its own bug that can
  make every retry attempt fail identically with an unrelated Anthropic
  API error (`tool_use ids were found without tool_result blocks`)
  instead of actually recovering. This is upstream flakiness, not
  something wrong with your setup -- but it can crash a 20-question
  run partway through on question 13 just as easily as question 1.
  Fix: `run_eval.py` now scores each question/metric pair inside its
  own try/except (`score_one()`); a failure is recorded as NaN for
  that cell and the run continues. NaN is already excluded from the
  mean, so a few flaky cells just mean a slightly smaller effective
  sample, not a crashed run or a corrupted average.
- The "API errors occurred: Bad request" from Langfuse at the end of
  the run, even with valid (non-NaN) scores: Langfuse's `create_score`
  requires exactly one of `trace_id`, `session_id`, or `dataset_run_id`
  -- a "standalone" score with none of those set is rejected outright.
  The original code called `create_score(name=..., value=...)` with
  none of them set. Fix: `run_eval.py` now wraps the whole eval run in
  one Langfuse span (`eval-run`), and every question's individual trace
  nests under it, so the aggregate scores get attached to that trace's
  ID. Bonus: your Langfuse dashboard now shows one trace per eval run
  with all 20 questions nested inside it, instead of 20 scattered
  top-level traces.

## Deepen this later

- Swap fixed-size chunking for markdown-aware or semantic chunking,
  compare RAGAS scores before/after
- Add a reranker (cross-encoder or hosted API) between retrieval and
  generation
- Turn it into an agent: add tool use (web search, multi-hop retrieval)
  via function calling or LangGraph
- Add a GitHub Actions workflow that runs `eval/run_eval.py` on every
  PR and fails the build if faithfulness drops below a threshold
- Add a thumbs up/down feedback control in the Streamlit UI, and a
  script that turns thumbs-down cases into new golden eval questions
- Deploy: FastAPI backend on Fly.io/Render, a proper frontend on
  Vercel, Langfuse dashboards for cost/latency/eval trends over time
