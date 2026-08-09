"""
Runs the golden QA set through the RAG pipeline, scores the results with
RAGAS, and writes a report. Ties scores back to Langfuse where possible.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --golden eval/golden_qa.json --out eval/results.json

Requires ANTHROPIC_API_KEY (used both as the app's LLM and, by default,
as the RAGAS judge LLM -- see EVAL_MODEL below if you want to use a
different/cheaper model to judge than the one that generates answers).

NOTE on ragas API: this uses ragas.metrics.collections (Faithfulness,
ContextPrecisionWithoutReference) scored one sample at a time via
metric.ascore(...), paired with an llm_factory-built LLM. This is
deliberate, not a stylistic choice -- in the pinned ragas==0.3.9, the
classic batch flow (SingleTurnSample + EvaluationDataset + evaluate())
expects an older LLM interface (agenerate_prompt) that neither
llm_factory's InstructorLLM nor LangchainLLMWrapper implement in this
version, and fails with an AttributeError. The collections API is the
one actually built to pair with llm_factory. Verified end-to-end with
a fake InstructorLLM subclass before wiring in the real Anthropic call.

NOTE on the client: uses AsyncAnthropic, not Anthropic. ragas checks
whether the client is actually async-capable (by checking if
`client.chat.completions.create` is a coroutine function) before
allowing `await llm.agenerate(...)`, which is what the collections
metrics call internally. A synchronous `Anthropic()` client fails with
`TypeError: Cannot use agenerate() with a synchronous client` -- but
only once you actually run the scoring step with a real key, not at
construction time, which is why this took a real run to surface.
"""
import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path

# allow `python eval/run_eval.py` to import rag_chain.py from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anthropic import AsyncAnthropic
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecisionWithoutReference, Faithfulness

from rag_chain import answer_question, shutdown

EVAL_MODEL = os.getenv("RAGAS_EVAL_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"))


def run_pipeline(golden_path: str):
    """Answer every golden question through the RAG pipeline. Returns the
    raw results (question/answer/sources/chunks) plus the golden items."""
    with open(golden_path) as f:
        golden = json.load(f)

    raw_results = []
    for item in golden:
        question = item["question"]
        print(f"Answering: {question}")
        raw_results.append(answer_question(question))
    return raw_results, golden


def source_hit_rate(raw_results, golden):
    """Cheap, LLM-free sanity check: did the expected doc file show up
    among the retrieved sources? Good first signal before you even look
    at the RAGAS scores.

    Returns (hit_rate, misses) -- misses is a list of dicts for every
    question where the expected file didn't show up in the retrieved
    sources, so you can actually look at *which* ones failed instead of
    just the aggregate percentage."""
    hits = 0
    misses = []
    for result, item in zip(raw_results, golden):
        hint = item.get("expected_source_hint", "")
        if hint and any(hint in s for s in result["sources"]):
            hits += 1
        elif hint:
            misses.append(
                {
                    "question": result["question"],
                    "expected_source_hint": hint,
                    "actual_sources": result["sources"],
                }
            )
    hit_rate = hits / len(golden) if golden else 0.0
    return hit_rate, misses


async def score_one(metric, name, question, answer, contexts):
    """Score a single question on a single metric, tolerating occasional
    judge-model flakiness (malformed structured output, or -- as seen in
    practice -- a bug in the `instructor` library's own retry-on-
    validation-error path that can make retries fail differently from
    the original error). One bad generation shouldn't crash a 20-question
    run; record NaN for this cell and keep going. nan_safe_mean() already
    excludes NaN from the aggregate, so a handful of failures just show
    up as a smaller effective sample, not a corrupted average."""
    try:
        result = await metric.ascore(user_input=question, response=answer, retrieved_contexts=contexts)
        return result.value
    except Exception as e:
        print(f"  [!] {name} scoring failed ({type(e).__name__}) for: {question[:60]}")
        return float("nan")


async def score_all(raw_results, faithfulness_metric, context_precision_metric):
    """Score every question on both metrics. Returns per-question rows."""
    rows = []
    for result in raw_results:
        contexts = [c["text"] for c in result["chunks"]]
        faithfulness_value = await score_one(
            faithfulness_metric, "faithfulness", result["question"], result["answer"], contexts
        )
        precision_value = await score_one(
            context_precision_metric,
            "context_precision_without_reference",
            result["question"],
            result["answer"],
            contexts,
        )
        rows.append(
            {
                "question": result["question"],
                "answer": result["answer"],
                "sources": result["sources"],
                "faithfulness": faithfulness_value,
                "context_precision_without_reference": precision_value,
            }
        )
        print(
            f"  scored -- faithfulness={faithfulness_value:.2f} "
            f"context_precision={precision_value:.2f}: {result['question'][:60]}"
        )
    return rows


def nan_safe_mean(values):
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(clean) / len(clean) if clean else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=os.path.join(os.path.dirname(__file__), "golden_qa.json"))
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "results.json"))
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY before running the eval.")

    from rag_chain import langfuse

    with langfuse.start_as_current_observation(as_type="span", name="eval-run") as eval_run_span:
        eval_trace_id = eval_run_span.trace_id

        raw_results, golden = run_pipeline(args.golden)

        hit_rate, misses = source_hit_rate(raw_results, golden)
        print(f"\nRetrieval sanity check -- expected source found in top-k: {hit_rate:.0%}")
        if misses:
            print(f"Missed {len(misses)} question(s):")
            for m in misses:
                print(f"  - {m['question']}")
                print(f"      expected hint: {m['expected_source_hint']!r}")
                print(f"      actually retrieved: {m['actual_sources']}")

        evaluator_llm = llm_factory(EVAL_MODEL, provider="anthropic", client=AsyncAnthropic())

        # Anthropic's current model generations reject requests that set both
        # temperature and top_p at once (400: "temperature and top_p cannot
        # both be specified for this model. Please use only one."). ragas's
        # llm_factory defaults to setting both -- drop top_p, keep temperature.
        evaluator_llm.model_args.pop("top_p", None)

        # ragas's default max_tokens (1024) can be too tight for the
        # structured statement-generation output the metrics ask for on
        # longer answers, causing truncated/invalid JSON. Give it headroom.
        evaluator_llm.model_args["max_tokens"] = 4096

        faithfulness_metric = Faithfulness(llm=evaluator_llm)
        context_precision_metric = ContextPrecisionWithoutReference(llm=evaluator_llm)

        print("\nRunning RAGAS metrics (this calls the judge LLM once per sample per metric)...")
        rows = asyncio.run(score_all(raw_results, faithfulness_metric, context_precision_metric))

        summary = {
            "source_hit_rate": hit_rate,
            "source_hit_misses": misses,
            "n_questions": len(golden),
            "mean_scores": {
                "faithfulness": nan_safe_mean([r["faithfulness"] for r in rows]),
                "context_precision_without_reference": nan_safe_mean(
                    [r["context_precision_without_reference"] for r in rows]
                ),
            },
        }

        print("\n=== Summary ===")
        print(json.dumps(summary, indent=2))

        with open(args.out, "w") as f:
            json.dump({"summary": summary, "per_question": rows}, f, indent=2)
        print(f"\nFull results written to {args.out}")

        eval_run_span.update(input={"n_questions": len(golden)}, output=summary["mean_scores"])

        # Best-effort: push scores to Langfuse so they sit next to your traces.
        # NaN scores are skipped -- Langfuse's API rejects non-finite numbers.
        # A trace_id is required -- Langfuse's create_score rejects a score
        # with none of trace_id/session_id/dataset_run_id set (a "standalone"
        # score isn't a valid request), which is exactly why this failed
        # with a generic "Bad request" before this fix.
        try:
            pushed = 0
            for metric, value in summary["mean_scores"].items():
                if value is None or math.isnan(value):
                    print(f"  (skipping {metric}: no valid scores to average)")
                    continue
                langfuse.create_score(name=f"eval-{metric}", value=value, trace_id=eval_trace_id)
                pushed += 1
            langfuse.flush()
            print(f"Logged {pushed} mean score(s) to Langfuse, attached to trace {eval_trace_id}.")
        except Exception as e:
            print(f"(Skipped pushing scores to Langfuse: {e})")

    shutdown()


if __name__ == "__main__":
    main()
