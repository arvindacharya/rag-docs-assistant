"""
Runs router_golden.json through the FULL router pipeline (routing +
retrieval + generation, not just routing) and reports two separate
things: did it route to the right source, and -- for whichever source
it picked -- was the resulting answer actually good, scored the same
way as eval/run_eval.py (RAGAS Faithfulness + ContextPrecisionWithoutReference).

This is a different question from eval/run_router_eval.py, which only
tests routing and is cheap (1 call/question). This script is the full
cost (route + generate + 2 judge calls per question) but tells you
whether a routing mistake or a bad retrieval/generation is the actual
cause when something goes wrong end-to-end.

Usage:
    python eval/run_router_quality_eval.py

Reuses the exact ragas/AsyncAnthropic/model_args fixes from run_eval.py
-- see that file's docstring for why each one is necessary. Same
library-version gotchas apply here since it's the same ragas/langgraph
pinned environment.
"""
import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anthropic import AsyncAnthropic
from ragas.llms import llm_factory
from ragas.metrics.collections import ContextPrecisionWithoutReference, Faithfulness

from router_chain import answer_question, shutdown

EVAL_MODEL = os.getenv("RAGAS_EVAL_MODEL", os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5"))


def run_pipeline(golden_path: str):
    with open(golden_path) as f:
        golden = json.load(f)

    raw_results = []
    for item in golden:
        question = item["question"]
        print(f"Answering: {question}")
        raw_results.append(answer_question(question))
    return raw_results, golden


def routing_accuracy(raw_results, golden):
    """Did the router pick the source we expected? Separate from
    whether the resulting answer was any good -- a question can route
    correctly and still get a bad answer, or route incorrectly and
    still stumble into an okay answer. Reported side by side, not
    combined into one number, so you can tell which failure mode
    you're looking at."""
    correct = 0
    misses = []
    for result, item in zip(raw_results, golden):
        expected = item["expected_source"]
        actual = result["source"]
        if actual == expected:
            correct += 1
        else:
            misses.append({"question": result["question"], "expected": expected, "got": actual})
    return correct / len(golden) if golden else 0.0, misses


async def score_one(metric, name, question, answer, contexts):
    try:
        result = await metric.ascore(user_input=question, response=answer, retrieved_contexts=contexts)
        return result.value
    except Exception as e:
        print(f"  [!] {name} scoring failed ({type(e).__name__}) for: {question[:60]}")
        return float("nan")


async def score_all(raw_results, faithfulness_metric, context_precision_metric):
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
                "routed_to": result["source"],
                "answer": result["answer"],
                "sources": result["sources"],
                "faithfulness": faithfulness_value,
                "context_precision_without_reference": precision_value,
            }
        )
        print(
            f"  scored -- routed_to={result['source']:8} faithfulness={faithfulness_value:.2f} "
            f"context_precision={precision_value:.2f}: {result['question'][:50]}"
        )
    return rows


def nan_safe_mean(values):
    clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return sum(clean) / len(clean) if clean else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=os.path.join(os.path.dirname(__file__), "router_golden.json"))
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "router_quality_results.json"))
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY before running the eval.")

    from router_chain import langfuse

    with langfuse.start_as_current_observation(as_type="span", name="router-quality-eval-run") as eval_run_span:
        eval_trace_id = eval_run_span.trace_id

        raw_results, golden = run_pipeline(args.golden)

        route_acc, route_misses = routing_accuracy(raw_results, golden)
        print(f"\nRouting accuracy: {route_acc:.0%}")
        if route_misses:
            print(f"Misrouted {len(route_misses)} question(s):")
            for m in route_misses:
                print(f"  - {m['question']} (expected {m['expected']!r}, got {m['got']!r})")

        evaluator_llm = llm_factory(EVAL_MODEL, provider="anthropic", client=AsyncAnthropic())
        evaluator_llm.model_args.pop("top_p", None)
        evaluator_llm.model_args["max_tokens"] = 4096

        faithfulness_metric = Faithfulness(llm=evaluator_llm)
        context_precision_metric = ContextPrecisionWithoutReference(llm=evaluator_llm)

        print("\nRunning RAGAS metrics on the generated answers...")
        rows = asyncio.run(score_all(raw_results, faithfulness_metric, context_precision_metric))

        summary = {
            "routing_accuracy": route_acc,
            "routing_misses": route_misses,
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

        eval_run_span.update(
            input={"n_questions": len(golden)},
            output={"routing_accuracy": route_acc, **summary["mean_scores"]},
        )

        try:
            pushed = 0
            scores_to_push = {"routing-accuracy": route_acc, **summary["mean_scores"]}
            for metric, value in scores_to_push.items():
                if value is None or math.isnan(value):
                    print(f"  (skipping {metric}: no valid scores to average)")
                    continue
                langfuse.create_score(name=f"router-eval-{metric}", value=value, trace_id=eval_trace_id)
                pushed += 1
            langfuse.flush()
            print(f"Logged {pushed} score(s) to Langfuse, attached to trace {eval_trace_id}.")
        except Exception as e:
            print(f"(Skipped pushing scores to Langfuse: {e})")

    shutdown()


if __name__ == "__main__":
    main()
