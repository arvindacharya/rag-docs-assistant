"""
Answers each question two ways -- with no context (Claude relying on
whatever it memorized during training) and through the RAG pipeline
(grounded in the actual current docs) -- and prints them side by side.

This is the artifact that actually answers "why not just ask ChatGPT":
run it and see where the two diverge, especially on the `gotcha: true`
questions in golden_qa.json, which target precise syntax that's easy
for a model to get subtly wrong from memory (e.g. an older, more
commonly-tutorialized pattern instead of the current one).

Usage:
    python eval/compare_baseline.py                # all questions
    python eval/compare_baseline.py --gotcha-only  # just the gotcha set
    python eval/compare_baseline.py --out eval/comparison.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anthropic import Anthropic

from rag_chain import answer_question, shutdown

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

BASELINE_SYSTEM_PROMPT = """You are a helpful assistant answering a question about FastAPI. \
Answer from what you know. Do not say you lack information -- give your best answer."""


def baseline_answer(question: str, client: Anthropic) -> str:
    """No retrieval, no context -- just the model answering from memory.
    This is the 'just ask ChatGPT' baseline."""
    message = client.messages.create(
        model=MODEL,
        max_tokens=400,
        system=BASELINE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )
    return message.content[0].text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default=os.path.join(os.path.dirname(__file__), "golden_qa.json"))
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "comparison.json"))
    parser.add_argument("--gotcha-only", action="store_true", help="Only run the gotcha-tagged questions")
    args = parser.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY before running this.")

    with open(args.golden) as f:
        golden = json.load(f)
    if args.gotcha_only:
        golden = [g for g in golden if g.get("gotcha")]

    client = Anthropic()
    results = []

    for item in golden:
        question = item["question"]
        print("\n" + "=" * 100)
        print(f"Q: {question}")
        if item.get("gotcha_reason"):
            print(f"   (gotcha: {item['gotcha_reason']})")

        base = baseline_answer(question, client)
        rag = answer_question(question)

        print(f"\n--- Baseline (no retrieval) ---\n{base}")
        print(f"\n--- RAG (grounded in current docs) ---\n{rag['answer']}")
        print(f"\n   sources cited by RAG: {rag['sources']}")

        results.append(
            {
                "question": question,
                "gotcha": item.get("gotcha", False),
                "gotcha_reason": item.get("gotcha_reason"),
                "reference_answer": item.get("reference_answer"),
                "baseline_answer": base,
                "rag_answer": rag["answer"],
                "rag_sources": rag["sources"],
            }
        )

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 100)
    print(f"Wrote {len(results)} comparisons to {args.out}")
    print(
        "\nThis script doesn't auto-judge which answer is 'right' -- read through "
        "the pairs yourself, especially the gotcha ones, and look for the baseline "
        "confidently giving an older/wrong pattern while the RAG answer cites the "
        "current doc. That side-by-side is the actual evidence for the project's "
        "problem statement -- screenshot the clearest examples for your README."
    )

    shutdown()


if __name__ == "__main__":
    main()
