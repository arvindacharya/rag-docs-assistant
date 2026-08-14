"""
Tests whether route_question() picks the correct source (fastapi vs
nodejs) on a golden set of questions -- including a couple of
deliberately vague ones that don't name the framework/runtime
explicitly, to check the router isn't just keyword-matching.

Usage:
    python eval/run_router_eval.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from router_chain import route_question, shutdown


def main():
    golden_path = os.path.join(os.path.dirname(__file__), "router_golden.json")
    with open(golden_path) as f:
        golden = json.load(f)

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY before running this.")

    correct = 0
    misses = []
    for item in golden:
        question = item["question"]
        expected = item["expected_source"]
        actual = route_question(question)
        hit = actual == expected
        correct += hit
        marker = "OK  " if hit else "MISS"
        print(f"[{marker}] expected={expected:8} got={actual:8} {question}")
        if not hit:
            misses.append({"question": question, "expected": expected, "got": actual, "note": item.get("note")})

    accuracy = correct / len(golden)
    print(f"\nRouter accuracy: {correct}/{len(golden)} ({accuracy:.0%})")

    if misses:
        print("\nMissed:")
        for m in misses:
            print(f"  - {m['question']}")
            print(f"      expected {m['expected']!r}, got {m['got']!r}" + (f"  ({m['note']})" if m["note"] else ""))

    shutdown()


if __name__ == "__main__":
    main()
