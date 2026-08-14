"""
Turns thumbs-down feedback into DRAFT candidate golden-eval questions
for human review. Deliberately does NOT auto-generate a reference
answer or expected source -- every other golden question in this
project (golden_qa.json, router_golden.json) was verified against the
real fetched docs by hand, and a thumbs-down feedback record has no
guaranteed-correct answer attached to it, only evidence that something
went wrong. Auto-fabricating a "correct answer" here would be a
plausible-sounding guess, which is exactly the failure mode this whole
project is built to catch elsewhere -- so this script's job is triage,
not final curation: collect what went wrong into one reviewable place,
a human checks the real docs and fills in the blanks, then copies
verified entries into golden_qa.json or router_golden.json themselves.

Usage:
    python eval/promote_feedback.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from feedback import load_feedback, mark_promoted

OUT_V1 = Path(__file__).resolve().parent / "golden_qa_candidates.json"
OUT_V2 = Path(__file__).resolve().parent / "router_golden_candidates.json"


def _append_candidates(path: Path, new_items: list):
    existing = json.loads(path.read_text()) if path.exists() else []
    existing.extend(new_items)
    path.write_text(json.dumps(existing, indent=2))


def main():
    records = load_feedback()

    # A person can click thumbs down, then up, then down again on the
    # SAME answer -- each click is a real, separately-logged event (by
    # design, so the raw log stays honest), but only the LATEST rating
    # for a given (app, question, answer) reflects their actual current
    # opinion. Group first, so re-rating the same thing doesn't create
    # duplicate candidates, and a final thumbs-up correctly cancels out
    # an earlier thumbs-down instead of still being flagged as bad.
    groups = defaultdict(list)
    for i, r in enumerate(records):
        if r.get("promoted"):
            continue
        key = (r["app"], r["question"], r["answer"])
        groups[key].append(i)

    if not groups:
        print("No new feedback to process.")
        return

    v1_candidates, v2_candidates = [], []
    promote_indices = []

    for (app, _, _), indices in groups.items():
        promote_indices.extend(indices)  # every instance of this exact (question, answer) is now handled
        latest = records[indices[-1]]  # records are appended in order, so the last index is the most recent
        if latest["rating"] != "down":
            continue  # final opinion wasn't negative -- nothing to flag

        bad_answer_preview = latest["answer"][:200] + ("..." if len(latest["answer"]) > 200 else "")
        candidate = {
            "question": latest["question"],
            "reference_answer": "TODO -- check the real docs and fill this in",
            "note": f"Flagged by user feedback (thumbs down). Answer given at the time: {bad_answer_preview!r}",
        }
        if app == "v1":
            candidate["expected_source_hint"] = "TODO"
            v1_candidates.append(candidate)
        else:
            # Keep the router's own category as a starting guess -- it's
            # what the app actually did, not necessarily what it should
            # have done, so still worth a human double-check.
            candidate["expected_source"] = latest.get("source_category") or "TODO"
            v2_candidates.append(candidate)

    if v1_candidates:
        _append_candidates(OUT_V1, v1_candidates)
        print(f"Wrote {len(v1_candidates)} v1 candidate(s) to {OUT_V1}")
    if v2_candidates:
        _append_candidates(OUT_V2, v2_candidates)
        print(f"Wrote {len(v2_candidates)} v2 candidate(s) to {OUT_V2}")
    if not v1_candidates and not v2_candidates:
        print("Processed feedback, but nothing's final rating was thumbs-down -- no candidates to write.")

    mark_promoted(promote_indices)
    print(
        "\nEach candidate has TODO placeholders for the reference answer "
        "(and expected source, for v1). Check the real docs, fill them in, "
        "then copy verified entries into golden_qa.json / router_golden.json "
        "yourself -- this script only triages, it doesn't curate."
    )


if __name__ == "__main__":
    main()