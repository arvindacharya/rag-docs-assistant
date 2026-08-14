"""
Shared feedback storage for both Streamlit apps (app_streamlit.py, the
v1 single-source app, and app_router_streamlit.py, the v2 router).
Appends one JSON record per rating to a local JSONL file -- no database
needed at this scale, and JSONL means an interrupted write can't
corrupt previously-saved records the way a single shared JSON array
file could.

Also pushes each rating to Langfuse as a real score attached to the
trace for that specific answer, so real user feedback shows up in
Langfuse dashboards over time -- not just in this local file. Requires
a trace_id, which is why answer_question() in rag_chain.py and
router_chain.py now return one (see the trace_id key in their result
dicts) -- same requirement Langfuse enforces on create_score() that
run_eval.py already had to work around (a score needs exactly one of
trace_id/session_id/dataset_run_id; there's no such thing as a
"standalone" score).

Paired with eval/promote_feedback.py, which turns thumbs-down records
into draft candidate golden-eval questions for human review.
"""
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

FEEDBACK_FILE = Path(os.getenv("FEEDBACK_FILE", "eval/feedback.jsonl"))
langfuse = get_client()


def save_feedback(
    app: str,
    question: str,
    answer: str,
    sources: list,
    rating: str,
    source_category: str = None,
    trace_id: str = None,
):
    """app is "v1" or "v2" (which Streamlit app this came from, so the
    promote script knows which golden-set format to draft a candidate
    for). rating is "up" or "down". source_category is the router's
    "fastapi"/"nodejs"/"coding"/"offtopic" decision for v2, or None for
    v1. trace_id is the Langfuse trace for the specific answer being
    rated -- pass None if unavailable and this just skips the Langfuse
    push (the local JSONL record is still saved either way)."""
    record = {
        "timestamp": time.time(),
        "app": app,
        "question": question,
        "answer": answer,
        "sources": sources,
        "source_category": source_category,
        "rating": rating,
        "trace_id": trace_id,
        "promoted": False,
    }
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    # Best-effort: also push to Langfuse so real user feedback appears in
    # dashboards, not just this local log. A hiccup here should never
    # break the actual feedback-saving experience in the Streamlit app --
    # the local record above already succeeded regardless.
    #
    # score_id is deliberately stable (same trace -> same ID every time),
    # not left to auto-generate. Langfuse's default behavior is to create
    # a brand new score on every call, even if one already exists for
    # that trace with the same name -- rating the same answer down, up,
    # down, up would otherwise leave 4 separate scores on one trace,
    # which corrupts any "average value" widget (0,1,0,1 averages to a
    # meaningless 0.5, hiding that the final opinion was "up"). Reusing
    # the same score_id makes a later call overwrite the earlier one
    # instead, as long as the name and calendar date also match --
    # Langfuse's documented mechanism for this exact case.
    #
    # Two scores are pushed per rating, not one, because they answer two
    # different questions and numeric scores can't do both: "user-feedback"
    # (NUMERIC, 1.0/0.0) powers an average-sentiment-over-time trend line,
    # while "user-feedback-category" (CATEGORICAL, "up"/"down") powers
    # Langfuse's built-in per-category count analytics -- a numeric score
    # only supports averaging, which blends up/down into one blurred
    # number and can't show "how many of each" the way a categorical
    # score's dedicated count-per-category view can.
    if trace_id:
        try:
            langfuse.create_score(
                name="user-feedback",
                value=1.0 if rating == "up" else 0.0,
                trace_id=trace_id,
                score_id=f"{trace_id}-user-feedback",
                data_type="NUMERIC",
                comment=f"app={app}, routed_to={source_category}" if source_category else f"app={app}",
            )
            langfuse.create_score(
                name="user-feedback-category",
                value=rating,  # "up" or "down" as an explicit category label
                trace_id=trace_id,
                score_id=f"{trace_id}-user-feedback-category",
                data_type="CATEGORICAL",
                comment=f"app={app}, routed_to={source_category}" if source_category else f"app={app}",
            )
            langfuse.flush()

        except Exception as e:
            print(f"(Skipped pushing feedback score to Langfuse: {e})")


def load_feedback():
    if not FEEDBACK_FILE.exists():
        return []
    records = []
    with open(FEEDBACK_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def mark_promoted(indices):
    """Rewrite the file with the given record indices flagged as
    promoted, so re-running promote_feedback.py doesn't create
    duplicate candidates from the same feedback record."""
    records = load_feedback()
    for i in indices:
        records[i]["promoted"] = True
    with open(FEEDBACK_FILE, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")