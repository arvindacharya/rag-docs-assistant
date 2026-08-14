"""
Shared feedback storage for both Streamlit apps (app_streamlit.py, the
v1 single-source app, and app_router_streamlit.py, the v2 router).
Appends one JSON record per rating to a local JSONL file -- no database
needed at this scale, and JSONL means an interrupted write can't
corrupt previously-saved records the way a single shared JSON array
file could.

Paired with eval/promote_feedback.py, which turns thumbs-down records
into draft candidate golden-eval questions for human review.
"""
import json
import os
import time
from pathlib import Path

FEEDBACK_FILE = Path(os.getenv("FEEDBACK_FILE", "eval/feedback.jsonl"))


def save_feedback(app: str, question: str, answer: str, sources: list, rating: str, source_category: str = None):
    """app is "v1" or "v2" (which Streamlit app this came from, so the
    promote script knows which golden-set format to draft a candidate
    for). rating is "up" or "down". source_category is the router's
    "fastapi"/"nodejs"/"coding"/"offtopic" decision for v2, or None for v1."""
    record = {
        "timestamp": time.time(),
        "app": app,
        "question": question,
        "answer": answer,
        "sources": sources,
        "source_category": source_category,
        "rating": rating,
        "promoted": False,
    }
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


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
