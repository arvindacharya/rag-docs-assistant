"""
Cross-encoder reranking, shared by rag_chain.py (v1) and router_chain.py
(v2). Retrieval pulls back a larger initial candidate set using
Chroma's fast similarity search, then this re-scores those candidates
by having a model read the question and each one together, keeping
only the real top-k to send to Claude.

Why this is a separate, second step instead of just trusting Chroma's
ranking: Chroma's search is a "bi-encoder" -- the question and each
chunk get embedded SEPARATELY into vectors, then compared by distance.
That's fast enough to search thousands of chunks, but the model never
actually reads the question and a specific chunk side by side. A
cross-encoder does read them together and scores that one pair
directly -- far more accurate, but too slow to run over an entire
corpus. The standard pattern (used here): let Chroma cheaply narrow
thousands of chunks down to a modest candidate set, then let the
slower, more accurate cross-encoder pick the real best ones out of
just that candidate set.

Runs locally via sentence-transformers (already a dependency from the
corporate-network embedding-model fix elsewhere in this project) --
no new API key, no extra per-call cost, just more local compute per
question. Verified the CrossEncoder API directly against the installed
sentence-transformers version before writing this:
    CrossEncoder(model_name).predict([(query, passage), ...]) -> scores
Could not verify actual reranking quality in my own sandbox, since
downloading the model needs Hugging Face Hub access, which this
sandbox's network policy blocks (the same limitation noted for the
embedding model elsewhere in this project) -- the model download will
work fine on a normal connection.
"""
import os

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

# Load .env here too, defensively -- don't rely on the importing file
# (rag_chain.py / router_chain.py) having already called load_dotenv()
# before this module's top-level os.getenv() calls run. python-dotenv's
# load_dotenv() is safe to call more than once (it's a no-op if the
# variables are already set), so this can't cause any conflict with
# whatever the importing file does.
load_dotenv()

RERANK_MODEL_NAME = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() not in ("false", "0", "no")

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL_NAME)
    return _reranker


def rerank(question: str, chunks: list, top_k: int) -> list:
    """chunks is a list of dicts with at least a "text" key (the shape
    already used throughout rag_chain.py/router_chain.py). Returns the
    top_k highest-scoring chunks, each with a "rerank_score" key added,
    sorted best-first. If reranking is disabled or there's nothing to
    rerank, just truncates to top_k using whatever order Chroma gave.
    """
    if not RERANK_ENABLED or not chunks:
        return chunks[:top_k]

    model = _get_reranker()
    pairs = [(question, c["text"]) for c in chunks]
    scores = model.predict(pairs)

    scored = list(zip(chunks, scores))
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return [{**chunk, "rerank_score": float(score)} for chunk, score in scored[:top_k]]