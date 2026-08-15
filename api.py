"""
FastAPI backend. Run with:
    uvicorn api:app --reload --port 8000

Two endpoints, two separate pipelines:
    /chat        -- v1, single-source (FastAPI docs only), via rag_chain.py
    /router-chat -- v2, four-way router (FastAPI docs, Node.js docs,
                    web search for other coding questions, or a decline
                    for anything non-technical), via router_chain.py

Set ENABLE_ROUTER_CHAT=false to skip importing router_chain.py (and
therefore /router-chat) entirely -- not just its data. This is a real
diagnostic tool, not just a feature flag: even with an empty/small
Node.js collection, importing router_chain.py still constructs a
second Chroma client, a second embedding function instance, and the
whole LangGraph graph -- all real memory cost paid once at startup,
regardless of whether /router-chat ever gets called. Disabling it
entirely isolates whether running both pipelines in one process is
itself the tipping point on a memory-constrained host, separate from
anything about the embedding model or corpus size.

Both cost-generating endpoints require an API key IF the API_ACCESS_KEY
environment variable is set -- this is a real, necessary protection
once this is deployed publicly: with no auth at all, anyone who finds
the URL can call /chat or /router-chat as many times as they want,
spending real Anthropic credits on your key with no limit. Auth is
optional-by-default (skipped entirely if API_ACCESS_KEY isn't set) so
local development stays frictionless -- set it as an environment
variable on Render (or wherever this is deployed) to actually enforce it.

Example (no auth configured, e.g. local dev):
    curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
         -d '{"question": "How do I add a path parameter?"}'

Example (auth configured, e.g. deployed):
    curl -X POST https://your-app.onrender.com/chat \
         -H "Content-Type: application/json" \
         -H "X-API-Key: your-secret-key" \
         -d '{"question": "How do I add a path parameter?"}'
"""
import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from rag_chain import answer_question

API_ACCESS_KEY = os.getenv("API_ACCESS_KEY")  # unset -> auth disabled (local dev default)
ENABLE_ROUTER_CHAT = os.getenv("ENABLE_ROUTER_CHAT", "true").lower() not in ("false", "0", "no")

router_answer_question = None
if ENABLE_ROUTER_CHAT:
    from router_chain import answer_question as router_answer_question

app = FastAPI(title="Docs RAG Assistant")


def require_api_key(x_api_key: str = Header(default=None)):
    """FastAPI dependency: enforces the API key only if one is
    configured. Deliberately checked on every request, not cached --
    the cost of a string comparison is negligible next to an LLM call."""
    if API_ACCESS_KEY and x_api_key != API_ACCESS_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid API key")


class ChatRequest(BaseModel):
    question: str
    top_k: int = 4


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


@app.get("/debug-embed")
def debug_embed():
    """TEMPORARY diagnostic endpoint -- isolates whether the embedding
    step alone (not Chroma's query, not the Anthropic call) is what
    crashes on a memory-constrained host. Calls rag_chain.py's
    embedding function directly on a short test string, with nothing
    else involved. Remove this once the real cause is confirmed --
    it's not meant to be a permanent part of the API."""
    from rag_chain import _embed_fn

    result = _embed_fn(["test question"])
    return {"embedding_dimensions": len(result[0])}


class RouterChatRequest(BaseModel):
    question: str


class RouterChatResponse(BaseModel):
    question: str
    source: str  # "fastapi", "nodejs", "coding", or "offtopic" -- which path the router took
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    # Deliberately NOT behind the API key -- Render's health checks hit
    # this constantly and don't send auth headers, so protecting it
    # would make Render think the service is unhealthy and restart it.
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    result = answer_question(req.question, k=req.top_k)
    return ChatResponse(question=result["question"], answer=result["answer"], sources=result["sources"])


if ENABLE_ROUTER_CHAT:

    @app.post("/router-chat", response_model=RouterChatResponse, dependencies=[Depends(require_api_key)])
    def router_chat(req: RouterChatRequest):
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="question must not be empty")
        result = router_answer_question(req.question)
        return RouterChatResponse(
            question=result["question"],
            source=result["source"],
            answer=result["answer"],
            sources=result["sources"],
        )