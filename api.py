"""
FastAPI backend. Run with:
    uvicorn api:app --reload --port 8000

Two endpoints, two separate pipelines:
    /chat        -- v1, single-source (FastAPI docs only), via rag_chain.py
    /router-chat -- v2, four-way router (FastAPI docs, Node.js docs,
                    web search for other coding questions, or a decline
                    for anything non-technical), via router_chain.py

Example:
    curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
         -d '{"question": "How do I add a path parameter?"}'

    curl -X POST localhost:8000/router-chat -H "Content-Type: application/json" \
         -d '{"question": "How do I use EventEmitter in Node.js?"}'
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_chain import answer_question
from router_chain import answer_question as router_answer_question

app = FastAPI(title="Docs RAG Assistant")


class ChatRequest(BaseModel):
    question: str
    top_k: int = 4


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


class RouterChatRequest(BaseModel):
    question: str


class RouterChatResponse(BaseModel):
    question: str
    source: str  # "fastapi", "nodejs", "coding", or "offtopic" -- which path the router took
    answer: str
    sources: list[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    result = answer_question(req.question, k=req.top_k)
    return ChatResponse(question=result["question"], answer=result["answer"], sources=result["sources"])


@app.post("/router-chat", response_model=RouterChatResponse)
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