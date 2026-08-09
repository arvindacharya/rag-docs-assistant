"""
FastAPI backend. Run with:
    uvicorn api:app --reload --port 8000

Then POST to /chat:
    curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
         -d '{"question": "How do I add a path parameter?"}'
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_chain import answer_question

app = FastAPI(title="Docs RAG Assistant")


class ChatRequest(BaseModel):
    question: str
    top_k: int = 4


class ChatResponse(BaseModel):
    question: str
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
