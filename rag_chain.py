"""
Core retrieval + generation logic, shared by the API, the Streamlit UI,
and the eval script. Instrumented with Langfuse so every call shows up
as a trace you can inspect.
"""
import os

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

DB_DIR = os.getenv("CHROMA_DB_DIR", "data/chroma_db")
COLLECTION = os.getenv("CHROMA_COLLECTION", "docs")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))

SYSTEM_PROMPT = """You are a documentation assistant. Answer the user's question \
using ONLY the provided context chunks. If the answer isn't in the context, say \
you don't know instead of guessing. Cite the source file for every claim, like \
[source: path/to/file.md]."""

_chroma_client = chromadb.PersistentClient(path=DB_DIR)
_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
_collection = _chroma_client.get_or_create_collection(COLLECTION, embedding_function=_embed_fn)
_anthropic_client = anthropic.Anthropic()
langfuse = get_client()


def retrieve(question: str, k: int = TOP_K):
    with langfuse.start_as_current_observation(as_type="span", name="retrieve") as span:
        results = _collection.query(query_texts=[question], n_results=k)
        chunks = [
            {"text": doc, "source": meta["source"], "distance": dist}
            for doc, meta, dist in zip(
                results["documents"][0], results["metadatas"][0], results["distances"][0]
            )
        ]
        span.update(
            input={"question": question, "k": k},
            output={"sources": [c["source"] for c in chunks], "distances": [c["distance"] for c in chunks]},
        )
        return chunks

def _extract_text(message) -> str:
    for block in message.content:
        if getattr(block, "type", None) == "text":
            return block.text
    block_types = [getattr(b, "type", type(b).__name__) for b in message.content]
    raise RuntimeError(f"No text block in Claude's response -- got only {block_types}.")

def generate_answer(question: str, chunks: list):
    context = "\n\n".join(f"[source: {c['source']}]\n{c['text']}" for c in chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    with langfuse.start_as_current_observation(
        as_type="generation", name="generate-answer", model=MODEL
    ) as generation:
        message = _anthropic_client.messages.create(
            model=MODEL,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer = _extract_text(message)
        generation.update(
            input=user_message,
            output=answer,
            usage_details={
                "input": message.usage.input_tokens,
                "output": message.usage.output_tokens,
            },
        )
        return answer


def answer_question(question: str, k: int = TOP_K):
    """Top-level entry point: one Langfuse trace per question, with
    retrieve() and generate_answer() nested underneath it."""
    with langfuse.start_as_current_observation(as_type="span", name="answer-question") as trace:
        trace.update(input={"question": question})
        chunks = retrieve(question, k=k)
        answer = generate_answer(question, chunks)
        result = {
            "question": question,
            "answer": answer,
            "sources": [c["source"] for c in chunks],
            "chunks": chunks,
        }
        trace.update(output={"answer": answer, "sources": result["sources"]})
        return result


def shutdown():
    """Call on process exit in short-lived scripts (eval runs, CLI use)
    so buffered trace events actually get sent to Langfuse."""
    langfuse.flush()
