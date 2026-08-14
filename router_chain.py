"""
Multi-source RAG router, built with LangGraph.

This is the "v2" agent version of the project: instead of one fixed
corpus, a question first goes through a router node that decides how
to handle it:

    question
       |
       v
    [route]  <- one Claude call: "fastapi", "nodejs", "coding", or "offtopic"
       |
       +-- "fastapi"  --> [retrieve_fastapi] --+
       |                                        +--> [generate] --> END
       +-- "nodejs"   --> [retrieve_nodejs]  --+
       |
       +-- "coding"   --> [web_search_node] -----------> END
       |
       +-- "offtopic" --> [decline_node] ---------------> END

Deliberately scoped: this tool answers FastAPI/Node.js questions via
its own docs, other genuine programming/technical questions via live
web search, and declines anything else outright rather than trying to
be a general-purpose assistant. "coding" and "offtopic" are a real,
separate decision (not "does the question mention FastAPI or Node"),
so a question like "how does Python's asyncio.gather work" correctly
gets web search, while "what's up" or general trivia gets declined
without ever calling an LLM for the actual response -- decline_node is
a fixed string, not a generation, which is both the cheapest possible
outcome (zero API cost) and the most reliable one (no risk of an LLM
drifting into being generally helpful about an out-of-scope question).

Why LangGraph here and not in rag_chain.py: rag_chain.py is a straight
line (retrieve -> generate), which doesn't need a graph framework --
there's no fork in it. This module has genuine decision points (which
source, and a hard scope boundary), which is exactly the case
LangGraph's conditional edges are built for.

Version note: pinned to langgraph==0.2.60 specifically because it's the
newest version whose langchain-core requirement (>=0.2.43,<0.4.0) is
still compatible with the langchain-core==0.3.86 pin the eval stack
(ragas) needs. langgraph 1.x requires langchain-core>=1.4, which
directly conflicts with ragas 0.3.9's requirements in the same
environment. If you ever bump ragas past the 0.4.x import bug noted in
requirements.txt, re-check this compatibility before bumping langgraph
too. Verified: both `import ragas` and this module's graph import
cleanly together at these pinned versions.
"""
import os
from typing import Literal, TypedDict

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langfuse import get_client
from langgraph.graph import END, StateGraph
from reranker import rerank

load_dotenv()

DB_DIR = os.getenv("CHROMA_DB_DIR", "data/chroma_db")
MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))

# FastAPI reuses the existing "docs" collection built by ingest.py in
# the v1 pipeline -- no need to re-ingest the same content twice.
# Node.js is a new collection; build it with:
#   python fetch_nodejs_docs.py
#   python ingest.py --docs_dir data/nodejs_docs --collection nodejs_docs
COLLECTIONS = {
    "fastapi": os.getenv("CHROMA_COLLECTION", "docs"),
    "nodejs": os.getenv("NODEJS_CHROMA_COLLECTION", "nodejs_docs"),
}

ROUTER_SYSTEM_PROMPT = """You decide how to handle a question for a documentation \
assistant that ONLY answers programming/technical questions -- specifically FastAPI, \
Node.js, and general coding topics. It does not handle anything else.
Reply with EXACTLY ONE WORD, no punctuation, no explanation:
- "fastapi" if the question is about the FastAPI Python web framework
- "nodejs" if the question is about Node.js, npm, or JavaScript runtime APIs
- "coding" if it's a genuine programming/technical/software question about
  something else -- another language, framework, library, tool, or API
- "offtopic" if it is NOT a programming/technical question at all -- casual
  conversation, general knowledge, current events, or any non-coding topic
If genuinely unsure between fastapi and nodejs, pick the one that seems more
likely rather than falling back to another category."""

ANSWER_SYSTEM_PROMPT = """You are a documentation assistant. Answer the user's question \
using ONLY the provided context chunks. If the answer isn't in the context, say \
you don't know instead of guessing. Cite the source file for every claim, like \
[source: path/to/file.md]."""

WEB_SEARCH_SYSTEM_PROMPT = """You are the fallback assistant inside a documentation Q&A \
tool specialized in FastAPI and Node.js. You only see questions that are genuine \
programming/technical questions about something other than those two -- another \
language, framework, library, tool, or API.

Answer the question directly and concisely, using web search if the question needs \
current or factual information you're not confident about. If asked what this tool \
does, explain plainly: it answers FastAPI and Node.js questions from their official \
documentation, and handles other coding questions via web search like this one."""

DECLINE_MESSAGE = (
    "I'm built specifically to answer programming and technical questions -- "
    "FastAPI and Node.js from their official documentation, and other coding "
    "topics via web search. I'm not able to help with non-technical questions."
)

# Anthropic's web search tool version. Confirmed against the installed
# anthropic SDK (0.121.0), which defines WebSearchTool20260318Param --
# this is a real, current tool type string, not guessed. max_uses caps
# how many searches one request can trigger, as a cost control.
WEB_SEARCH_TOOL = {"type": "web_search_20260318", "name": "web_search", "max_uses": 3}

# IMPORTANT: this must match whatever embedding function ingest.py used
# to build both collections, or similarity scores are meaningless
# (comparing vectors from two different embedding spaces). Matches the
# SentenceTransformerEmbeddingFunction swap made for the corporate-
# network S3-block fix earlier in this project -- update here too if
# you change it in ingest.py/rag_chain.py.
_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

_chroma_client = chromadb.PersistentClient(path=DB_DIR)
_anthropic_client = anthropic.Anthropic()
langfuse = get_client()


class RouterState(TypedDict):
    question: str
    source: str
    chunks: list
    answer: str


def _get_collection(name: str):
    return _chroma_client.get_or_create_collection(name, embedding_function=_embed_fn)


def _retrieve(collection_name: str, question: str, k: int = TOP_K):
    collection = _get_collection(collection_name)
    candidate_k = max(k * 4, 15)
    results = collection.query(query_texts=[question], n_results=candidate_k)
    chunks = [
        {"text": doc, "source": meta["source"], "distance": dist}
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        )
    ]
    return rerank(question, chunks, top_k=k)


def _extract_text(message) -> str:
    """Pull the actual answer text out of a Claude response, without
    assuming content[0] is the text block.

    Claude Sonnet 5 runs with adaptive thinking on by default -- the
    model decides per-request whether reasoning is warranted, and when
    it does, the response includes a ThinkingBlock (which has no .text
    attribute at all, only .thinking) ahead of the TextBlock. Anthropic's
    own migration docs show exactly this pattern and recommend iterating
    over response.content rather than indexing it. Confirmed in practice:
    route_node's tiny max_tokens=10 budget is a hard ceiling on thinking
    PLUS the answer combined, so a request that triggers even brief
    thinking can exhaust the budget before any text is emitted at all,
    leaving content with only a ThinkingBlock and no TextBlock -- that
    exact failure crashed this project's first real router-eval run.
    """
    for block in message.content:
        if getattr(block, "type", None) == "text":
            return block.text
    block_types = [getattr(b, "type", type(b).__name__) for b in message.content]
    raise RuntimeError(
        f"No text block in Claude's response -- got only {block_types}. "
        "This can happen when max_tokens is too tight for a model with "
        "adaptive thinking on by default (thinking can consume the whole "
        "budget before any answer text is produced). Try raising max_tokens."
    )


def _extract_web_search_result(message):
    """Web search responses can interleave several blocks: thinking,
    server_tool_use (the search call itself), web_search_tool_result
    (raw results), and one or more text blocks -- Claude sometimes
    writes a bit of text, searches again, then writes more. Unlike
    _extract_text() (fine for the single-text-block case elsewhere in
    this file), this concatenates every text block in order and
    collects citations from all of them, so a multi-step search
    doesn't silently drop part of the answer or its sources.

    Returns (answer_text, sources) where sources is a deduplicated list
    of "Title (url)" strings pulled from Claude's own citation objects
    (CitationsWebSearchResultLocation: .url, .title, .cited_text) --
    confirmed against the installed anthropic SDK's type definitions.
    """
    text_parts = []
    seen_urls = set()
    sources = []
    for block in message.content:
        if getattr(block, "type", None) != "text":
            continue
        text_parts.append(block.text)
        for citation in getattr(block, "citations", None) or []:
            url = getattr(citation, "url", None)
            if url and url not in seen_urls:
                seen_urls.add(url)
                title = getattr(citation, "title", None) or url
                sources.append(f"{title} ({url})")

    if not text_parts:
        block_types = [getattr(b, "type", type(b).__name__) for b in message.content]
        raise RuntimeError(f"No text block in Claude's web search response -- got only {block_types}.")

    return "".join(text_parts), sources


def route_node(state: RouterState) -> dict:
    with langfuse.start_as_current_observation(
        as_type="generation", name="route-question", model=MODEL
    ) as gen:
        message = _anthropic_client.messages.create(
            model=MODEL,
            max_tokens=20,
            thinking={"type": "disabled"},  # a one-word classification needs no reasoning,
            # and this sidesteps the max_tokens-vs-thinking-budget issue entirely for this call
            system=ROUTER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": state["question"]}],
        )
        raw = _extract_text(message).strip().lower()
        if "offtopic" in raw:
            source = "offtopic"
        elif "coding" in raw:
            source = "coding"
        elif "nodejs" in raw:
            source = "nodejs"
        else:
            source = "fastapi"  # default on any ambiguity between the two doc sources
        gen.update(input=state["question"], output=source)
    return {"source": source}


def retrieve_fastapi_node(state: RouterState) -> dict:
    with langfuse.start_as_current_observation(as_type="span", name="retrieve-fastapi") as span:
        chunks = _retrieve(COLLECTIONS["fastapi"], state["question"])
        span.update(input=state["question"], output={"sources": [c["source"] for c in chunks]})
    return {"chunks": chunks}


def retrieve_nodejs_node(state: RouterState) -> dict:
    with langfuse.start_as_current_observation(as_type="span", name="retrieve-nodejs") as span:
        chunks = _retrieve(COLLECTIONS["nodejs"], state["question"])
        span.update(input=state["question"], output={"sources": [c["source"] for c in chunks]})
    return {"chunks": chunks}


def decline_node(state: RouterState) -> dict:
    """Out-of-scope questions get a fixed message, not a generation.
    Deliberate: this is both the cheapest possible outcome (zero API
    cost -- no Claude call at all) and the most reliable one (a fixed
    string can't drift into being generally helpful about something
    this tool is explicitly scoped not to handle)."""
    with langfuse.start_as_current_observation(as_type="span", name="decline") as span:
        span.update(input=state["question"], output=DECLINE_MESSAGE)
    return {"answer": DECLINE_MESSAGE, "chunks": []}


def web_search_node(state: RouterState) -> dict:
    """Handles anything outside the two doc corpora. Anthropic's
    web_search tool runs the search AND writes the cited answer in the
    same API call -- there's no separate retrieve step, so this node
    both gathers information and produces the final answer, unlike the
    doc branches which split those into two nodes."""
    with langfuse.start_as_current_observation(
        as_type="generation", name="web-search-answer", model=MODEL
    ) as gen:
        message = _anthropic_client.messages.create(
            model=MODEL,
            max_tokens=800,
            system=WEB_SEARCH_SYSTEM_PROMPT,
            tools=[WEB_SEARCH_TOOL],
            messages=[{"role": "user", "content": state["question"]}],
        )
        answer, sources = _extract_web_search_result(message)
        gen.update(
            input=state["question"],
            output=answer,
            usage_details={"input": message.usage.input_tokens, "output": message.usage.output_tokens},
        )
    chunks = [{"text": "", "source": s, "distance": 0.0} for s in sources]
    return {"answer": answer, "chunks": chunks}


def generate_node(state: RouterState) -> dict:
    context = "\n\n".join(f"[source: {c['source']}]\n{c['text']}" for c in state["chunks"])
    user_message = f"Context:\n{context}\n\nQuestion: {state['question']}"
    with langfuse.start_as_current_observation(
        as_type="generation", name="generate-answer", model=MODEL
    ) as gen:
        message = _anthropic_client.messages.create(
            model=MODEL,
            max_tokens=600,
            system=ANSWER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        answer = _extract_text(message)
        gen.update(
            input=user_message,
            output=answer,
            usage_details={"input": message.usage.input_tokens, "output": message.usage.output_tokens},
        )
    return {"answer": answer}


def _route_decision(state: RouterState) -> Literal["fastapi", "nodejs", "coding", "offtopic"]:
    return state["source"]


def build_graph():
    graph = StateGraph(RouterState)
    graph.add_node("route", route_node)
    graph.add_node("retrieve_fastapi", retrieve_fastapi_node)
    graph.add_node("retrieve_nodejs", retrieve_nodejs_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("decline", decline_node)
    graph.add_node("generate", generate_node)

    graph.set_entry_point("route")
    graph.add_conditional_edges(
        "route",
        _route_decision,
        {
            "fastapi": "retrieve_fastapi",
            "nodejs": "retrieve_nodejs",
            "coding": "web_search",
            "offtopic": "decline",
        },
    )
    graph.add_edge("retrieve_fastapi", "generate")
    graph.add_edge("retrieve_nodejs", "generate")
    graph.add_edge("generate", END)
    graph.add_edge("web_search", END)  # bypasses "generate" -- the web_search tool already wrote the answer
    graph.add_edge("decline", END)  # bypasses everything -- fixed message, no LLM call at all
    return graph.compile()


_graph = build_graph()


def route_question(question: str) -> str:
    """Just the routing decision, no retrieval or generation. Used by
    the router-accuracy eval so testing routing doesn't cost a full
    retrieve+generate cycle (and its own Langfuse trace) per question."""
    return route_node({"question": question, "source": "", "chunks": [], "answer": ""})["source"]


def answer_question(question: str):
    """Top-level entry point, mirrors rag_chain.answer_question()'s
    return shape plus a 'source' field showing which corpus was used."""
    with langfuse.start_as_current_observation(as_type="span", name="router-answer-question") as trace:
        trace.update(input={"question": question})
        result = _graph.invoke({"question": question, "source": "", "chunks": [], "answer": ""})
        output = {
            "question": question,
            "source": result["source"],
            "answer": result["answer"],
            "sources": [c["source"] for c in result["chunks"]],
            "chunks": result["chunks"],
            "trace_id": trace.trace_id,
        }
        trace.update(output={"source": output["source"], "answer": output["answer"], "sources": output["sources"]})
        return output


def shutdown():
    langfuse.flush()
