"""
Streamlit chat UI for the v2 multi-source router agent (FastAPI + Node.js).
This is separate from app_streamlit.py (the v1 single-source app) so v1
stays intact and demoable on its own.

Run with:
    streamlit run app_router_streamlit.py
"""
import streamlit as st

from router_chain import answer_question

st.set_page_config(page_title="Multi-Source Docs Agent", page_icon="🧭")
st.title("🧭 Multi-Source Docs Agent")
st.caption(
    "Programming questions only. FastAPI and Node.js are answered from "
    "their official docs; other coding questions use live web search; "
    "anything non-technical gets politely declined."
)

SOURCE_LABELS = {
    "fastapi": "🐍 FastAPI docs",
    "nodejs": "🟢 Node.js docs",
    "coding": "🌐 Other coding (web search)",
    "offtopic": "🚫 Out of scope",
}


def render_sources(sources):
    if not sources:
        st.caption("Answered directly, no sources needed.")
        return
    with st.expander("Sources"):
        for s in sources:
            if s.startswith("http") or " (http" in s:
                st.markdown(f"- {s}")  # web search citations already read as "Title (url)"
            else:
                st.markdown(f"- `{s}`")  # local doc file paths


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("source"):
            st.caption(f"Routed to: {SOURCE_LABELS.get(msg['source'], msg['source'])}")
        render_sources(msg.get("sources"))

if question := st.chat_input("Ask a FastAPI, Node.js, or other coding question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Routing, then retrieving or searching, then generating..."):
            result = answer_question(question)
        st.markdown(result["answer"])
        st.caption(f"Routed to: {SOURCE_LABELS.get(result['source'], result['source'])}")
        render_sources(result["sources"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "source": result["source"],
            "sources": result["sources"],
        }
    )
