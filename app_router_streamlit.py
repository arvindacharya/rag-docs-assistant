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
    "Ask about FastAPI or Node.js. A router decides which docs to search "
    "before answering -- see which source it picked below each answer."
)

SOURCE_LABELS = {"fastapi": "🐍 FastAPI docs", "nodejs": "🟢 Node.js docs"}

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("source"):
            st.caption(f"Routed to: {SOURCE_LABELS.get(msg['source'], msg['source'])}")
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- `{s}`")

if question := st.chat_input("Ask about FastAPI or Node.js..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Routing, retrieving, and generating..."):
            result = answer_question(question)
        st.markdown(result["answer"])
        st.caption(f"Routed to: {SOURCE_LABELS.get(result['source'], result['source'])}")
        with st.expander("Sources"):
            for s in result["sources"]:
                st.markdown(f"- `{s}`")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "source": result["source"],
            "sources": result["sources"],
        }
    )
