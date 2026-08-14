"""
Streamlit chat UI for the v2 multi-source router agent (FastAPI + Node.js
+ web search + decline). This is separate from app_streamlit.py (the v1
single-source app) so v1 stays intact and demoable on its own.

Run with:
    streamlit run app_router_streamlit.py
"""
import streamlit as st

from feedback import save_feedback
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

if "messages" not in st.session_state:
    st.session_state.messages = []


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


def render_message(msg, index):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("source"):
            st.caption(f"Routed to: {SOURCE_LABELS.get(msg['source'], msg['source'])}")
        if msg["role"] == "assistant":
            render_sources(msg.get("sources"))
            feedback = st.feedback("thumbs", key=f"feedback_{index}")
            # Only save on an actual value change -- st.feedback re-reports
            # its current value on every rerun, not just on the click that
            # set it, so an unguarded save would duplicate on every
            # unrelated interaction elsewhere on the page.
            if feedback is not None and msg.get("feedback_saved") != feedback:
                save_feedback(
                    app="v2",
                    question=msg.get("question", ""),
                    answer=msg["content"],
                    sources=msg.get("sources", []),
                    rating="up" if feedback == 1 else "down",
                    source_category=msg.get("source"),
                )
                msg["feedback_saved"] = feedback


for i, msg in enumerate(st.session_state.messages):
    render_message(msg, i)

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
            "question": question,  # kept so feedback can be tied back to what was asked
        }
    )
    st.rerun()  # re-render through the loop above so the new message gets its feedback widget
