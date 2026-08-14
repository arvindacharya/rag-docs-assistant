"""
Streamlit chat UI for the docs assistant.
Run with:
    streamlit run app_streamlit.py
"""
import streamlit as st

from feedback import save_feedback
from rag_chain import answer_question

st.set_page_config(page_title="Docs RAG Assistant", page_icon="📚")
st.title("📚 Docs RAG Assistant")
st.caption("Ask a question. Answers are grounded in the ingested docs and cite their sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_message(msg, index):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- `{s}`")
        if msg["role"] == "assistant":
            feedback = st.feedback("thumbs", key=f"feedback_{index}")
            # st.feedback fires on every rerun once a value is set, not just
            # on the click itself -- only save when the value actually
            # changes, or every unrelated interaction on the page would
            # re-append a duplicate record for this same message.
            if feedback is not None and msg.get("feedback_saved") != feedback:
                save_feedback(
                    app="v1",
                    question=msg.get("question", ""),
                    answer=msg["content"],
                    sources=msg.get("sources", []),
                    rating="up" if feedback == 1 else "down",
                )
                msg["feedback_saved"] = feedback


for i, msg in enumerate(st.session_state.messages):
    render_message(msg, i)

if question := st.chat_input("Ask a question about the docs..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating an answer..."):
            result = answer_question(question)
        st.markdown(result["answer"])
        with st.expander("Sources"):
            for s in result["sources"]:
                st.markdown(f"- `{s}`")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "question": question,  # kept so feedback can be tied back to what was asked
        }
    )
    st.rerun()  # re-render through the loop above so the new message gets its feedback widget
