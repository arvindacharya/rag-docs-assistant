"""
Streamlit chat UI for the docs assistant.
Run with:
    streamlit run app_streamlit.py
"""
import streamlit as st

from rag_chain import answer_question

st.set_page_config(page_title="Docs RAG Assistant", page_icon="📚")
st.title("📚 Docs RAG Assistant")
st.caption("Ask a question. Answers are grounded in the ingested docs and cite their sources.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- `{s}`")

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
        {"role": "assistant", "content": result["answer"], "sources": result["sources"]}
    )
