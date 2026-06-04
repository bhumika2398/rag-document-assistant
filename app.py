import streamlit as st
import tempfile
import os
from rag_pipeline import load_and_store_pdf
from agent import create_agent, ask_agent
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Document Q&A", layout="wide")
st.title("AI Document Q&A Assistant")
st.caption("Upload a PDF and ask questions about it")

with st.sidebar:
    st.header("Upload Document")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )
    process_btn = st.button("Process Documents")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "agent" not in st.session_state:
    st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if process_btn and uploaded_files:
    with st.spinner("Reading and indexing your documents..."):
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded_file.read())
                tmp_path = f.name
            st.session_state.vectorstore = load_and_store_pdf(tmp_path)
            os.unlink(tmp_path)
        st.session_state.agent = create_agent(st.session_state.vectorstore)
    st.success(f"Processed {len(uploaded_files)} document(s). Ask a question!")

for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

question = st.chat_input("Ask a question about your documents...")

if question and st.session_state.agent:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, updated_messages = ask_agent(
                st.session_state.agent,
                question,
                st.session_state.messages
            )
        st.write(answer)
    st.session_state.messages = updated_messages

elif question and not st.session_state.agent:
    st.warning("Please upload and process a PDF first.")