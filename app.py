import streamlit as st
import tempfile
import os
from rag_pipeline import load_and_store_pdf
from agent import create_qa_chain, get_answer

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

if "chain" not in st.session_state:
    st.session_state.chain = None

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
        st.session_state.chain = create_qa_chain(st.session_state.vectorstore)
    st.success(f"Processed {len(uploaded_files)} document(s). Ask a question!")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask a question about your documents...")

if question and st.session_state.chain:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer, sources = get_answer(st.session_state.chain, question)
        st.write(answer)
        with st.expander("View source passages"):
            for i, doc in enumerate(sources[:3]):
                st.caption(f"Source {i+1} — Page {doc.metadata.get('page', '?')}")
                st.write(doc.page_content[:300] + "...")
    st.session_state.messages.append({"role": "assistant", "content": answer})

elif question and not st.session_state.chain:
    st.warning("Please upload and process a PDF first.")