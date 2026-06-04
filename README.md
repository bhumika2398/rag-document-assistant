# RAG Document Q&A Assistant

An AI-powered document question-answering system built using RAG 
(Retrieval-Augmented Generation) architecture.

## What it does
Upload any PDF document and ask questions about it. The system finds 
the most relevant passages and generates accurate cited answers.

## Tech Stack
- LangChain — RAG pipeline and chain orchestration
- ChromaDB — Vector database for semantic search
- HuggingFace Embeddings — all-MiniLM-L6-v2 for text embeddings
- Groq LLM — Fast inference using Llama 3.3
- Streamlit — Web interface
- MMR Retrieval — Maximum Marginal Relevance for diverse results

## How it works
1. PDF is loaded and split into 500-character chunks
2. Each chunk is converted to embeddings using HuggingFace
3. Embeddings stored in ChromaDB vector database
4. User question is embedded and matched against stored chunks
5. Top 5 relevant chunks retrieved using MMR search
6. Chunks passed to LLM with prompt to generate cited answer

## Run locally
pip install -r requirements.txt
streamlit run app.py

## Sample questions tested
- "What is the Transformer model?"
- "What is multi-head attention?"
- "What is self-attention?"
- "What BLEU scores did the model achieve?"