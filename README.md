# RAG Document Q&A Assistant

An AI-powered document question-answering system built using RAG 
(Retrieval-Augmented Generation) architecture with conversational memory.

## What it does
Upload any PDF document and have a multi-turn conversation about it. 
The system finds the most relevant passages and generates accurate 
cited answers. It remembers previous questions so follow-up questions 
work naturally.

## Live Demo
[Coming soon — deploying on Streamlit Cloud]

## Tech Stack
- LangChain — RAG pipeline and chain orchestration
- LangGraph — Conversational memory via state machine (Retrieve → Generate nodes)
- ChromaDB — Vector database for semantic search
- HuggingFace Embeddings — all-MiniLM-L6-v2 for text embeddings
- Groq LLM — Fast inference using Llama 3.3 70B
- Streamlit — Web interface
- MMR Retrieval — Maximum Marginal Relevance for diverse results
- Custom Evaluation — Faithfulness 0.66 and Answer Relevancy 0.70 
  measured across 10 test questions

## How it works
1. PDF is loaded and split into 500-character chunks with 50-char overlap
2. Each chunk is converted to embeddings using HuggingFace all-MiniLM-L6-v2
3. Embeddings stored in ChromaDB vector database on local disk
4. User question is embedded and matched against stored chunks
5. Top 5 relevant chunks retrieved using MMR search
6. Chunks passed to Groq LLM with prompt to generate cited answer
7. Conversation history stored in LangGraph state — enables multi-turn Q&A
8. Pipeline evaluated using custom scoring framework measuring 
   Faithfulness and Answer Relevancy across 10 test cases

## Architecture
User uploads PDF
      ↓
PyPDF loads + RecursiveCharacterTextSplitter chunks (500 chars)
      ↓
HuggingFace all-MiniLM-L6-v2 creates embeddings
      ↓
ChromaDB stores embeddings locally
      ↓
User asks question
      ↓
LangGraph Retrieve Node — MMR search ChromaDB (top 5 chunks)
      ↓
LangGraph Generate Node — Groq LLM + conversation history
      ↓
Cited answer returned to Streamlit UI

## Evaluation Results
Tested on 10 questions from the Attention Is All You Need paper:

Faithfulness Score:     0.66
Answer Relevancy Score: 0.70

Faithfulness measures how grounded answers are in retrieved document 
context. Answer Relevancy measures how well answers address the question.

## Sample questions tested
- "What is the Transformer model?"
- "What is multi-head attention?"
- "What is self-attention?"
- "What BLEU scores did the model achieve?"
- "How many attention heads does the Transformer use?"
- "What optimizer was used?"

## Memory demonstration
The system remembers conversation context. Example:

User: "What is self-attention?"
Assistant: [explains self-attention]
User: "How is it different from multi-head attention?"
Assistant: [answers using context from previous exchange]

## Run locally
1. Clone the repo
   git clone https://github.com/bhumika2398/rag-document-assistant.git

2. Create virtual environment
   python -m venv venv
   venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

4. Add your API key to .env file
   GROQ_API_KEY=your-groq-key-here

5. Run the app
   streamlit run app.py

## Project structure
rag-document-assistant/
├── app.py              — Streamlit web interface
├── agent.py            — LangGraph state machine with memory
├── rag_pipeline.py     — PDF loading, chunking, ChromaDB storage
├── evaluate.py         — Custom evaluation framework
├── requirements.txt    — All dependencies
└── README.md           — This file