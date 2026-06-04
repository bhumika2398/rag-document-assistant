from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_and_store_pdf(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    print("Stored in ChromaDB successfully")
    return vectorstore

def retrieve_chunks(vectorstore, question: str, k: int = 5):
    docs = vectorstore.similarity_search(question, k=k)
    return docs

if __name__ == "__main__":
    vs = load_and_store_pdf("test.pdf")
    results = retrieve_chunks(vs, "What is multi-head attention?")
    for i, doc in enumerate(results):
        print(f"\n--- Chunk {i+1} ---")
        print(doc.page_content[:300])