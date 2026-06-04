from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

def create_qa_chain(vectorstore):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    prompt = PromptTemplate(
        template="""You are a helpful assistant. Answer the question using 
ONLY the context below. If the answer is not in the context, say 
"I don't have enough information in the documents to answer this."

Context: {context}

Question: {question}

Answer:""",
        input_variables=["context", "question"]
    )

    retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5, "fetch_k": 10}
)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

def get_answer(chain_tuple, question: str):
    chain, retriever = chain_tuple
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    return answer, sources