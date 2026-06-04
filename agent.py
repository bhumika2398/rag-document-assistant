from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Annotated
import operator
from dotenv import load_dotenv
import os

load_dotenv()

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]
    question: str
    context: str
    answer: str

def create_agent(vectorstore):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 10}
    )

    def retrieve_node(state: AgentState):
        question = state["question"]
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        return {"context": context}

    def generate_node(state: AgentState):
        history = state["messages"][-10:]
        history_text = ""
        for msg in history:
            if isinstance(msg, HumanMessage):
                history_text += f"User: {msg.content}\n"
            else:
                history_text += f"Assistant: {msg.content}\n"

        prompt_text = f"""You are a helpful assistant answering questions about documents.
Use ONLY the context below to answer. If the answer is not in the context, say so.

Conversation so far:
{history_text}

Document context:
{state['context']}

Current question: {state['question']}

Answer:"""

        response = llm.invoke(prompt_text)
        answer = response.content

        new_messages = [
            HumanMessage(content=state["question"]),
            AIMessage(content=answer)
        ]

        return {"answer": answer, "messages": new_messages}

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()

def ask_agent(agent, question: str, history: list):
    result = agent.invoke({
        "question": question,
        "messages": history,
        "context": "",
        "answer": ""
    })
    return result["answer"], result["messages"]