import os
from dotenv import load_dotenv
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from tools import rag_search, web_search

load_dotenv()

class AgentState(TypedDict):
    question: str
    rag_context: str
    web_context: str
    final_answer: str
    source: str

def get_llm():
    return ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0.3,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

def retrieve_node(state: AgentState) -> AgentState:
    result = rag_search.run(state["question"])
    state["rag_context"] = result
    return state

def relevance_check_node(state: AgentState) -> str:
    ctx = state["rag_context"]
    if not ctx or ctx == "NO_RELEVANT_CONTEXT" or len(ctx.strip()) < 50:
        return "web_search"
    return "generate_answer"

def web_search_node(state: AgentState) -> AgentState:
    result = web_search.run(state["question"])
    state["web_context"] = result
    state["source"] = "web"
    return state

def generate_answer_node(state: AgentState) -> AgentState:
    llm = get_llm()
    if state.get("source") == "web":
        context = state["web_context"]
        source_note = "web search"
    else:
        context = state["rag_context"]
        source_note = "your uploaded documents"
        state["source"] = "rag"

    prompt = PromptTemplate(
        input_variables=["context", "question", "source_note"],
        template="""You are a friendly document assistant named DocChat.
Answer using the context from {source_note}. Be conversational.

Context:
{context}

Question: {question}

Answer:"""
    )
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": state["question"],
        "source_note": source_note
    })
    state["final_answer"] = response.content
    return state

def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generate_answer", generate_answer_node)
    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve",
        relevance_check_node,
        {
            "generate_answer": "generate_answer",
            "web_search": "web_search"
        }
    )
    graph.add_edge("web_search", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()

_agent = None

def get_agent():
    global _agent
    if _agent is None:
        _agent = build_agent()
    return _agent

def run_agent(question: str) -> dict:
    agent = get_agent()
    result = agent.invoke({
        "question": question,
        "rag_context": "",
        "web_context": "",
        "final_answer": "",
        "source": "rag"
    })
    return {
        "answer": result["final_answer"],
        "source": result["source"]
    }