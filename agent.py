import os
import re
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from tools import rag_search, web_search

load_dotenv()


class AgentState(TypedDict):
    question: str
    chat_history: str
    rag_context: str
    web_context: str
    final_answer: str
    source: str


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the environment or .env file.")

    return ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0.2,
        groq_api_key=api_key,
    )


def _is_follow_up(question: str) -> bool:
    """Identify references that need previous-turn context for retrieval."""
    text = question.strip().lower()
    if not text:
        return False
    patterns = (
        r"\b(it|this|that|they|them|these|those|their|its|above|previous|same)\b",
        r"\b(tell me more|go on|continue|elaborate|expand on|explain further)\b",
        r"\b(the first|the second|the third|the last)\s+(point|item|part|reason|example|section)\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def retrieve_node(state: AgentState) -> AgentState:
    question = state["question"]
    history = state.get("chat_history", "").strip()

    # Keep standalone queries focused. For follow-ups such as "tell me about it",
    # include recent turns so retrieval can resolve what "it" refers to.
    retrieval_query = question
    if history and _is_follow_up(question):
        retrieval_query = (
            f"Recent conversation:\n{history[-4500:]}\n\n"
            f"Current follow-up: {question}"
        )

    state["rag_context"] = rag_search.run(retrieval_query)
    return state


def relevance_check_node(state: AgentState) -> str:
    context = state["rag_context"]
    if not context or context == "NO_RELEVANT_CONTEXT" or len(context.strip()) < 50:
        return "web_search"
    return "generate_answer"


def web_search_node(state: AgentState) -> AgentState:
    state["web_context"] = web_search.run(state["question"])
    state["source"] = "web"
    return state


def generate_answer_node(state: AgentState) -> AgentState:
    llm = get_llm()
    if state.get("source") == "web":
        context = state["web_context"]
        source_note = "web search"
    else:
        context = state["rag_context"]
        source_note = "the uploaded documents"
        state["source"] = "rag"

    prompt = PromptTemplate(
        input_variables=["context", "question", "source_note", "chat_history"],
        template="""You are DocChat, an assistant helping the user understand documents.

Conversation so far:
{chat_history}

Answer the latest question using the relevant information from {source_note}.
Follow these response rules:
- Continue the existing conversation when history is present. Resolve follow-up references such as "it", "that", or "the second point" using the conversation.
- Answer the question directly. Do not start with "Hey there", "Hi", "Hello", or another greeting unless the user is greeting you for the first time.
- Do not reintroduce yourself or repeat generic opening phrases on each turn.
- Use clean Markdown: headings, bullets, numbered lists, and tables only when they improve readability.
- Never emit HTML tags such as <br>. Use Markdown line breaks and list syntax instead.
- Do not wrap ordinary currency or calculations in dollar-sign math delimiters. Write money as normal text, for example USD 2,400 per year.
- Keep table labels concise and make first-column labels complete words.
- Do not invent details unsupported by the context. If the context does not answer the question, say what is missing.

Relevant context:
{context}

Latest question: {question}

Answer:""",
    )
    response = (prompt | llm).invoke(
        {
            "context": context,
            "question": state["question"],
            "source_note": source_note,
            "chat_history": state.get("chat_history", "") or "(No previous conversation.)",
        }
    )
    answer = response.content
    if not isinstance(answer, str):
        answer = str(answer)
    state["final_answer"] = answer.strip()
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
            "web_search": "web_search",
        },
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


def run_agent(question: str, chat_history: str = "") -> dict:
    """Run DocChat. chat_history remains optional for backwards compatibility."""
    result = get_agent().invoke(
        {
            "question": question,
            "chat_history": chat_history,
            "rag_context": "",
            "web_context": "",
            "final_answer": "",
            "source": "rag",
        }
    )
    return {"answer": result["final_answer"], "source": result["source"]}
