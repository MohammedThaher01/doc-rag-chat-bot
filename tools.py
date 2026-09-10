from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

FAISS_PATH = "faiss_index"
EMBED_MODEL = "all-MiniLM-L6-v2"

search = DuckDuckGoSearchRun()

@tool
def web_search(query: str) -> str:
    """Search the web for information not found in documents."""
    return search.run(query)

@tool
def rag_search(query: str) -> str:
    """Search the uploaded documents for relevant context."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
        vectorstore = FAISS.load_local(FAISS_PATH, embeddings,
                                       allow_dangerous_deserialization=True)
        docs = vectorstore.similarity_search(query, k=2)
        if not docs:
            return "NO_RELEVANT_CONTEXT"
        return "\n\n".join([d.page_content for d in docs])
    except Exception:
        return "NO_RELEVANT_CONTEXT"