import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

load_dotenv()
print(f"DEBUG: GROQ_API_KEY present: {bool(os.getenv('GROQ_API_KEY'))}")

FAISS_PATH = "faiss_index"
EMBED_MODEL = "all-MiniLM-L6-v2"   # free, ~80MB, fast on M4

def load_documents(file_paths: list[str]):
    docs = []
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.endswith(".txt"):
            loader = TextLoader(path)
        elif path.endswith(".docx"):
            loader = Docx2txtLoader(path)
        else:
            continue
        docs.extend(loader.load())
    return docs

def build_vectorstore(file_paths: list[str]):
    docs = load_documents(file_paths)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,       # overlap preserves context at chunk edges
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_PATH)
    return vectorstore, len(chunks)

def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(FAISS_PATH, embeddings,
                            allow_dangerous_deserialization=True)

def get_qa_chain(vectorstore, chat_history=None):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in environment or .env file.")
    
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0.3,
        groq_api_key=api_key
    )
    
    # Use the provided chat_history if available, otherwise create a new buffer
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=chat_history,
        return_messages=True,
        output_key="answer"
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        memory=memory,
        return_source_documents=True,
        verbose=False
    )
    return chain