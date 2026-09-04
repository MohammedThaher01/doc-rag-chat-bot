import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

load_dotenv()
print(f"DEBUG: GROQ_API_KEY present: {bool(os.getenv('GROQ_API_KEY'))}")

FAISS_PATH = "faiss_index"
EMBED_MODEL = "all-MiniLM-L6-v2"

CUSTOM_PROMPT = PromptTemplate(
    input_variables=["context", "question", "chat_history"],
    template="""You are a friendly and helpful document assistant named DocChat. Your job is to answer questions based on the uploaded document context below.

If the question is unrelated to the documents, respond warmly and gently redirect the user. Don't be blunt — be conversational and encouraging.

Chat History:
{chat_history}

Context from documents:
{context}

Question: {question}

Answer:"""
)


def load_scanned_pdf(path: str) -> list:
    images = convert_from_path(path)
    docs = []
    for i, img in enumerate(images):
        text = pytesseract.image_to_string(img)
        if text.strip():
            docs.append(Document(
                page_content=text,
                metadata={"source": path, "page": i}
            ))
    return docs


def load_documents(file_paths: list[str]):
    docs = []
    for path in file_paths:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path)
            loaded = loader.load()
            if not loaded or all(d.page_content.strip() == "" for d in loaded):
                loaded = load_scanned_pdf(path)
            docs.extend(loaded)
        elif path.endswith(".txt"):
            docs.extend(TextLoader(path).load())
        elif path.endswith(".docx"):
            docs.extend(Docx2txtLoader(path).load())
    return docs


def build_vectorstore(file_paths: list[str]):
    docs = load_documents(file_paths)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise ValueError(
            "No text could be extracted from the uploaded files. "
            "They may be scanned images or image-based PDFs. "
            "Try uploading a text-based PDF or a .txt file instead."
        )

    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_PATH)
    return vectorstore, len(chunks)


def load_vectorstore():
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return FAISS.load_local(FAISS_PATH, embeddings,
                            allow_dangerous_deserialization=True)


def get_qa_chain(vectorstore, chat_history=None):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in environment or .env file.")

    llm = ChatGroq(
        model_name="llama3-8b-8192",
        temperature=0.3,
        groq_api_key=api_key
    )

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
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
        verbose=False
    )
    return chain
