import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.schema import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

load_dotenv()

FAISS_PATH = "faiss_index"
EMBED_MODEL = "all-MiniLM-L6-v2"


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
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise ValueError(
            "No text could be extracted from the uploaded files. "
            "They may be scanned images or image-based PDFs. "
            "Try uploading a text-based PDF or a .txt file instead."
        )
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(FAISS_PATH)
    return vectorstore, len(chunks)


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(FAISS_PATH, embeddings,
                            allow_dangerous_deserialization=True)


def get_qa_chain(vectorstore):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in environment or .env file.")

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        groq_api_key=api_key
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    prompt = ChatPromptTemplate.from_template("""You are a helpful document assistant. Answer based only on the context below. If the question is unrelated to the documents, say so warmly.

Context:
{context}

Question: {question}

Answer:""")

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
