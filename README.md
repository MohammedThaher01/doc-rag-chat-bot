
# 📄 DocChat — Document RAG Chatbot with LangGraph Agent

An end-to-end AI-powered document assistant that lets you upload PDFs, TXTs, or DOCXs and chat with them using a LangGraph autonomous agent backed by Groq's ultra-fast LPU inference.

🔗 **Live Demo:** [dog-rag-chatbot.streamlit.app](https://dog-rag-chatbot-v2ofgfbtpgppjkxpppmbee.streamlit.app/)

---

## 🧠 How It Works

This app uses a **RAG-first autonomous agent** built with LangGraph:


```
User Question
      ↓
[Retrieve Node] → FAISS similarity search on uploaded docs
      ↓
[Relevance Check] → Is the context useful?
      ↓                        ↓
[Generate Answer]         [Web Search Node] → DuckDuckGo
      ↓                        ↓
              [Generate Answer]
                      ↓
              Streamed response + Source label
```


1. Every question first hits the **FAISS vector store** built from your uploaded documents
2. If the retrieved context is insufficient, the agent **falls back to web search** automatically
3. The LLM generates a response and the UI shows whether the answer came from your **documents or the web**

---

## 🏗️ Architecture

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Agent Orchestration | LangGraph |
| LLM Inference | Groq API (GPT-OSS 20B) |
| Vector Store | FAISS (local) |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Web Search | DuckDuckGo (via `ddgs`) |
| OCR Fallback | Tesseract + pdf2image |
| Document Loaders | LangChain (PDF, TXT, DOCX) |

---

## ✨ Features

- Upload and chat with PDF, TXT, DOCX files
- LangGraph autonomous agent — RAG-first, web search fallback
- FAISS vector store with recursive chunking and persistent indexing
- OCR fallback for scanned/image-based PDFs using Tesseract
- Web search when document context is insufficient
- Conversational UI with source attribution (Document vs Web)
- Groq LPU inference — sub-second response times

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/doc-rag-chatbot.git
cd doc-rag-chatbot
```

### 2. Create and activate conda environment
```bash
conda create -n aienv python=3.11
conda activate aienv
```

### 3. Install system dependencies (Mac)
```bash
brew install tesseract poppler
```

### 4. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 5. Set up environment variables
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get your free key at [console.groq.com](https://console.groq.com) — no credit card needed.

### 6. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
doc-rag-chatbot/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # Document ingestion, FAISS vectorstore, embedding
├── agent.py            # LangGraph agent graph definition
├── tools.py            # RAG search and web search tool definitions
├── requirements.txt    # Python dependencies
├── packages.txt        # System dependencies for Streamlit Cloud
├── .env                # API keys (never commit this)
└── faiss_index/        # Auto-created vector index (never commit this)
```

---

## 🔧 Tech Stack

```
Python 3.11
LangChain + LangGraph
FAISS (CPU)
HuggingFace Sentence Transformers
Groq API
DuckDuckGo Search
Tesseract OCR
Streamlit
```

---

## 🌐 Deploy on Streamlit Cloud

1. Push repo to GitHub (ensure `faiss_index/` and `.env` are in `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app
3. Select your repo, branch, and `app.py`
4. Under **Secrets**, add:
```
GROQ_API_KEY = "your_key_here"
```
5. Deploy — `packages.txt` handles system deps automatically

---

## 📌 Key Design Decisions

**Why FAISS over Pinecone/Chroma?**
Runs in-process with zero infrastructure. No API keys, no database to manage. Trade-off: no metadata filtering, but sufficient for document Q&A.

**Why RAG-first agent?**
Keeps answers grounded in your documents by default. Web search only activates when the document context scores below the relevance threshold — prevents hallucination while maintaining utility for off-topic questions.

**Why Groq?**
LPU hardware delivers 500-1000 tokens/second — 5-10x faster than GPU inference. Entire stack is free for prototyping.

**Why `all-MiniLM-L6-v2`?**
Lightweight (~80MB), runs locally, strong semantic similarity performance. No API calls for embeddings.

---

## 🙋 Author

By **Mohammed Thaher S**

[LinkedIn](https://www.linkedin.com/in/mohammed-thaher-s/) • [GitHub](https://github.com/MohammedThaher01)
