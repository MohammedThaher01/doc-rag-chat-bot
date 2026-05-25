# 📄 DocChat: AI-Powered Document RAG Chatbot

DocChat is a Retrieval-Augmented Generation (RAG) chatbot that allows you to upload documents and have conversational interactions with their content. It leverages high-performance LLMs via Groq and local vector storage for fast and accurate information retrieval.

## 🚀 Features

- **Multi-format Support:** Upload and process `.pdf`, `.txt`, and `.docx` files.
- **OCR Fallback:** Automatically handles scanned PDFs and images within documents using Tesseract OCR.
- **Persistent Memory:** Maintains conversation history for natural, context-aware dialogues.
- **Source Attribution:** Clearly identifies which parts of your documents were used to generate each answer.
- **Fast Retrieval:** Uses FAISS for efficient similarity search and indexing.
- **Groq Integration:** Powered by `llama-3.3-70b-versatile` for lightning-fast responses.

## 🛠️ Tech Stack

- **Frontend:** [Streamlit](https://streamlit.io/)
- **LLM Orchestration:** [LangChain](https://www.langchain.com/)
- **LLM Provider:** [Groq Cloud](https://groq.com/)
- **Embeddings:** [HuggingFace](https://huggingface.co/) (`all-MiniLM-L6-v2`)
- **Vector Store:** [FAISS](https://github.com/facebookresearch/faiss)
- **OCR Engine:** [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

## 📋 Prerequisites

Before running the application, ensure you have the following installed:

1. **Python 3.9+**
2. **Tesseract OCR:** 
   - macOS: `brew install tesseract`
   - Ubuntu: `sudo apt install tesseract-ocr`
3. **Poppler (for PDF processing):**
   - macOS: `brew install poppler`
   - Ubuntu: `sudo apt install poppler-utils`

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd doc-rag-chatbot
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Environment Variables:**
   Create a `.env` file in the root directory and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## 🏃 Running the App

Start the Streamlit server:
```bash
streamlit run app.py
```

1. Open your browser to the URL provided (usually `http://localhost:8501`).
2. Upload your documents in the sidebar.
3. Click **⚡ Process Documents**.
4. Start chatting with your documents!

## 📁 Project Structure

- `app.py`: Streamlit frontend and chat interface logic.
- `rag_pipeline.py`: Core RAG logic, document loading, and vector store management.
- `requirements.txt`: Python library dependencies.
- `packages.txt`: List of system-level packages (useful for deployment).
- `faiss_index/`: Local directory where the vector index is persisted.

---
*Created by [Mohammed Thahers](https://github.com/mohammedthahers](https://github.com/MohammedThaher01)*
