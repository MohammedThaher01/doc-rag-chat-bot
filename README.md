
# DocChat

DocChat is a document-aware chat application designed for PDF, TXT, and DOCX files. It combines a Streamlit interface, a LangGraph agent, vector retrieval, and Groq-hosted language model inference to help users ask grounded questions about uploaded documents.

A live deployment is available here: [dog-rag-chatbot.streamlit.app](https://dog-rag-chatbot-v2ofgfbtpgppjkxpppmbee.streamlit.app/)

## Features

- Chat with PDF, TXT, and DOCX files
- LangGraph autonomous agent with RAG-first, web search fallback
- Question classification: overview, project, and specific retrieval strategies
- Multi-query retrieval with chunk deduplication
- OCR fallback for scanned and image-based PDFs
- Source attribution on every response
- Groq LPU inference for sub-second responses
  
## Overview

DocChat is built for document Q&A workflows where answers should remain grounded in the uploaded material. The system performs the following steps:

1. Extract text from the uploaded files.
2. Split the document into overlapping chunks.
3. Generate embeddings and build a local FAISS index.
4. Classify the incoming question to determine the retrieval strategy.
5. Retrieve relevant passages from the active workspace index.
6. Use a LangGraph agent to answer the question with conversation context and source awareness.
7. Fall back to web search only when the uploaded context is insufficient.

## Agent Flow

```
User Question
      ↓
[Question Classifier] → overview / project / specific
      ↓
[Retrieve Node] → FAISS similarity search (multi-query for broad questions)
      ↓
[Relevance Check] → Is the context sufficient?
      ↓                        ↓
[Generate Answer]     [Web Search Node] → DuckDuckGo
      ↓                        ↓
              [Generate Answer]
                      ↓
         Response + Source label (Document / Web)
```
## Key capabilities

- Chat with multiple document types: PDF, TXT, and DOCX
- Conversational memory for follow-up questions
- Workspace-aware retrieval using the active FAISS index
- Multi-query retrieval for broad questions and project summaries
- Full-document retrieval for smaller collections to avoid missing important sections
- Deduplication of overlapping chunks before passing context to the model
- Source attribution between document-based answers and web fallback answers
- Markdown-safe response formatting with cleaner rendering in the UI

## Architecture

| Layer | Technology |
|---|---|
| User interface | Streamlit |
| Agent orchestration | LangGraph |
| LLM inference | Groq API with GPT-OSS 20B |
| Vector store | FAISS |
| Embedding model | HuggingFace all-MiniLM-L6-v2 |
| Web search fallback | DuckDuckGo |
| OCR support | Tesseract and pdf2image |
| Document loaders | LangChain |

## Retrieval design

The retrieval layer was improved to better support resume, portfolio, and document-summary questions.

### Question classification

Incoming prompts are classified into one of three groups:

- Overview: broad questions about the whole document, such as summary or general overview
- Projects: questions about portfolio items, projects, recognitions, or achievements
- Specific: direct factual questions that target one detail or section

### Retrieval strategy

- Overview and project questions expand the query into multiple complementary search terms.
- Smaller indexes can retrieve all chunks instead of only the top similarity matches.
- Larger collections use multi-query semantic search with deduplication.
- Results are capped to keep the final prompt manageable.
- Previous user turns are used to resolve follow-up references, while earlier assistant responses are not treated as evidence.

This makes the system more reliable for conversational document exploration compared with a simple top-two-chunk retrieval approach.

## Document processing

The ingestion pipeline uses recursive chunking to preserve context while keeping the chunk size practical for retrieval.

- Chunk size: 850 characters
- Chunk overlap: 100 characters
- Embedding model: all-MiniLM-L6-v2
- Vector database: FAISS

Larger chunks preserve surrounding context, while overlap helps prevent losing information at boundaries between sections.

## Project structure

```text
doc-rag-chatbot/
├── app.py              # Streamlit UI and chat rendering
├── agent.py            # LangGraph agent and conversational logic
├── tools.py            # Retrieval, web fallback, and query expansion
├── rag_pipeline.py     # Document loading, chunking, and FAISS index creation
├── requirements.txt    # Python runtime dependencies
├── packages.txt        # System packages for deployment environments
├── .env                # Local environment variables such as GROQ_API_KEY
├── faiss_index/        # Generated FAISS index files
├── evaluate.py         # Evaluation utilities
├── test_agent.py       # Agent-related tests
├── README.md           # Project documentation
└── .gitignore          # Excludes local secrets and generated artifacts
```

## Local setup

### 1. Clone the repository

```bash
git clone https://github.com/MohammedThaher01/doc-rag-chatbot.git
cd doc-rag-chatbot
```

### 2. Create and activate a Python environment

```bash
conda create -n aienv python=3.11
conda activate aienv
```

### 3. Install system dependencies

For macOS:

```bash
brew install tesseract poppler
```

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 6. Run the app

```bash
streamlit run app.py
```

## Deployment

To deploy on Streamlit Cloud:

1. Push the project to GitHub.
2. Create a new app in Streamlit Cloud.
3. Select the repository and branch.
4. Set the main file to `app.py`.
5. Add the following secret under app settings:

```env
GROQ_API_KEY = "your_key_here"
```

The project includes `packages.txt` for environment-level dependencies required during deployment.

## Notes

- The application uses the active workspace vector store instead of a stale saved index whenever one is available.
- The agent keeps a short conversation history to resolve follow-up references such as "it" or "tell me more".
- If the retrieved document context is insufficient, the agent can consult DuckDuckGo before responding.
- The answer layer formats content as clean Markdown and avoids HTML output in the final answer.

## Author

Mohammed Thaher S

- [LinkedIn](https://www.linkedin.com/in/mohammed-thaher-s/)
- [GitHub](https://github.com/MohammedThaher01)
