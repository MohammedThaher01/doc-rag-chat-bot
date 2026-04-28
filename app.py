import streamlit as st
import tempfile, os
from dotenv import load_dotenv
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from rag_pipeline import build_vectorstore, load_vectorstore, get_qa_chain

# Load environment variables at the very beginning
load_dotenv()

st.set_page_config(page_title="DocChat", page_icon="📄", layout="wide")
st.title("📄 Document RAG Chatbot")
st.caption("Upload documents and chat with them using Groq + LangChain + FAISS")

# --- Sidebar: document upload & ingestion ---
with st.sidebar:
    st.header("📂 Your Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs, TXTs, or DOCXs",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )
    if st.button("⚡ Process Documents", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one document.")
        else:
            with st.spinner("Chunking and embedding (OCR fallback enabled)..."):
                try:
                    tmp_paths = []
                    for f in uploaded_files:
                        suffix = os.path.splitext(f.name)[1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(f.read())
                            tmp_paths.append(tmp.name)
                    
                    vs, n_chunks = build_vectorstore(tmp_paths)
                    st.session_state.vectorstore = vs
                    
                    # Setup memory and chain
                    msgs = StreamlitChatMessageHistory(key="chat_messages")
                    msgs.clear() # Clear on new docs
                    st.session_state.qa_chain = get_qa_chain(vs, msgs)
                    
                    st.session_state.messages = []   # for UI display
                    st.success(f"✅ Indexed {n_chunks} chunks from {len(uploaded_files)} file(s)")
                except ValueError as e:
                    st.error(f"⚠️ {e}")
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")

    if os.path.exists("faiss_index") and "vectorstore" not in st.session_state:
        if st.button("🔄 Load Existing Index"):
            with st.spinner("Loading..."):
                vs = load_vectorstore()
                st.session_state.vectorstore = vs
                msgs = StreamlitChatMessageHistory(key="chat_messages")
                st.session_state.qa_chain = get_qa_chain(vs, msgs)
                st.success("Index loaded!")

# --- Chat interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Ensure the chain is always linked to the current Streamlit memory
if "vectorstore" in st.session_state:
    msgs = StreamlitChatMessageHistory(key="chat_messages")
    st.session_state.qa_chain = get_qa_chain(st.session_state.vectorstore, msgs)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask anything about your documents..."):
    if "qa_chain" not in st.session_state:
        st.warning("⬅️ Please upload and process documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.qa_chain({"question": prompt})
                answer = result["answer"]
                sources = result.get("source_documents", [])

            st.markdown(answer)

            # Show source attribution (great for portfolio demos)
            if sources:
                with st.expander("📚 Sources used"):
                    seen = set()
                    for doc in sources:
                        src = doc.metadata.get("source", "Unknown")
                        page = doc.metadata.get("page", "")
                        label = f"{os.path.basename(src)}" + (f" — page {page+1}" if page != "" else "")
                        if label not in seen:
                            st.caption(f"• {label}")
                            seen.add(label)

        st.session_state.messages.append({"role": "assistant", "content": answer})