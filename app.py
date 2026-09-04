import streamlit as st
import tempfile, os
from dotenv import load_dotenv
from rag_pipeline import build_vectorstore, load_vectorstore, get_qa_chain

load_dotenv()

st.set_page_config(page_title="DocChat", page_icon="📄", layout="wide")
st.title("📄 Document RAG Chatbot")
st.caption("Upload documents and chat with them using Groq + LangChain + FAISS")

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
            with st.spinner("Chunking and embedding..."):
                try:
                    tmp_paths = []
                    for f in uploaded_files:
                        suffix = os.path.splitext(f.name)[1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(f.read())
                            tmp_paths.append(tmp.name)
                    vs, n_chunks = build_vectorstore(tmp_paths)
                    st.session_state.vectorstore = vs
                    st.session_state.qa_chain = get_qa_chain(vs)
                    st.session_state.messages = []
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
                st.session_state.qa_chain = get_qa_chain(vs)
                st.success("Index loaded!")

if "messages" not in st.session_state:
    st.session_state.messages = []

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
                answer = st.session_state.qa_chain.invoke(prompt)
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
