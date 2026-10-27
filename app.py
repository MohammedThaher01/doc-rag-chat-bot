import html
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from agent import run_agent
from rag_pipeline import build_vectorstore, load_vectorstore

load_dotenv()

st.set_page_config(
    page_title="DocChat | Document workspace",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #17231f;
        --muted: #718078;
        --line: #e7ece8;
        --paper: #f7f8f6;
        --green: #23745a;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        color: var(--ink);
    }
    .stApp {
        background: var(--paper);
    }
    [data-testid="stHeader"] {
        background: transparent;
    }
    [data-testid="stAppViewContainer"], [data-testid="stMain"] {
        background: var(--paper) !important;
    }
    .stApp h1.hero-title, .stApp h2.welcome-title {
        color: var(--ink) !important;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 1180px;
        padding: 2.5rem 3.2rem 3rem;
    }
    [data-testid="stSidebar"] {
        background: #172c25;
        border-right: 1px solid rgba(255,255,255,.07);
    }
    [data-testid="stSidebar"] > div:first-child {
        padding: 1.65rem 1.25rem 1.4rem;
    }
    [data-testid="stSidebar"] * {
        color: #eef4f0;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #b7c8bf;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section {
        background: #203a30;
        border: 1px dashed #527363;
        border-radius: 12px;
        padding: 1rem .75rem;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] section:hover {
        border-color: #9dc8ae;
    }
    [data-testid="stSidebar"] [data-testid="stFileChip"] {
        background: #29483b;
        border: 1px solid rgba(202, 226, 211, 0.2);
        border-radius: 9px;
        padding: 0.45rem 0.55rem;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileChipName"] {
        color: #f0f5f1 !important;
        font-size: 0.75rem;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] [data-testid="stFileChip"] small button {
        background: transparent !important;
        border-radius: 6px;
        color: #c1d1c8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileChip"] small button * {
        color: #c1d1c8 !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileChip"] small button:hover {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #fff !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button {
        background: #eef4f0;
        color: #173c2c;
        border: 0;
        border-radius: 7px;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] button * {
        color: #173c2c !important;
    }
    [data-testid="stSidebar"] [data-testid="stFileUploader"] small,
    [data-testid="stSidebar"] [data-testid="stFileUploader"] span {
        color: #c1d1c8;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button {
        background: #203a30;
        color: #eef4f0;
        min-height: 2.8rem;
        border: 1px solid rgba(255,255,255,.15);
        border-radius: 9px;
        font-weight: 600;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] {
        background: #a7d6b7;
        color: #173c2c !important;
        border: 0;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] *,
    [data-testid="stSidebar"] [data-testid="stButton"] button * {
        color: inherit !important;
    }
    [data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"]:hover {
        background: #bce2c8;
        color: #173c2c;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(255, 255, 255, 0.13);
    }
    [data-testid="stSidebar"] [data-testid="stAlert"] * {
        color: inherit;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 11px;
        margin: 0 0 2.1rem;
    }
    .brand-mark {
        display: grid;
        place-items: center;
        width: 38px;
        height: 38px;
        border-radius: 11px;
        background: #a7d6b7;
        color: #173c2c;
        font: 800 17px 'Manrope', sans-serif;
        letter-spacing: -1px;
    }
    .brand-name {
        color: #f4f7f4;
        font: 700 17px 'Manrope', sans-serif;
        letter-spacing: -.5px;
    }
    .brand-caption {
        color: #9db2a6;
        font-size: 11px;
        margin-top: 1px;
    }
    .side-label {
        color: #93aa9d;
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.15px;
        margin: 0 0 9px;
        text-transform: uppercase;
    }
    .side-heading {
        color: #f3f6f3;
        font: 700 15px 'Manrope', sans-serif;
        margin: 0 0 5px;
    }
    .side-copy {
        color: #aebfb5;
        font-size: 12px;
        line-height: 1.55;
        margin: 0 0 15px;
    }
    .file-row {
        align-items: center;
        background: rgba(255,255,255,.055);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 8px;
        color: #e1ebe4;
        display: flex;
        font-size: 11px;
        gap: 8px;
        margin: 6px 0;
        padding: 8px 9px;
        overflow-wrap: anywhere;
        white-space: normal;
    }
    .file-row .file-dot {
        align-self: flex-start;
        margin-top: 4px;
    }
    .file-dot {
        background: #a7d6b7;
        border-radius: 3px;
        flex: 0 0 7px;
        height: 7px;
    }
    .privacy-note {
        border-top: 1px solid rgba(255,255,255,.12);
        color: #98ada1;
        font-size: 11px;
        line-height: 1.55;
        margin-top: 1.6rem;
        padding-top: 1rem;
    }
    .privacy-note strong {
        color: #dce8df;
        font-weight: 600;
    }

    .topline {
        align-items: center;
        color: #728078;
        display: flex;
        font-size: 11px;
        font-weight: 700;
        gap: 8px;
        letter-spacing: 1.15px;
        margin: 0 0 19px;
        text-transform: uppercase;
    }
    .topline-mark {
        background: var(--green);
        border-radius: 50%;
        height: 7px;
        width: 7px;
    }
    .hero-title {
        color: #17231f;
        font: 700 clamp(30px, 4vw, 43px)/1.14 'Manrope', sans-serif;
        letter-spacing: -1.7px;
        margin: 0;
        max-width: 760px;
    }
    .hero-copy {
        color: #718078;
        font-size: 15px;
        line-height: 1.65;
        margin: 14px 0 0;
        max-width: 620px;
    }
    .workspace-bar {
        align-items: center;
        border-bottom: 1px solid var(--line);
        display: flex;
        gap: 10px;
        justify-content: space-between;
        margin: 25px 0 27px;
        padding: 0 0 16px;
    }
    .workspace-label {
        color: #748179;
        font-size: 12px;
    }
    .status-pill {
        align-items: center;
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 999px;
        color: #68776e;
        display: inline-flex;
        font-size: 11px;
        font-weight: 600;
        gap: 7px;
        padding: 7px 11px;
    }
    .status-pill.ready {
        background: #edf6ef;
        border-color: #d8e9dc;
        color: #286047;
    }
    .status-dot {
        background: #a4b0a9;
        border-radius: 50%;
        height: 7px;
        width: 7px;
    }
    .status-pill.ready .status-dot {
        background: #3a9364;
    }
    .welcome-card {
        background: #fff;
        border: 1px solid var(--line);
        border-radius: 16px;
        box-shadow: 0 8px 28px rgba(27, 49, 38, .035);
        margin-top: 4px;
        padding: 27px 28px 25px;
    }
    .welcome-eyebrow {
        color: var(--green);
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.1px;
        margin: 0 0 7px;
        text-transform: uppercase;
    }
    .welcome-title {
        color: var(--ink);
        font: 700 19px 'Manrope', sans-serif;
        letter-spacing: -.45px;
        margin: 0;
    }
    .welcome-copy {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.6;
        margin: 7px 0 20px;
    }
    .suggestion-grid {
        display: grid;
        gap: 11px;
        grid-template-columns: repeat(3, 1fr);
    }
    .suggestion {
        background: #f8faf8;
        border: 1px solid #e8eee9;
        border-radius: 10px;
        min-height: 84px;
        padding: 13px 14px;
    }
    .suggestion-icon {
        color: var(--green);
        font-size: 14px;
        margin-bottom: 8px;
    }
    .suggestion-title {
        color: #33423a;
        font-size: 12px;
        font-weight: 600;
    }
    .suggestion-copy {
        color: #89958e;
        font-size: 11px;
        line-height: 1.45;
        margin-top: 3px;
    }
    .chat-divider {
        align-items: center;
        color: #98a39c;
        display: flex;
        font-size: 10px;
        font-weight: 700;
        gap: 11px;
        letter-spacing: 1px;
        margin: 25px 0 12px;
        text-transform: uppercase;
    }
    .chat-divider:after {
        background: var(--line);
        content: "";
        flex: 1;
        height: 1px;
    }
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: 0 !important;
        border-radius: 0;
        box-shadow: none !important;
        margin-bottom: 18px;
        padding: 0 !important;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        display: flex;
        justify-content: flex-end;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
    > div:has([data-testid="stMarkdownContainer"]) {
        background: #e7f1e9;
        border: 1px solid #d5e5d8;
        border-radius: 16px 16px 4px 16px;
        box-sizing: border-box;
        flex: 0 1 auto;
        margin-left: auto;
        max-width: min(80%, 760px);
        padding: 11px 15px;
        width: fit-content;
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"],
    [data-testid="stChatMessage"] [data-testid^="chatAvatarIcon-"],
    [data-testid="stChatMessage"] [data-testid="stIconMaterial"],
    [data-testid="stChatMessage"] .stChatMessageAvatar {
        display: none !important;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p {
        font-size: 14px;
        line-height: 1.7;
        margin: 0 0 0.8rem;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p:last-child {
        margin-bottom: 0;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3 {
        color: #24352c;
        font-family: 'Manrope', sans-serif;
        font-weight: 700;
        line-height: 1.35;
        margin: 1.1rem 0 0.55rem;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h1 {
        font-size: 1.35rem;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h2 {
        font-size: 1.2rem;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] h3 {
        font-size: 1.05rem;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ul,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] ol {
        margin: 0.35rem 0 0.85rem;
        padding-left: 1.4rem;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li {
        line-height: 1.65;
        padding-left: 0.15rem;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li + li {
        margin-top: 0.25rem;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] blockquote {
        background: rgba(255, 255, 255, 0.5);
        border-left: 3px solid #8caf98;
        border-radius: 0 7px 7px 0;
        color: #52635a;
        margin: 0.8rem 0;
        padding: 0.55rem 0.9rem;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre {
        background: #202a26;
        border: 1px solid rgba(25, 45, 35, 0.12);
        border-radius: 9px;
        margin: 0.75rem 0;
        max-width: 100%;
        overflow-x: auto;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] pre code {
        color: #e4eee8;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
        font-size: 0.88em;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] :not(pre) > code {
        background: rgba(35, 116, 90, 0.09);
        border-radius: 4px;
        color: #245c43;
        padding: 0.12rem 0.3rem;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] table {
        border-collapse: collapse;
        display: block;
        margin: 0.75rem 0;
        max-width: 100%;
        overflow-x: auto;
        width: max-content;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] td {
        border: 1px solid #dce4de;
        padding: 0.5rem 0.7rem;
        text-align: left;
        vertical-align: top;
    }
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] th {
        background: rgba(255, 255, 255, 0.65);
        color: #34463b;
        font-weight: 700;
    }
    .source-label {
        color: #78867e;
        font-size: 11px;
        margin-top: 7px;
    }
    [data-testid="stChatInput"] {
        background: transparent !important;
        border: 0 !important;
        border-radius: 12px;
        box-shadow: none !important;
    }
    [data-testid="stChatInputTextArea"] {
        background: #edf4ef !important;
        border: 1px solid #cfddd2 !important;
        border-radius: 12px;
        color: var(--ink) !important;
        font-size: 14px !important;
        line-height: 1.55;
        min-height: 56px;
        padding: 0.8rem 1rem;
        transition: border-color 150ms ease, box-shadow 150ms ease;
    }
    [data-testid="stChatInputTextArea"]:focus {
        border-color: #6a9c7a !important;
        box-shadow: 0 0 0 3px rgba(72, 133, 95, 0.14);
    }
    [data-testid="stChatInputTextArea"]::placeholder {
        color: #7d8b82;
    }
    [data-testid="stBottom"] {
        background: var(--paper) !important;
    }
    [data-testid="stBottom"] > div {
        background: transparent !important;
    }
    [data-testid="stBottom"] [data-testid="stChatInput"] button {
        background: var(--green) !important;
        border-radius: 9px;
    }
    [data-testid="stMainBlockContainer"] [data-testid="stAlert"] {
        border-radius: 10px;
    }
    @media (max-width: 760px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1.6rem 1.15rem 2rem;
        }
        .hero-title {
            letter-spacing: -1px;
        }
        .suggestion-grid {
            grid-template-columns: 1fr;
        }
        .suggestion {
            min-height: auto;
        }
        .welcome-card {
            padding: 21px 18px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">D</div>
            <div>
                <div class="brand-name">DocChat</div>
                <div class="brand-caption">Your document workspace</div>
            </div>
        </div>
        <div class="side-label">Workspace</div>
        <div class="side-heading">Add your documents</div>
        <div class="side-copy">
            Bring your files together and get clear answers, grounded in their contents.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Upload PDFs, text files, or Word documents",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed",
        help="Supported formats: PDF, TXT, DOCX",
    )

    if uploaded_files:
        file_rows = "".join(
            '<div class="file-row"><span class="file-dot"></span>'
            f"{html.escape(file.name)}</div>"
            for file in uploaded_files
        )
        st.markdown(file_rows, unsafe_allow_html=True)
        file_count = len(uploaded_files)
        file_label = "file" if file_count == 1 else "files"
        st.caption(f"{file_count} {file_label} selected")

    if st.button("Process documents", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Choose at least one document to continue.")
        else:
            with st.spinner("Reading and indexing your documents..."):
                try:
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_paths = []
                        for file in uploaded_files:
                            suffix = os.path.splitext(file.name)[1]
                            path = os.path.join(
                                temp_dir,
                                f"document_{len(temp_paths)}{suffix}",
                            )
                            with open(path, "wb") as temp_file:
                                temp_file.write(file.getvalue())
                            temp_paths.append(path)
                        vectorstore, chunk_count = build_vectorstore(temp_paths)

                    st.session_state.vectorstore = vectorstore
                    st.session_state.indexed_files = [file.name for file in uploaded_files]
                    st.session_state.chunk_count = chunk_count
                    st.session_state.messages = []
                    st.success(
                        f"Ready to chat. Indexed {chunk_count} sections from "
                        f"{len(uploaded_files)} file(s)."
                    )
                except ValueError as error:
                    st.error(str(error))
                except Exception as error:
                    st.error(f"Could not process the documents: {error}")

    index_exists = os.path.exists("faiss_index")
    if index_exists and "vectorstore" not in st.session_state:
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        if st.button("Load saved workspace", use_container_width=True):
            with st.spinner("Loading your saved workspace..."):
                try:
                    st.session_state.vectorstore = load_vectorstore()
                    st.session_state.indexed_files = []
                    st.success("Saved workspace loaded.")
                except Exception as error:
                    st.error(f"Could not load the saved workspace: {error}")

    if "vectorstore" in st.session_state:
        st.markdown("---")
        st.markdown(
            '<div class="side-label">In this workspace</div>',
            unsafe_allow_html=True,
        )
        indexed_files = st.session_state.get("indexed_files", [])
        if indexed_files:
            for filename in indexed_files:
                st.markdown(
                    '<div class="file-row"><span class="file-dot"></span>'
                    f"{html.escape(filename)}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="side-copy">Saved document index is ready to search.</div>',
                unsafe_allow_html=True,
            )
        chunk_count = st.session_state.get("chunk_count")
        if chunk_count:
            st.caption(f"{chunk_count} searchable sections")

    st.markdown(
        """
        <div class="privacy-note">
            <strong>Made for focused work.</strong><br>
            Ask questions, explore ideas, and find the details that matter.
        </div>
        """,
        unsafe_allow_html=True,
    )

is_ready = "vectorstore" in st.session_state
indexed_files = st.session_state.get("indexed_files", [])
if is_ready and indexed_files:
    document_count = len(indexed_files)
    document_label = "document" if document_count == 1 else "documents"
    workspace_text = f"{document_count} {document_label} in this workspace"
elif is_ready:
    workspace_text = "Saved document workspace"
else:
    workspace_text = "No documents added yet"

st.markdown(
    """
    <div class="topline"><span class="topline-mark"></span> A calmer way to find what matters</div>
    <h1 class="hero-title">Make more of what you know.</h1>
    <p class="hero-copy">
        Bring your documents into one thoughtful workspace. Ask a question and get a clear answer
        grounded in the information you already have.
    </p>
    """,
    unsafe_allow_html=True,
)

status_class = "status-pill ready" if is_ready else "status-pill"
status_label = "Workspace ready" if is_ready else "Awaiting documents"
st.markdown(
    f"""
    <div class="workspace-bar">
        <span class="workspace-label">{html.escape(workspace_text)}</span>
        <span class="{status_class}"><span class="status-dot"></span>{status_label}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    if is_ready:
        welcome_title = "Your documents are ready."
        welcome_copy = (
            "Start with a question. DocChat will look through your documents "
            "and help you find a useful answer."
        )
    else:
        welcome_title = "A good question is a great place to start."
        welcome_copy = (
            "Add a document from the left to get started. Then ask for a summary, "
            "a specific detail, or a connection between ideas."
        )

    st.markdown(
        f"""
        <div class="welcome-card">
            <div class="welcome-eyebrow">Your workspace</div>
            <h2 class="welcome-title">{welcome_title}</h2>
            <p class="welcome-copy">{welcome_copy}</p>
            <div class="suggestion-grid">
                <div class="suggestion">
                    <div class="suggestion-icon">↗</div>
                    <div class="suggestion-title">Get the overview</div>
                    <div class="suggestion-copy">Summarize the key ideas and takeaways.</div>
                </div>
                <div class="suggestion">
                    <div class="suggestion-icon">⌕</div>
                    <div class="suggestion-title">Find a detail</div>
                    <div class="suggestion-copy">Look up a specific fact, date, or figure.</div>
                </div>
                <div class="suggestion">
                    <div class="suggestion-icon">◇</div>
                    <div class="suggestion-title">Make a connection</div>
                    <div class="suggestion-copy">Compare ideas across your documents.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown('<div class="chat-divider">Conversation</div>', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("source"):
                source_label = (
                    "Web search" if message["source"] == "web" else "Your documents"
                )
                st.markdown(
                    f'<div class="source-label">Source · {source_label}</div>',
                    unsafe_allow_html=True,
                )

if prompt := st.chat_input("Ask a question about your documents..."):
    if not is_ready:
        st.warning(
            "Add and process your documents in the workspace panel "
            "before asking a question."
        )
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Looking through your workspace..."):
                try:
                    result = run_agent(prompt)
                    answer = result["answer"]
                    source = result["source"]
                    st.markdown(answer)
                    source_label = "Web search" if source == "web" else "Your documents"
                    st.markdown(
                        f'<div class="source-label">Source · {source_label}</div>',
                        unsafe_allow_html=True,
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "source": source}
                    )
                except Exception as error:
                    st.error(f"Could not complete your answer: {error}")
