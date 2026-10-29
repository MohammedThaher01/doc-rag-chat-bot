from __future__ import annotations

import re
from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS
from langchain.tools import tool

FAISS_PATH = "faiss_index"
EMBED_MODEL = "all-MiniLM-L6-v2"

search = DuckDuckGoSearchRun()


def _load_saved_vectorstore() -> FAISS:
    """Load the persisted index only when a caller has no session index."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _store_size(vectorstore: Any) -> int:
    try:
        return int(vectorstore.index.ntotal)
    except (AttributeError, TypeError, ValueError):
        return 0


def _all_documents(vectorstore: Any) -> list[Any]:
    """Return indexed chunks in original index order for small-document questions.

    This uses FAISS's public mapping/docstore attributes, not similarity ranking.
    It is intentionally only selected for small indexes, where including every
    chunk gives reliable whole-document summaries and complete project lists.
    """
    mapping = getattr(vectorstore, "index_to_docstore_id", None)
    docstore = getattr(vectorstore, "docstore", None)
    if not mapping or docstore is None:
        return []

    documents: list[Any] = []
    seen: set[tuple[str, str]] = set()
    for index in sorted(mapping):
        try:
            doc = docstore.search(mapping[index])
        except Exception:
            continue
        if not hasattr(doc, "page_content"):
            continue
        content = str(doc.page_content or "").strip()
        if not content:
            continue
        metadata = getattr(doc, "metadata", {}) or {}
        key = (re.sub(r"\s+", " ", content), str(metadata.get("source", "")))
        if key in seen:
            continue
        seen.add(key)
        documents.append(doc)
    return documents


def _query_type(query: str) -> str:
    text = query.lower()
    project_terms = ("project", "portfolio", "hackathon", "recognition", "achievement")
    overview_terms = (
        "resume", "cv", "document", "file", "summari", "overview",
        "what is this about", "tell me about", "key points", "main points",
        "give me an overview", "whole document", "entire document",
    )
    if any(term in text for term in project_terms):
        return "projects"
    if any(term in text for term in overview_terms):
        return "overview"
    return "specific"


def _query_variants(query: str, kind: str) -> list[str]:
    """Expand broad queries so each major section can contribute evidence."""
    variants = [query]
    if kind == "overview":
        variants.extend([
            "document summary main sections key facts and important details",
            "projects project names descriptions technologies and accomplishments",
            "work experience internship responsibilities and achievements",
            "skills education certifications dates metrics awards and recognitions",
        ])
    elif kind == "projects":
        variants.extend([
            "project titles names descriptions technologies and accomplishments",
            "hackathons competitions awards recognitions and technical solutions",
            "project results metrics tools and implementation details",
        ])
    return variants


def _append_unique(target: list[Any], candidates: list[Any], seen: set[str]) -> None:
    for doc in candidates:
        content = re.sub(r"\s+", " ", str(getattr(doc, "page_content", ""))).strip()
        if not content:
            continue
        # Similarity chunks may overlap; content-based deduplication avoids
        # feeding the same passage multiple times to the LLM.
        key = content
        if key not in seen:
            seen.add(key)
            target.append(doc)


def search_documents(
    query: str,
    vectorstore: Any | None = None,
    *,
    max_context_chars: int = 18000,
) -> str:
    """Retrieve context from the active workspace, not another session's index.

    Broad summary/project questions search more than the two nearest chunks.
    For small indexes we include every chunk, so the agent does not claim an
    incomplete subset is the entire document. Specific questions use semantic
    search with several complementary queries.
    """
    try:
        store = vectorstore if vectorstore is not None else _load_saved_vectorstore()
        total_chunks = _store_size(store)
        kind = _query_type(query)

        # Most resumes and short uploaded documents fit in the small-index path.
        # Reading every chunk is more reliable than relying on top-k similarity
        # to enumerate every project, award, or section.
        include_all_limit = 20 if kind in {"overview", "projects"} else 0
        if total_chunks and total_chunks <= include_all_limit:
            docs = _all_documents(store)
        else:
            docs = []
            seen: set[str] = set()
            variants = _query_variants(query, kind)
            per_query_k = 5 if kind in {"overview", "projects"} else 7
            for variant in variants:
                try:
                    results = store.similarity_search(variant, k=per_query_k)
                except Exception:
                    results = []
                _append_unique(docs, results, seen)
                # Cap the number of chunks before assembling context.
                if len(docs) >= 18:
                    break

            # When a broad question targets a larger collection, also include
            # a few leading chunks from the original query if expansion found few.
            if not docs:
                docs = store.similarity_search(query, k=8)

        # If a vector-store implementation exposes no readable docstore mapping,
        # fall back to semantic search instead of silently returning no context.
        if not docs:
            docs = []
            seen: set[str] = set()
            for variant in _query_variants(query, kind):
                try:
                    results = store.similarity_search(
                        variant,
                        k=5 if kind in {"overview", "projects"} else 7,
                    )
                except Exception:
                    results = []
                _append_unique(docs, results, seen)
                if len(docs) >= 18:
                    break

        if not docs:
            return "NO_RELEVANT_CONTEXT"

        parts: list[str] = []
        used = 0
        for index, doc in enumerate(docs, start=1):
            content = str(getattr(doc, "page_content", "")).strip()
            if not content:
                continue
            excerpt = f"[Retrieved excerpt {index}]\n{content}"
            if used + len(excerpt) > max_context_chars:
                remaining = max_context_chars - used
                if remaining > 300:
                    parts.append(excerpt[:remaining])
                break
            parts.append(excerpt)
            used += len(excerpt)

        return "\n\n".join(parts) if parts else "NO_RELEVANT_CONTEXT"
    except Exception:
        return "NO_RELEVANT_CONTEXT"


@tool
def web_search(query: str) -> str:
    """Search the web for information not found in uploaded documents."""
    return search.run(query)


@tool
def rag_search(query: str) -> str:
    """Search all relevant uploaded-document sections for grounded context."""
    return search_documents(query)
