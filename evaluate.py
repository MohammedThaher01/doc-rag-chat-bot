"""
Lightweight RAG evaluation for resume-worthy metrics.

Measures:
  - Vector retrieval latency (FAISS top-k)
  - Embedding throughput (chunks/sec)
  - Index statistics (#chunks, embedding dim)
  - End-to-end RAG response latency (requires GROQ_API_KEY)
  - Retrieval relevance hit-rate (keyword overlap heuristic)

Run:
  python evaluate.py
"""
import os
import time
import statistics
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

FAISS_PATH = "faiss_index"
EMBED_MODEL = "all-MiniLM-L6-v2"

SAMPLE_QUERIES = [
    "What is the main topic of the documents?",
    "Summarize the key findings.",
    "Who is the intended audience?",
    "What are the main conclusions?",
    "List the important details mentioned.",
]

N_RETRIEVAL_RUNS = 20
N_EMBED_SAMPLES = 100

def load_vs():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(FAISS_PATH, embeddings,
                            allow_dangerous_deserialization=True)

def benchmark_retrieval(vs):
    times = []
    for q in SAMPLE_QUERIES:
        for _ in range(N_RETRIEVAL_RUNS):
            t0 = time.perf_counter()
            vs.similarity_search(q, k=2)
            times.append((time.perf_counter() - t0) * 1000)  # ms
    return {
        "retrieval_mean_ms": round(statistics.mean(times), 2),
        "retrieval_p95_ms": round(sorted(times)[int(len(times) * 0.95)], 2),
        "retrieval_runs": len(times),
    }

def benchmark_embeddings():
    emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    samples = [
        "The quick brown fox jumps over the lazy dog. " * 10
        for _ in range(N_EMBED_SAMPLES)
    ]
    t0 = time.perf_counter()
    emb.embed_documents(samples)
    elapsed = time.perf_counter() - t0
    return {
        "embedding_throughput_chunks_per_sec": round(N_EMBED_SAMPLES / elapsed, 1),
        "embedding_dim": len(emb.embed_query("test")),
    }

def measure_end_to_end():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {"e2e_latency_sec": None, "note": "GROQ_API_KEY not set — skipped."}
    try:
        from agent import run_agent
    except Exception as e:
        return {"e2e_latency_sec": None, "note": f"agent import failed: {e}"}

    times = []
    for q in SAMPLE_QUERIES[:3]:
        t0 = time.perf_counter()
        run_agent(q)
        times.append(time.perf_counter() - t0)
    return {
        "e2e_latency_mean_sec": round(statistics.mean(times), 2),
        "e2e_latency_runs": len(times),
    }

def retrieval_relevance(vs):
    """Keyword-overlap heuristic: does a top-2 chunk share any
    non-stopword token with the query?"""
    stop = set("the a an of to and or in is it for on with by this that are was be as at".split())
    hits = 0
    for q in SAMPLE_QUERIES:
        q_tokens = set(t.lower().strip(".,?") for t in q.split()) - stop
        docs = vs.similarity_search(q, k=2)
        for d in docs:
            d_tokens = set(t.lower().strip(".,?") for t in d.page_content.split())
            if q_tokens & d_tokens:
                hits += 1
                break
    return {"retrieval_hit_rate_pct": round(100 * hits / len(SAMPLE_QUERIES), 1)}

def index_stats(vs):
    n = vs.index.ntotal
    return {"indexed_chunks": n, "vector_dim": vs.index.d}

def main():
    print("=" * 60)
    print("  DocChat RAG Pipeline — Evaluation Report")
    print("=" * 60)

    vs = load_vs()

    results = {}
    results.update(index_stats(vs))
    results.update(benchmark_embeddings())
    results.update(benchmark_retrieval(vs))
    results.update(retrieval_relevance(vs))
    results.update(measure_end_to_end())

    print()
    for k, v in results.items():
        print(f"  {k:40s} {v}")
    print()
    print("=" * 60)

    # Resume-friendly headline
    chunks = results.get("indexed_chunks")
    ret_ms = results.get("retrieval_mean_ms")
    e2e = results.get("e2e_latency_mean_sec")
    line = "Headline for resume: "
    if ret_ms:
        line += f"sub-{ret_ms}ms vector retrieval "
        if chunks:
            line += f"over {chunks} chunks"
    if e2e:
        line += f" | {e2e}s end-to-end RAG response"
    print(line)
    print("=" * 60)

if __name__ == "__main__":
    main()