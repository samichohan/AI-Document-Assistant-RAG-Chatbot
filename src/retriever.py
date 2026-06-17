"""
retriever.py
────────────
Similarity search with source-level metadata.
Works with both FAISS and Pinecone stores transparently.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from vector_store import load_vector_store
from config import TOP_K


def retrieve_chunks(
    query: str,
    vectorstore=None,
    top_k: int = TOP_K,
    source_filter: str | None = None,
) -> list[dict]:
    """
    Returns a list of dicts:
      { "text": str, "source": str, "score": float }

    Args:
        query        : user question
        vectorstore  : pre-loaded store; loads from disk/cloud if None
        top_k        : number of chunks to retrieve
        source_filter: optional filename to restrict search to one document
    """
    if vectorstore is None:
        vectorstore = load_vector_store()

    results_with_scores = vectorstore.similarity_search_with_score(query, k=top_k * 2)

    chunks = []
    seen   = set()

    for doc, score in results_with_scores:
        src  = doc.metadata.get("source", "unknown")
        text = doc.page_content.strip()

        # optional single-doc filter
        if source_filter and source_filter not in src:
            continue

        # deduplicate identical text blocks
        if text in seen:
            continue
        seen.add(text)

        chunks.append({
            "text"  : text,
            "source": os.path.basename(src),
            "score" : float(score),
        })

        if len(chunks) >= top_k:
            break

    return chunks


def retrieve_text_only(query: str, vectorstore=None, top_k: int = TOP_K) -> list[str]:
    """Backwards-compatible helper — returns plain text list."""
    return [c["text"] for c in retrieve_chunks(query, vectorstore, top_k)]