"""
vector_store.py
───────────────
Supports:
  • Local  → FAISS  (default, works offline)
  • Cloud  → Pinecone  (set PINECONE_API_KEY env var to enable)

Multi-document search is supported: every chunk keeps its "source" metadata,
so retrieval results can be filtered or labelled by document.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    USE_PINECONE,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_DIMENSION,
)


# ── Shared embedding model (loaded once) ──────────────────────────────────────

_embedding_model = None

def get_embedding_model() -> HuggingFaceEmbeddings:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _embedding_model


# ── FAISS (local) ─────────────────────────────────────────────────────────────

def create_faiss_store(chunks: list[Document]) -> FAISS:
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore


def load_faiss_store() -> FAISS:
    embeddings = get_embedding_model()
    return FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def faiss_index_exists() -> bool:
    return os.path.exists(FAISS_INDEX_PATH)


def add_chunks_to_faiss(new_chunks: list[Document], existing_store: FAISS) -> FAISS:
    """Merge new chunks into an already-loaded FAISS store (multi-doc)."""
    embeddings = get_embedding_model()
    new_store   = FAISS.from_documents(new_chunks, embeddings)
    existing_store.merge_from(new_store)
    existing_store.save_local(FAISS_INDEX_PATH)
    return existing_store


# ── Pinecone (cloud) ──────────────────────────────────────────────────────────

def _get_pinecone_store():
    """Return a LangChain Pinecone vectorstore (creates index if needed)."""
    try:
        from pinecone import Pinecone, ServerlessSpec
        from langchain_pinecone import PineconeVectorStore
    except ImportError:
        raise ImportError(
            "pinecone-client and langchain-pinecone are required for cloud support. "
            "Run: pip install pinecone-client langchain-pinecone"
        )

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=PINECONE_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    embeddings = get_embedding_model()
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
    )


def create_pinecone_store(chunks: list[Document]):
    store = _get_pinecone_store()
    store.add_documents(chunks)
    return store


# ── Unified public API ────────────────────────────────────────────────────────

def create_vector_store(chunks: list[Document], existing_store=None):
    """
    Create (or extend) a vector store.
    - If USE_PINECONE=True  → Pinecone cloud
    - Otherwise             → FAISS local
    existing_store: pass the current FAISS store to ADD new docs (multi-doc).
    """
    if USE_PINECONE:
        return create_pinecone_store(chunks)

    if existing_store is not None:
        # multi-doc: merge into existing FAISS
        return add_chunks_to_faiss(chunks, existing_store)

    return create_faiss_store(chunks)


def load_vector_store():
    if USE_PINECONE:
        return _get_pinecone_store()
    return load_faiss_store()


def index_exists() -> bool:
    if USE_PINECONE:
        return True          # Pinecone index is always ready
    return faiss_index_exists()