"""
vector_store.py
---------------
Embedding Generation + FAISS Vector Storage
- Chunks ko embeddings mein convert karo (numbers)
- FAISS mein save karo
- Baad mein load kar sako
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL, FAISS_INDEX_PATH


def get_embedding_model():
    """HuggingFace free embedding model load karo"""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return embeddings


def create_vector_store(chunks: list):
    """
    Chunks lo → embeddings banao → FAISS mein store karo → disk pe save karo
    """
    embeddings = get_embedding_model()

    # FAISS vector store banao
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Disk pe save karo taake baar baar process na karna pade
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)

    return vectorstore


def load_vector_store():
    """
    Pehle se saved FAISS index load karo
    """
    embeddings = get_embedding_model()
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def index_exists() -> bool:
    """Check karo FAISS index already bana hua hai ya nahi"""
    return os.path.exists(FAISS_INDEX_PATH)