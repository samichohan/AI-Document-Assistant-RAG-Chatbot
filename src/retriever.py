"""
retriever.py
------------
Retriever Component
- User query lo
- FAISS mein se sabse similar chunks dhundo
- Top K chunks return karo
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.vector_store import load_vector_store
from config import TOP_K


def retrieve_chunks(query: str, vectorstore=None) -> list:
    """
    Query ke based pe most relevant chunks dhundo

    Args:
        query: User ka sawaal
        vectorstore: Pehle se loaded vectorstore (optional)

    Returns:
        List of relevant text chunks
    """
    if vectorstore is None:
        vectorstore = load_vector_store()

    # FAISS mein similarity search karo
    results = vectorstore.similarity_search(query, k=TOP_K)

    # Sirf text content return karo
    chunks = [doc.page_content for doc in results]
    return chunks