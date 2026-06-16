"""
generator.py
------------
Answer Generation using Groq LLM
- Relevant chunks + user query lo
- Groq LLM ko context de ke answer generate karo
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from config import GROQ_API_KEY, GROQ_MODEL


def get_llm():
    """Groq LLM model initialize karo"""
    model = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=GROQ_MODEL,
        temperature=0.7,
        max_tokens=1000
    )
    return model


def generate_answer(query: str, chunks: list) -> str:
    """
    Query + relevant chunks lo → LLM se answer generate karo

    Args:
        query: User ka sawaal
        chunks: FAISS se retrieve kiye gaye relevant text chunks

    Returns:
        LLM ka generated answer
    """
    model = get_llm()

    # Chunks ko ek context string mein jodo
    context = "\n\n".join([f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)])

    messages = [
        SystemMessage(content="""You are a helpful AI assistant. 
Answer the user's question based ONLY on the provided context. 
If the answer is not in the context, say 'I could not find this information in the provided document.'
Be clear and concise."""),

        HumanMessage(content=f"""Context:
{context}

Question: {query}

Answer:""")
    ]

    response = model.invoke(messages)
    return response.content