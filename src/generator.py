"""
generator.py
────────────
Generates answers using Groq LLM with:
  • Retrieved context chunks
  • Rolling chat history memory (last N turns)
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from config import GROQ_MODEL, MAX_HISTORY_TURNS


SYSTEM_PROMPT = """You are an expert AI research assistant.

Rules:
1. Answer ONLY from the provided document context.
2. If the answer is not in the context say: "I could not find this information in the provided document(s)."
3. Be concise, clear, and structured. Use bullet points or numbered lists when helpful.
4. When referencing information, mention the source document name if available.
5. You may use the conversation history for context on follow-up questions."""


def _build_history_messages(chat_history: list[dict]) -> list:
    """Convert stored history dicts to LangChain message objects (last N turns)."""
    messages = []
    # keep only the last MAX_HISTORY_TURNS turns (each turn = user + assistant)
    recent = chat_history[-(MAX_HISTORY_TURNS * 2):]
    for msg in recent:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages


def generate_answer(
    query: str,
    chunks: list[dict],
    chat_history: list[dict] | None = None,
) -> str:
    """
    Args:
        query        : current user question
        chunks       : list of {"text":..., "source":..., "score":...} dicts
        chat_history : full conversation so far (list of {"role","content"} dicts)
    Returns:
        answer string
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables!")

    model = ChatGroq(
        api_key=api_key,
        model_name=GROQ_MODEL,
        temperature=0.7,
        max_tokens=1500,
    )

    # ── Build context block ────────────────────────────────────────────────────
    if chunks:
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            src  = chunk.get("source", "unknown")
            text = chunk.get("text", "")
            context_parts.append(f"[Chunk {i} | Source: {src}]\n{text}")
        context = "\n\n---\n\n".join(context_parts)
    else:
        context = "No relevant chunks found."

    # ── Assemble messages ──────────────────────────────────────────────────────
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    # inject chat history for memory
    if chat_history:
        messages.extend(_build_history_messages(chat_history))

    # current turn
    messages.append(HumanMessage(content=f"""Document Context:
{context}

Question: {query}

Answer:"""))

    response = model.invoke(messages)
    return response.content