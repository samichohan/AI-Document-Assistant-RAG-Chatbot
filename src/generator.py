import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


def generate_answer(query: str, chunks: list) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables!")

    model = ChatGroq(
        api_key=api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=1000
    )

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