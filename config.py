import os
from dotenv import load_dotenv

load_dotenv()

# Groq LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Embedding Model (free, local - no API key needed)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Retriever - kitne chunks wapas lao
TOP_K = 4

# FAISS index save location
FAISS_INDEX_PATH = "/tmp/faiss_index"