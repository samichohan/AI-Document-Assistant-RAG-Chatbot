import os
from dotenv import load_dotenv

load_dotenv()

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ───────────────────────────────────────────────────────────────────────
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_MODEL     = "llama-3.3-70b-versatile"

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K = 4

# ── Local Vector Store ────────────────────────────────────────────────────────
FAISS_INDEX_PATH = "/tmp/faiss_index"

# ── Cloud Vector Store (Pinecone) ─────────────────────────────────────────────
PINECONE_API_KEY     = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME  = os.getenv("PINECONE_INDEX_NAME", "rag-documents")
PINECONE_DIMENSION   = 384          # all-MiniLM-L6-v2 output dim
USE_PINECONE         = bool(PINECONE_API_KEY)

# ── OCR ───────────────────────────────────────────────────────────────────────
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")   # override if needed

# ── Chat History ──────────────────────────────────────────────────────────────
MAX_HISTORY_TURNS = 6          # last N turns sent to LLM as context