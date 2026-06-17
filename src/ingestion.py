"""
ingestion.py
────────────
Supports:
  • PDF  → PyPDFLoader
  • TXT  → plain read
  • PNG / JPG / JPEG / BMP / TIFF → EasyOCR (works Windows + Streamlit Cloud)
Returns a list of LangChain Document chunks ready for embedding.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_pdf(file_path: str) -> list[Document]:
    loader = PyPDFLoader(file_path)
    return loader.load()


def load_text(file_path: str) -> list[Document]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return [Document(page_content=text, metadata={"source": file_path})]


def load_image_ocr(file_path: str) -> list[Document]:
    """
    Extract text from image using EasyOCR.
    Works on Windows local + Streamlit Cloud — no system install needed.
    """
    try:
        import easyocr
    except ImportError:
        raise ImportError("Run: pip install easyocr")

    reader  = easyocr.Reader(['en'], gpu=False)
    results = reader.readtext(file_path, detail=0)
    text    = "\n".join(results)

    if not text.strip():
        text = "[No readable text found in this image.]"

    return [Document(
        page_content=text,
        metadata={"source": file_path, "type": "image_ocr"}
    )]


# ── Chunker ────────────────────────────────────────────────────────────────────

def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    return splitter.split_documents(documents)


# ── Public API ─────────────────────────────────────────────────────────────────

def ingest(file_path: str) -> list[Document]:
    """Load a single file and return text chunks."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        docs = load_pdf(file_path)
    elif ext == ".txt":
        docs = load_text(file_path)
    elif ext in IMAGE_EXTENSIONS:
        docs = load_image_ocr(file_path)
    else:
        raise ValueError(f"Unsupported file type: '{ext}'")

    return chunk_documents(docs)


def ingest_multiple(file_paths: list[str]) -> list[Document]:
    """Load multiple files and return combined chunks (multi-doc support)."""
    all_chunks = []
    for path in file_paths:
        try:
            chunks = ingest(path)
            fname = os.path.basename(path)
            for chunk in chunks:
                chunk.metadata.setdefault("source", fname)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[WARN] Could not ingest '{path}': {e}")
    return all_chunks