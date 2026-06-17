import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP
 
 
def load_pdf(file_path: str) -> list:
    """PDF file read karo aur documents list return karo"""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents
 
 
def load_text(text: str) -> list:   
    """Plain text string ko document format mein convert karo"""
    from langchain.schema import Document
    return [Document(page_content=text, metadata={"source": "user_input"})]
 
 
def chunk_documents(documents: list) -> list:
    """
    Documents ko chote chunks mein todo
    chunk_size  = har chunk mein max characters
    chunk_overlap = chunks ke beech shared characters (context preserve karne ke liye)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(documents)
    return chunks
 
 
def ingest(file_path: str) -> list:
    """
    Full ingestion pipeline:
    file_path → load → chunk → return chunks
    """
    if file_path.endswith(".pdf"):
        docs = load_pdf(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        docs = load_text(text)
 
    chunks = chunk_documents(docs)
    return chunks