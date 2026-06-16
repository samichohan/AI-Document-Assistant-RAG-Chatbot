import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "/tmp/faiss_index"


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_vector_store(chunks: list):
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    return vectorstore


def load_vector_store():
    embeddings = get_embedding_model()
    vectorstore = FAISS.load_local(
        FAISS_INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore


def index_exists() -> bool:
    return os.path.exists(FAISS_INDEX_PATH)