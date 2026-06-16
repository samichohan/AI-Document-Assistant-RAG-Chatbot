import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
import tempfile

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Application",
    page_icon="📚",
    layout="centered"
)

# ── API KEY CHECK — sabse pehle ───────────────────────────────────────────────
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    st.error("❌ GROQ_API_KEY not found! Please add it in HuggingFace Space Settings → Secrets.")
    st.stop()

from src.ingestion import ingest
from src.vector_store import create_vector_store
from src.retriever import retrieve_chunks
from src.generator import generate_answer

st.title("📚 RAG Application — Document Q&A")
st.caption("Upload a PDF or text file and ask questions about it!")

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])

    if uploaded_file and not st.session_state.file_processed:
        with st.spinner("Processing document... please wait ⏳"):
            suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            try:
                chunks = ingest(tmp_path)
                vectorstore = create_vector_store(chunks)
                st.session_state.vectorstore = vectorstore
                st.session_state.file_processed = True
                os.unlink(tmp_path)
                st.success(f"✅ Document processed! {len(chunks)} chunks created.")
            except Exception as e:
                st.error(f"❌ Error processing document: {str(e)}")
                os.unlink(tmp_path)

    if uploaded_file and st.session_state.file_processed:
        st.info("📄 Document is ready. Ask your questions!")

    if st.button("🔄 Upload New Document"):
        st.session_state.messages = []
        st.session_state.vectorstore = None
        st.session_state.file_processed = False
        st.rerun()

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("1. Upload PDF/TXT\n2. Document gets chunked\n3. Embeddings stored in FAISS\n4. Ask questions!\n5. Relevant chunks retrieved\n6. Groq LLM answers")

# ── Main Chat Area ────────────────────────────────────────────────────────────
if not st.session_state.file_processed:
    st.info("👈 Please upload a document from the sidebar to get started.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    query = st.chat_input("Ask a question about your document...")

    if query:
        with st.chat_message("user"):
            st.write(query)
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("Searching document... 🔍"):
                try:
                    chunks = retrieve_chunks(query, st.session_state.vectorstore)
                    answer = generate_answer(query, chunks)
                except Exception as e:
                    answer = f"❌ Error: {str(e)}"

            st.write(answer)

            with st.expander("📎 Source Chunks"):
                for i, chunk in enumerate(chunks):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(chunk)
                    st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})