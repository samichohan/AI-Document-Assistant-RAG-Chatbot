import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import streamlit as st
import tempfile

from src.ingestion import ingest
from src.vector_store import create_vector_store, load_vector_store, index_exists
from src.retriever import retrieve_chunks
from src.generator import generate_answer

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Application",
    page_icon="📚",
    layout="centered"
)

st.title("📚 RAG Application — Document Q&A")
st.caption("Upload a PDF or text file and ask questions about it!")

# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "file_processed" not in st.session_state:
    st.session_state.file_processed = False

# ── Sidebar — File Upload ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=["pdf", "txt"]
    )

    if uploaded_file and not st.session_state.file_processed:
        with st.spinner("Processing document... please wait ⏳"):
            # Temp file mein save karo
            suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Ingest + chunk + embed + store in FAISS
            chunks = ingest(tmp_path)
            vectorstore = create_vector_store(chunks)
            st.session_state.vectorstore = vectorstore
            st.session_state.file_processed = True
            os.unlink(tmp_path)  # temp file delete karo

        st.success(f"✅ Document processed! {len(chunks)} chunks created.")

    if uploaded_file and st.session_state.file_processed:
        st.info("📄 Document is ready. Ask your questions!")

    # Reset button
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
    # Show old messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User input
    query = st.chat_input("Ask a question about your document...")

    if query:
        # User message show + save
        with st.chat_message("user"):
            st.write(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Searching document... 🔍"):
                chunks = retrieve_chunks(query, st.session_state.vectorstore)
                answer = generate_answer(query, chunks)

            st.write(answer)

            # Source chunks dikhao (same as sir's app)
            with st.expander("📎 Source Chunks"):
                for i, chunk in enumerate(chunks):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(chunk)
                    st.divider()

        # Save assistant message
        st.session_state.messages.append({"role": "assistant", "content": answer})