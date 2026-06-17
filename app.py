"""
app.py  —  RAG Document Q&A  (Professional Edition)
═════════════════════════════════════════════════════
Features:
  ✅ PDF / TXT support
  ✅ Image support (PNG, JPG, etc.) with OCR
  ✅ Multi-document upload & search
  ✅ Chat history memory (rolling window)
  ✅ Cloud vector DB (Pinecone) with local FAISS fallback
  ✅ Per-chunk source attribution
"""

import sys, os, tempfile
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📚",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Sidebar styling */
[data-testid="stSidebar"] { background: #0f1117; }
[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

/* Chat bubbles */
.user-bubble {
    background: #1e3a5f; color: #fff;
    border-radius: 12px 12px 2px 12px;
    padding: 10px 15px; margin: 6px 0;
    max-width: 80%; margin-left: auto;
}
.assistant-bubble {
    background: #1a1a2e; color: #e0e0e0;
    border-radius: 12px 12px 12px 2px;
    padding: 10px 15px; margin: 6px 0;
    max-width: 80%;
    border-left: 3px solid #4a90d9;
}

/* Source badge */
.src-badge {
    display: inline-block;
    background: #2c2c54; color: #a29bfe;
    border-radius: 20px; font-size: 11px;
    padding: 2px 8px; margin: 2px 3px;
    border: 1px solid #4a4a8a;
}

/* Status pill */
.status-pill {
    display: inline-block;
    padding: 3px 10px; border-radius: 20px;
    font-size: 12px; font-weight: 600;
}
.pill-green { background: #1a3a1a; color: #4caf50; border: 1px solid #4caf50; }
.pill-blue  { background: #1a2a3a; color: #4a90d9; border: 1px solid #4a90d9; }

/* Top header */
.app-header {
    background: linear-gradient(135deg, #0f1117 0%, #1a1a2e 100%);
    border-bottom: 2px solid #4a90d9;
    padding: 16px 24px; margin: -1rem -1rem 1.5rem -1rem;
}
</style>
""", unsafe_allow_html=True)


# ── API Key Check ──────────────────────────────────────────────────────────────
groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    st.error("❌ **GROQ_API_KEY not found!**  Add it in HuggingFace Space Settings → Secrets.")
    st.stop()

from src.ingestion    import ingest, IMAGE_EXTENSIONS
from src.vector_store import create_vector_store, index_exists, USE_PINECONE
from src.retriever    import retrieve_chunks
from src.generator    import generate_answer

ACCEPTED_TYPES = ["pdf", "txt", "png", "jpg", "jpeg", "bmp", "tiff", "webp"]


# ── Session State Init ─────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages"      : [],          # chat history  [{"role","content"}]
        "vectorstore"   : None,        # active vector store
        "uploaded_docs" : [],          # [{"name","chunks"}]
        "processing"    : False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <h2 style="margin:0;color:#ffffff;">📚 RAG Document Q&amp;A</h2>
  <p style="margin:4px 0 0;color:#9aa5b4;font-size:14px;">
      Upload any PDF, TXT, or Image — ask questions, get cited answers.
  </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # Vector DB badge
    db_type = "☁️ Pinecone (Cloud)" if USE_PINECONE else "💾 FAISS (Local)"
    db_color = "pill-blue" if USE_PINECONE else "pill-green"
    st.markdown(
        f'<span class="status-pill {db_color}">Vector DB: {db_type}</span>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("## 📂 Upload Documents")
    st.caption("Supported: PDF · TXT · PNG · JPG · BMP · TIFF · WEBP")

    uploaded_files = st.file_uploader(
        "Choose one or more files",
        type=ACCEPTED_TYPES,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("🚀 Process Documents", use_container_width=True, type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one file first.")
        else:
            already_loaded = {d["name"] for d in st.session_state.uploaded_docs}
            new_files = [f for f in uploaded_files if f.name not in already_loaded]

            if not new_files:
                st.info("All selected files are already processed.")
            else:
                progress = st.progress(0, text="Starting…")
                for idx, uf in enumerate(new_files):
                    progress.progress(
                        (idx) / len(new_files),
                        text=f"Processing {uf.name}…",
                    )
                    ext    = os.path.splitext(uf.name)[1].lower()
                    suffix = ext if ext else ".tmp"

                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(uf.read())
                        tmp_path = tmp.name

                    try:
                        chunks = ingest(tmp_path)
                        # tag every chunk with the original filename
                        for c in chunks:
                            c.metadata["source"] = uf.name

                        st.session_state.vectorstore = create_vector_store(
                            chunks,
                            existing_store=st.session_state.vectorstore,
                        )
                        st.session_state.uploaded_docs.append({
                            "name"  : uf.name,
                            "chunks": len(chunks),
                            "type"  : "🖼️ Image/OCR" if ext in IMAGE_EXTENSIONS else "📄 Text",
                        })
                        os.unlink(tmp_path)

                    except Exception as e:
                        st.error(f"❌ {uf.name}: {e}")
                        os.unlink(tmp_path)

                progress.progress(1.0, text="Done!")
                st.success(f"✅ {len(new_files)} file(s) processed!")
                st.rerun()

    # ── Loaded documents list ─────────────────────────────────────────────────
    if st.session_state.uploaded_docs:
        st.divider()
        st.markdown("### 📋 Loaded Documents")
        for doc in st.session_state.uploaded_docs:
            st.markdown(
                f"{doc['type']} **{doc['name']}**  "
                f"<span class='src-badge'>{doc['chunks']} chunks</span>",
                unsafe_allow_html=True,
            )

        # Source filter for targeted search
        st.divider()
        all_names = ["🔍 All Documents"] + [d["name"] for d in st.session_state.uploaded_docs]
        selected_source = st.selectbox("Search scope:", all_names)
        st.session_state["source_filter"] = (
            None if selected_source == "🔍 All Documents" else selected_source
        )

    st.divider()

    # Clear everything button
    if st.button("🗑️ Clear Everything", use_container_width=True):
        for k in ["messages", "vectorstore", "uploaded_docs", "source_filter"]:
            if k in st.session_state:
                del st.session_state[k]
        _init_state()
        st.rerun()

    # How it works
    with st.expander("ℹ️ How it works"):
        st.markdown("""
1. **Upload** PDF / TXT / Image files  
2. **OCR** extracts text from images automatically  
3. **Chunks** are stored in a vector database  
4. **Ask** a question in the chat  
5. **Retrieval** finds the most relevant chunks  
6. **Groq LLM** answers with source citations  
7. **Memory** keeps track of your conversation  
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════

col_chat, col_info = st.columns([3, 1])

with col_chat:
    if not st.session_state.uploaded_docs:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#666;">
          <div style="font-size:64px;margin-bottom:16px;">📂</div>
          <h3 style="color:#9aa5b4;">No documents loaded yet</h3>
          <p style="color:#666;">Upload one or more files using the sidebar, then start asking questions!</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Render chat history ───────────────────────────────────────────────
        chat_container = st.container()
        with chat_container:
            if not st.session_state.messages:
                st.markdown(
                    "<p style='color:#666;text-align:center;margin-top:40px;'>"
                    "💬 Ask your first question below…</p>",
                    unsafe_allow_html=True,
                )

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    # show source badges for assistant messages
                    if msg["role"] == "assistant" and msg.get("sources"):
                        badges = " ".join(
                            f'<span class="src-badge">📄 {s}</span>'
                            for s in msg["sources"]
                        )
                        st.markdown(badges, unsafe_allow_html=True)

        # ── Chat input ────────────────────────────────────────────────────────
        query = st.chat_input("Ask a question about your document(s)…")

        if query:
            # Display user message
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})

            # Generate answer
            with st.chat_message("assistant"):
                with st.spinner("Searching & generating answer… 🔍"):
                    try:
                        source_filter = st.session_state.get("source_filter")
                        chunks = retrieve_chunks(
                            query,
                            st.session_state.vectorstore,
                            source_filter=source_filter,
                        )
                        # pass full chat history for memory
                        answer = generate_answer(
                            query,
                            chunks,
                            chat_history=st.session_state.messages[:-1],  # exclude current
                        )
                        sources = list({c["source"] for c in chunks})
                    except Exception as e:
                        answer  = f"❌ Error: {e}"
                        chunks  = []
                        sources = []

                st.markdown(answer)

                # source badges
                if sources:
                    badges = " ".join(
                        f'<span class="src-badge">📄 {s}</span>'
                        for s in sources
                    )
                    st.markdown(badges, unsafe_allow_html=True)

                # expandable source chunks
                if chunks:
                    with st.expander(f"📎 View {len(chunks)} source chunk(s)"):
                        for i, chunk in enumerate(chunks, 1):
                            score_pct = max(0, 100 - chunk["score"] * 100)
                            st.markdown(
                                f"**Chunk {i}** — "
                                f'<span class="src-badge">📄 {chunk["source"]}</span> '
                                f'<span class="src-badge">relevance: {score_pct:.0f}%</span>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f'<div style="background:#111;padding:10px;border-radius:6px;'
                                f'font-size:13px;color:#ccc;white-space:pre-wrap;">'
                                f'{chunk["text"]}</div>',
                                unsafe_allow_html=True,
                            )
                            st.divider()

            # Save to history
            st.session_state.messages.append({
                "role"   : "assistant",
                "content": answer,
                "sources": sources,
            })


# ── Right info panel ──────────────────────────────────────────────────────────
with col_info:
    if st.session_state.uploaded_docs:
        st.markdown("### 📊 Stats")
        total_chunks = sum(d["chunks"] for d in st.session_state.uploaded_docs)
        st.metric("Documents", len(st.session_state.uploaded_docs))
        st.metric("Total Chunks", total_chunks)
        st.metric("Messages", len(st.session_state.messages))

        st.divider()
        st.markdown("### 🔧 Tools Active")
        st.markdown("✅ PDF Parser")
        st.markdown("✅ TXT Reader")
        st.markdown("✅ OCR Engine")
        st.markdown(f"✅ {'Pinecone' if USE_PINECONE else 'FAISS'} VectorDB")
        st.markdown("✅ Groq LLM")
        st.markdown("✅ Chat Memory")

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()