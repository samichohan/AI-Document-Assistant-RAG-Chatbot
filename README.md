# 📚 RAG Application — Document Q&A

A Retrieval-Augmented Generation (RAG) app that lets you upload any PDF or text file and ask questions about it using AI.

## 🗂️ Project Structure

```
RAG-APP/
├── src/
│   ├── ingestion.py      # PDF/text load + chunking
│   ├── vector_store.py   # Embeddings + FAISS storage
│   ├── retriever.py      # Relevant chunk retrieval
│   └── generator.py      # Groq LLM answer generation
├── data/                 # FAISS index saved here
├── app.py                # Streamlit UI
├── config.py             # All settings
├── requirements.txt
└── .env
```

## ⚙️ How It Works

```
PDF Upload → Chunking → Embeddings → FAISS
                                        ↓
User Query → Embedding → FAISS Search → Top Chunks → Groq LLM → Answer
```

## 🚀 Setup & Run

**1. Install dependencies:**
```bash
pip install -r requirements.txt
```

**2. Create `.env` file:**
```
GROQ_API_KEY=your_key_here
```

**3. Run the app:**
```bash
streamlit run app.py
```

## 🛠️ Tech Stack

- **LLM:** Groq (llama-3.3-70b-versatile) — Free
- **Embeddings:** HuggingFace all-MiniLM-L6-v2 — Free, runs locally
- **Vector DB:** FAISS — Free, runs locally
- **UI:** Streamlit
- **Framework:** LangChain