# 🤖 AI Document Assistant — RAG Chatbot

An AI-powered Retrieval-Augmented Generation (RAG) application that allows users to upload documents and images, perform semantic search, and get accurate answers grounded in their own data.

## 🌐 Live Demo

**Try the application here:**

https://ai-document-assistant-rag-chatbot.streamlit.app/

---

## 🚀 Features

### 📄 Multi-Document Support

* Upload multiple PDF files
* Upload TXT documents
* Search across all uploaded documents

### 🖼️ Image Support with OCR

* Upload JPG, PNG, BMP, TIFF, and WEBP images
* Extract text using OCR
* Ask questions about image content

### 🧠 Retrieval-Augmented Generation (RAG)

* Intelligent document chunking
* Semantic search using vector embeddings
* Context-aware answers powered by LLMs

### 🔍 Advanced Search

* Multi-document retrieval
* Relevant chunk ranking
* Source citation support

### 💬 Conversational Interface

* Chat-style user experience
* Chat history memory
* Easy document exploration

### ⚡ High Performance

* FAISS vector database
* Hugging Face embeddings
* Groq LLM integration
* Fast retrieval and response generation

---

## 🏗️ Architecture

```text
Documents / Images
        │
        ▼
 Data Ingestion
        │
        ▼
 OCR (Images)
        │
        ▼
 Text Chunking
        │
        ▼
 Embedding Generation
        │
        ▼
 FAISS Vector Database
        │
        ▼
 Similarity Search
        │
        ▼
 Relevant Context Retrieval
        │
        ▼
 Groq LLM
        │
        ▼
 Final Answer
```

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### LLM

* Groq
* Llama 3.3 70B Versatile

### Embeddings

* Hugging Face
* sentence-transformers/all-MiniLM-L6-v2

### Vector Database

* FAISS

### OCR

* Tesseract OCR
* Pillow

### Frameworks

* LangChain
* Python

---

## 📂 Supported File Types

| Type      | Formats                   |
| --------- | ------------------------- |
| Documents | PDF, TXT                  |
| Images    | JPG, PNG, BMP, TIFF, WEBP |

---

## 📸 Example Use Cases

### Education

* Study notes assistant
* Research paper Q&A
* Course material search

### Business

* Internal knowledge base
* Company documentation search
* Report analysis

### Legal & Compliance

* Contract analysis
* Policy document retrieval

### Personal Productivity

* Chat with your PDFs
* Organize personal documents
* Image text extraction

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/samichohan/AI-Document-Assistant-RAG-Chatbot.git
cd AI-Document-Assistant-RAG-Chatbot
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
```

### Run Application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```text
AI-Document-Assistant-RAG-Chatbot/
│
├── src/
│   ├── ingestion.py
│   ├── vector_store.py
│   ├── retriever.py
│   └── generator.py
│
├── app.py
├── config.py
├── requirements.txt
├── README.md
└── .env
```

---

## 🎯 Key Capabilities

✅ PDF Question Answering

✅ Image OCR Processing

✅ Multi-Document Search

✅ Semantic Retrieval

✅ Context-Aware Responses

✅ Chat History Memory

✅ Streamlit Web Interface

✅ FAISS Vector Search

---

## 👨‍💻 Developer

**Sami Chohan**

AI & Machine Learning Enthusiast

GitHub: https://github.com/samichohan

---

## ⭐ Future Improvements

* Cloud Vector Database Support (Pinecone, Weaviate)
* User Authentication
* Document Summarization
* API Endpoints
* Multi-Language Support
* Advanced Analytics Dashboard

---

If you found this project useful, consider giving it a ⭐ on GitHub.
