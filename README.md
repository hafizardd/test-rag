# RAG-Portfolio

A Retrieval-Augmented Generation (RAG) chatbot built with Streamlit for intelligent pdf screening. This application uses Sentence Transformers for embeddings, FAISS for vector search, and Groq API for natural language responses (model).

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)

---

## 🌟 About

This RAG (Retrieval-Augmented Generation) chatbot helps us efficiently know the context by:

- **Uploading PDF resumes** and automatically indexing them
- **Asking questions** like "What major Amelia took this semester?"
- **Getting AI-generated answers** based on the actual content of uploaded pdf
- **Viewing retrieved snippets** to verify the source information

The system combines:
- 🔍 **Semantic Search** using Sentence Transformers embeddings
- ⚡ **Fast Retrieval** with FAISS vector indexing
- 🧠 **Smart Answers** powered by Groq's LLM API

You can access it on chrome/browser (will be work if you've done clone this repo) :
- http://192.168.100.6:8501

---

## ✨ Key Features

### ✅ Currently Implemented

- [x] **PDF Resume Upload** - Upload PDF
- [x] **Automatic Text Extraction** - Extracts text from PDF documents
- [x] **Intelligent Chunking** - Splits documents into manageable chunks with overlap
- [x] **Semantic Embeddings** - Uses `sentence-transformers/all-mpnet-base-v2` for high-quality embeddings
- [x] **FAISS Vector Search** - Fast similarity search with FAISS IndexFlatIP
- [x] **RAG Pipeline** - Retrieves relevant context and generates answers
- [x] **Multiple LLM Support** - Choose between Llama-3-70B, Llama-3-8B, or Mixtral-8x7B
- [x] **Top-K Retrieval** - Adjustable number of retrieved snippets (1-10)
- [x] **Retrieved Snippets Display** - View the actual resume sections used for answers
- [x] **Index Management** - Rebuild or reset index 

---

## 🚀 Workflow

### High-Level Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   PDF       │────>│   Text       │────>│  Chunking   │
│  Resumes    │     │ Extraction   │     │    Strategy │
└─────────────┘     └──────────────┘     └─────────────┘
                                                 │
                                                 ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   FAISS     │<────│  Sentence    │<────│   Text      │
│   Index     │     │ Transformers │     │  Chunks     │
└─────────────┘     └──────────────┘     └─────────────┘
      │
      ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Query     │────>│  Vector      │────>│  Retrieved  │
│ Embedding   │     │  Search      │     │  Chunks     │
└─────────────┘     └──────────────┘     └─────────────┘
                                                 │
                                                 ▼
                                          ┌─────────────┐
                                          │   Model     │
                                          │   LLM       │
                                          │  Response   │
                                          └─────────────┘
```

## 📋 Implementation Checklist (ONGOING)

### Phase 1: Core Functionality (Completed)

- [x] Set up project structure
- [x] Implement PDF text extraction
- [x] Create text chunking with overlap
- [x] Integrate Sentence Transformers embeddings
- [x] Build FAISS vector index
- [x] Implement semantic search
- [x] Connect Groq API for LLM responses
- [x] Create basic Streamlit UI
- [x] Add model caching for performance
- [x] Fix embedding serialization issues
- [x] Display retrieved snippets

### Phase 2: Enhanced Features (In Progress)

- [x] Multiple chat model selection
- [x] Adjustable Top-K parameter
- [ ] Add multiple upload pdf
- [ ] Implement conversation history
- [ ] Add Reranking, Quality Metrics, Metadata Extraction	
- [ ] Work for Indonesian Languange
dst

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Groq API key ([Get one here](https://console.groq.com))

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/AmeliaSyahla/rag_portfolio.git
   cd rag_portfolio
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   
   # Windows
   .venv\Scripts\activate
   
   # Linux/Mac
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   HF_API_KEU=your_hugging_face_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run main/app/streamlit_app.py
   ```

6. **Open in browser**
   Navigate to ` http://192.168.100.6:8501`

---

## Project Structure

```
rag_portfolio/
├── main/
│   └── app/
│       └── streamlit_app.py      # Streamlit UI application
├── rag/
│   └── rag_core.py                # Core RAG functions (embedding, search, answer)
├── .env                           # Environment variables (API keys)
├── .gitignore                     # Git ignore file
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🔧 Configuration

### Embedding Model
Default: `sentence-transformers/all-mpnet-base-v2`

To change the model, edit `streamlit_app.py`:
```python
embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
```

### Chunking Parameters
Edit in `rag_core.py`:
```python
def chunk_text(text: str, max_tokens: int = 400, overlap_tokens: int = 60):
```

### LLM Temperature
Edit in `rag_core.py`:
```python
resp = client.chat.completions.create(
    model=chat_model,
    temperature=0.2,  # Lower = more focused, Higher = more creative
)
```

---

## 📊 Project Status

**Status**: In Active Development

**⭐ If you find this project useful, please consider giving it a star!**
