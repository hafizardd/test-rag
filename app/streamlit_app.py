import os
import io
from typing import List
import streamlit as st
import pandas as pd
import numpy as np
import faiss

from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag.rag_core import (
    embed_texts,
    chunk_text,
    build_faiss_index,
    answer_with_rag,
    DocChunk,
    get_groq_client,
    enhanced_retrieval
)

load_dotenv()

# Custom CSS styling
def add_custom_css():
    st.markdown("""
        <style>
        /* Background and text styling */
        .main {
            background-color: #0d1117;
            color: #e6edf3;
        }
        body {
            background-color: #0d1117;
        }

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #161b22;
        }

        /* Buttons */
        .stButton>button {
            background-color: #238636;
            color: #ffffff;
            border-radius: 8px;
            border: none;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #2ea043;
            color: white;
        }

        /* Reset button - red */
        .stButton > button[kind="secondary"] {
            background-color: #da3633 !important;
            color: white !important;
        }

        /* Dataframe styling */
        .stDataFrame {
            border-radius: 10px !important;
            overflow: hidden !important;
        }

        /* Answer card */
        .answer-box {
            background-color: #161b22;
            padding: 15px;
            border-radius: 12px;
            border: 1px solid #30363d;
            margin-top: 10px;
        }

        /* Header accent */
        h1, h2, h3 {
            color: #58a6ff;
            font-weight: 700;
        }

        </style>
    """, unsafe_allow_html=True)


st.set_page_config(page_title="RAG Chatbot", layout="wide")
add_custom_css()
st.title("RAG Chatbot")

# Session state - INISIALISASI DI AWAL UNTUK MENCEGAH ERROR
if "index" not in st.session_state:
    st.session_state.index = None
if "meta" not in st.session_state:
    st.session_state.meta = None

# Load embedding model (cached)
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')

embed_model = load_embedding_model()
embed_model_name = 'sentence-transformers/all-mpnet-base-v2' 

# Sidebar config
with st.sidebar:
    uploaded_files = st.file_uploader(
        "Upload PDF resumes", 
        type=["pdf", "docx", "csv", "xlsx"], 
        accept_multiple_files=True
    )

    col_build, col_reset = st.columns([2, 1])
    with col_build:
        build_button = st.button("(Re)build Index")
    with col_reset:
        reset_button = st.button("🗑️ Reset", help="Clear index and start fresh")

    st.markdown("---")
    st.header("Settings")

    # Dropdown for Chat Models
    chat_model_options = [
    "openai/gpt-oss-20b",
    "llama3-70b-8192"
]
    chat_model = st.selectbox(
        "GROQ Chat Model",
        options=chat_model_options,
        index=chat_model_options.index("openai/gpt-oss-20b"),
    )

    top_k = st.slider("Top-K", 1, 10, 5)


# Session state sudah diinisialisasi di atas

# info model yang sedang digunakan untuk index
with st.sidebar:
    st.info(f"Using: {embed_model_name}")

# Handle reset button
if reset_button:
    st.session_state.index = None
    st.session_state.meta = None
    st.success("Index berhasil direset!")


def build_index_from_files(files: List[io.BytesIO], embed_model: SentenceTransformer):
    rows = []

    for f in files:
        file_name = getattr(f, "name", "uploaded")

        # PDF
        if file_name.endswith(".pdf"):
            try:
                reader = PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                for i, ch in enumerate(chunk_text(text)):
                    rows.append({"doc_id": file_name, "chunk_id": i, "text": ch, "category": "PDF"})
            except Exception as e:
                st.warning(f"Failed to read PDF {file_name}: {e}")

        # DOCX
        elif file_name.endswith(".docx"):
            try:
                doc = Document(f)
                text = "\n".join([p.text for p in doc.paragraphs])
                for i, ch in enumerate(chunk_text(text)):
                    rows.append({"doc_id": file_name, "chunk_id": i, "text": ch, "category": "DOCX"})
            except Exception as e:
                st.warning(f"Failed to read DOCX {file_name}: {e}")

        # CSV
        elif file_name.endswith(".csv"):
            try:
                df = pd.read_csv(f)
                text = df.to_string()
                for i, ch in enumerate(chunk_text(text)):
                    rows.append({"doc_id": file_name, "chunk_id": i, "text": ch, "category": "CSV"})
            except Exception as e:
                st.warning(f"Failed to read CSV {file_name}: {e}")

        # XLSX / Excel
        elif file_name.endswith(".xlsx"):
            try:
                df = pd.read_excel(f)
                text = df.to_string()
                for i, ch in enumerate(chunk_text(text)):
                    rows.append({"doc_id": file_name, "chunk_id": i, "text": ch, "category": "XLSX"})
            except Exception as e:
                st.warning(f"Failed to read XLSX {file_name}: {e}")
    
    if not rows:
        return None, None
    
    chunk_df = pd.DataFrame(rows)
    
    vecs = embed_texts(
        chunk_df["text"].tolist(),
        embed_model=embed_model,
        show_progress=True,
    )
    index = build_faiss_index(vecs)
    
    meta = {
        "doc_ids": chunk_df["doc_id"].tolist(),
        "chunk_ids": chunk_df["chunk_id"].tolist(),
        "texts": chunk_df["text"].tolist(),
        "categories": chunk_df["category"].tolist(),
    }
    return index, meta


if build_button:
    with st.spinner("Building index from uploaded PDFs..."):
        try:
            if uploaded_files:
                index, meta = build_index_from_files(uploaded_files, embed_model)
                st.session_state.index = index
                st.session_state.meta = meta
                st.success(f"Index ready! Using {embed_model_name}")
            else:
                st.warning("Please upload at least one PDF to build the index.")
        except Exception as e:
            st.error(f"Error building index: {e}")

# Chat input
query = st.text_input(
    "Ask a question (contoh, *Dimana Amel kuliah?*)"
)
ask = st.button("Ask")

# Display
answer_text=None
retrieved_docs  = []
search_indices = None

if ask:
    if st.session_state.index is None or st.session_state.meta is None:
        answer_text = "⚠️ Build the index first from the sidebar."
    else:
        # Embed query
        results = enhanced_retrieval(
            index=st.session_state.index,
            docs=[
                DocChunk(
                    doc_id=st.session_state.meta["doc_ids"][i],
                    chunk_id=st.session_state.meta["chunk_ids"][i],
                    text=st.session_state.meta["texts"][i],
                    meta={"category": st.session_state.meta["categories"][i]},
                )
                for i in range(len(st.session_state.meta["texts"]))
            ],
            query=query,
            embed_model=embed_model,
            k=top_k,
        )
        retrieved_docs = [r["document"] for r in results]

        # SAVE SEARCH INDICES FOR DISPLAY
        search_indices = np.array([[doc.chunk_id for doc in retrieved_docs]])

        retrieved_docs = [r["document"] for r in results]
        
        # Generate answer
        if retrieved_docs:
            answer_text = answer_with_rag(query, retrieved_docs, chat_model)
        else:
            answer_text = "No relevant documents found."

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("💡 Answer")
    if answer_text:
        st.markdown(f"<div class='answer-box'>{answer_text}</div>", unsafe_allow_html=True)
    else:
        st.caption("Ask a question to see the answer here.")

with col2:
    st.subheader("📄 Retrieved Snippets")
    if search_indices is not None and st.session_state.meta:
        meta = st.session_state.meta
        rows = []
        for rank, idx in enumerate(search_indices[0]):
            if idx < 0: continue
            rows.append(
                {
                    "Rank": rank + 1,
                    "Document": meta["doc_ids"][idx],
                    "Category": meta["categories"][idx],
                    "Snippet": meta["texts"][idx][:400] + ("..." if len(meta["texts"][idx]) > 400 else "")
                }
            )
        df_snippets = pd.DataFrame(rows)
        st.dataframe(df_snippets)
        # --- DOWNLOAD AS CSV ---
        csv_data = df_snippets.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Retrieved Snippets (CSV)",
            data=csv_data,
            file_name="retrieved_snippets.csv",
            mime="text/csv"
        )
    else:
        st.caption("Upload PDF and ask a question to see retrieved context.")
