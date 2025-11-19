import os
import io
from typing import List
import streamlit as st
import pandas as pd
import numpy as np
import faiss

from dotenv import load_dotenv
from pypdf import PdfReader
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
)

load_dotenv()

st.set_page_config(page_title="CV Screening RAG Chatbot", page_icon="🧠", layout="wide")
st.title("🧠 CV Screening RAG Chatbot")

# Session state - INISIALISASI DI AWAL UNTUK MENCEGAH ERROR
if "index" not in st.session_state:
    st.session_state.index = None
if "meta" not in st.session_state:
    st.session_state.meta = None

embed_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # default

# Sidebar config
with st.sidebar:
    uploaded_pdfs = st.file_uploader(
        "Upload PDF resumes", type=["pdf"], accept_multiple_files=True
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
    "llama3-70b-8192",
    "llama3-8b-8192"
]
    chat_model = st.selectbox(
        "GROQ Chat Model",
        options=chat_model_options,
        index=chat_model_options.index("openai/gpt-oss-20b"),
    )

    top_k = st.slider("Top-K", 1, 10, 5)


# Session state sudah diinisialisasi di atas

# Tampilkan info model yang sedang digunakan untuk index (SETELAH INISIALISASI)
with st.sidebar:
    if st.session_state.meta:
        current_model = st.session_state.meta.get("embed_model", "Unknown")
        st.info(f"📊 Index menggunakan: {current_model}")
        if embed_model != current_model:
            st.warning("⚠️ Model tidak konsisten! Rebuild index diperlukan.")

# Handle reset button
if reset_button:
    st.session_state.index = None
    st.session_state.meta = None
    st.success("✅ Index berhasil direset!")

# Auto-reset jika model berubah untuk mencegah error dimensi
if st.session_state.meta is not None:
    current_model = st.session_state.meta.get("embed_model", "text-embedding-3-small")
    if embed_model != current_model:
        # Auto reset untuk mencegah error dimensi
        if st.session_state.index is not None:
            st.session_state.index = None
            st.session_state.meta = None
            st.warning(
                f"🔄 Auto-reset: Model berubah dari {current_model} ke {embed_model}. Silakan build index ulang."
            )


def build_index_from_pdfs(files: List[io.BytesIO], embed_model: str):
    rows = []
    for f in files:
        try:
            reader = PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            doc_id = getattr(f, "name", "uploaded.pdf")
            for i, ch in enumerate(chunk_text(text)):
                rows.append(
                    {
                        "doc_id": doc_id,
                        "chunk_id": i,
                        "text": ch,
                        "category": "Uploaded",
                    }
                )
        except Exception as e:
            st.warning(f"Failed to parse {getattr(f, 'name', 'PDF')}: {e}")
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
        "embed_model": embed_model,  # Simpan model yang digunakan
    }
    return index, meta


if build_button:
    with st.spinner("Building index from uploaded PDFs..."):
        try:
            if uploaded_pdfs:
                index, meta = build_index_from_pdfs(uploaded_pdfs, embed_model)
                st.session_state.index = index
                st.session_state.meta = meta
                st.success(f"Index ready! Using {embed_model}")
            else:
                st.warning("Please upload at least one PDF to build the index.")
        except Exception as e:
            st.error(f"Error building index: {e}")

# Chat input
query = st.text_input(
    "Ask a question (e.g., *Who has strong Python + SQL for data engineering?*)"
)
ask = st.button("Ask")

# Display
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Answer")
    if ask:
        if st.session_state.index is None or st.session_state.meta is None:
            st.warning("Build the index first from the sidebar.")
        else:
            # Validasi model consistency
            index_model = st.session_state.meta.get(
                "embed_model", "text-embedding-3-small"
            )
            if embed_model != index_model:
                st.error(
                    f"⚠️ Model mismatch! Index dibuat dengan {index_model}, tapi Anda memilih {embed_model}. Silakan rebuild index atau ganti model embedding."
                )
                st.info(
                    "💡 Tip: Klik tombol 🗑️ Reset untuk clear index lama, lalu build ulang."
                )
            else:
                os.environ["OPENAI_EMBEDDING_MODEL"] = embed_model
                os.environ["OPENAI_CHAT_MODEL"] = chat_model

                client = get_groq_client()
                embedding_response = client.embeddings.create(
                    model=embed_model,
                    input=query,
                )
                qvec = np.array([embedding_response.data[0].embedding], dtype="float32")

                D, I = st.session_state.index.search(qvec, int(top_k))

                retrieved = []
                meta = st.session_state.meta
                for idx in I[0]:
                    if idx < 0:
                        continue
                    retrieved.append(
                        DocChunk(
                            doc_id=meta["doc_ids"][idx],
                            chunk_id=meta["chunk_ids"][idx],
                            text=meta["texts"][idx],
                            meta={"category": meta["categories"][idx]},
                        )
                    )
                response = client.chat.completions.create(
                    model=chat_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful AI assistant for CV screening."},
                        {"role": "user", "content": f"Context:\n{''.join([c.text for c in retrieved])}\n\nQuestion: {query}"}
                    ]
                )

                answer = response.choices[0].message["content"]
                st.write(answer)

with col2:
    st.subheader("Retrieved snippets")
    if ask and st.session_state.meta:
        # Cek konsistensi model
        index_model = st.session_state.meta.get("embed_model", "text-embedding-3-small")
        if embed_model == index_model and st.session_state.index is not None:
            meta = st.session_state.meta
            rows = []
            for rank, idx in enumerate(I[0]):
                if idx < 0:
                    continue
                rows.append(
                    {
                        "rank": rank + 1,
                        "doc_id": meta["doc_ids"][idx],
                        "category": meta["categories"][idx],
                        "snippet": meta["texts"][idx][:400]
                        + ("..." if len(meta["texts"][idx]) > 400 else ""),
                    }
                )
            st.dataframe(pd.DataFrame(rows))
        else:
            st.caption("Model tidak konsisten. Silakan rebuild index atau ganti model.")
    else:
        st.caption(
            "Upload PDFs, build the index, and ask a question to see retrieved context here."
        )
