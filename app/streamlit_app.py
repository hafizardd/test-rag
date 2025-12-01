import os
import io
import sys
from typing import List
from datetime import datetime
import streamlit as st
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import RAG functions
from rag.rag_core import (
    chunk_text_advanced,
    extract_keywords_simple,
    generate_document_summary,
    build_vector_store_with_metadata,
    save_vector_store_with_metadata,
    load_vector_store_with_metadata,
    search_vector_store_with_reranking,
    answer_with_rag,
    DocChunk,
    DocumentMetadata,
)

load_dotenv()

# ========== Page Configuration ==========
st.set_page_config(
    page_title="RAG Chatbot - Advanced",
    page_icon="📚",
    layout="wide"
)

# ========== Custom CSS Styling ==========
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

        /* Dataframe styling */
        .stDataFrame {
            border-radius: 10px ! important;
            overflow: hidden !important;
        }

        /* Answer card */
        .answer-box {
            background-color: #161b22;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #30363d;
            margin-top: 10px;
            line-height: 1.6;
        }

        /* Header accent */
        h1, h2, h3 {
            color: #58a6ff;
            font-weight: 700;
        }

        /* Score badge */
        .score-badge {
            background-color: #238636;
            color: white;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.85em;
            font-weight: 600;
        }
        </style>
    """, unsafe_allow_html=True)


add_custom_css()

# ========== Title ==========
st.title("RAG Chatbot ")
# ========== Session State Initialization ==========
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "doc_metadata" not in st.session_state:
    st.session_state.doc_metadata = None
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

# ========== Load Embedding Model (Cached) ==========
@st.cache_resource
def load_embedding_model():
    """Load SentenceTransformer model (cached)"""
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')

embed_model = load_embedding_model()
embed_model_name = 'sentence-transformers/all-mpnet-base-v2'

# ========== Sidebar Configuration ==========
with st.sidebar:
    st.header("📤 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload files (PDF, DOCX, CSV, XLSX)", 
        type=["pdf", "docx", "csv", "xlsx"], 
        accept_multiple_files=True
    )
    
    col_build, col_reset = st.columns([2, 1])
    with col_build:
        build_button = st.button("🔨 Build Index", use_container_width=True)
    with col_reset:
        reset_button = st.button("🗑️", help="Clear index")
    
    st.markdown("---")
    st.header("⚙️ Chunking Settings")
    
    chunking_strategy = st.selectbox(
        "Strategy",
        options=["recursive", "semantic", "paragraph", "simple"],
        index=0,
        help="Choose chunking strategy"
    )
    
    chunk_size = st.slider("Chunk Size (chars)", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk Overlap (chars)", 50, 500, 100, 50)
    
    st.markdown("---")
    st.header("🎯 Retrieval Settings")
    
    # Reranking options
    enable_reranking = st.checkbox("Enable Reranking", value=True)
    
    if enable_reranking:
        rerank_method = st.selectbox(
            "Reranking Method",
            options=["hybrid", "keyword", "semantic"],
            index=0,
            help="Hybrid: 70% semantic + 30% keyword"
        )
        rerank_top_k = st. slider("Initial Retrieval (before rerank)", 10, 50, 20, 5)
    else:
        rerank_method = "semantic"
        rerank_top_k = 5
    
    top_k = st.slider("Final Results", 1, 10, 5)
    
    st.markdown("---")
    st.header("🤖 Model Settings")
    
    chat_model_options = [
        "openai/gpt-oss-20b"
    ]
    
    chat_model = st.selectbox(
        "GROQ Chat Model",
        options=chat_model_options,
        index=0,
    )
    
    st.markdown("---")
    st.info(f"🔧 **Embedding Model**\n\n{embed_model_name}")
    
    # Show index status
    if st.session_state.vector_store is not None:
        st.success(f"✅ Index Loaded\n\n{st.session_state.total_chunks} chunks indexed")
    else:
        st.warning("⚠️ No index loaded")

# ========== Handle Reset Button ==========
if reset_button:
    st.session_state.vector_store = None
    st.session_state. doc_chunks = []
    st.session_state.doc_metadata = None
    st.session_state.total_chunks = 0
    st.rerun()

# ========== Build Index from Uploaded Files ==========
def build_index_from_files(
    files: List[io.BytesIO],
    embed_model: SentenceTransformer,
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int
):
    """
    Build vector store from uploaded files
    
    Returns:
        (vector_store, doc_chunks, doc_metadata_list)
    """
    documents = []
    doc_metadata_list = []
    
    for f in files:
        file_name = getattr(f, "name", "uploaded")
        file_text = ""
        
        try:
            # PDF
            if file_name.endswith(".pdf"):
                reader = PdfReader(f)
                file_text = ""
                page_count = len(reader.pages)
                for page in reader.pages:
                    file_text += page.extract_text() or ""
            
            # DOCX
            elif file_name.endswith(".docx"):
                doc = Document(f)
                file_text = "\n".join([p.text for p in doc.paragraphs])
                page_count = 1
            
            # CSV
            elif file_name.endswith(".csv"):
                df = pd. read_csv(f)
                file_text = df.to_string()
                page_count = 1
            
            # XLSX
            elif file_name.endswith(".xlsx"):
                df = pd.read_excel(f)
                file_text = df. to_string()
                page_count = 1
            
            # Process file text
            if file_text. strip():
                # Extract metadata
                keywords = extract_keywords_simple(file_text, top_n=5)
                summary = generate_document_summary(file_text, max_length=150)
                
                # Create DocumentMetadata
                doc_metadata = DocumentMetadata(
                    filename=file_name,
                    file_size=len(file_text),
                    creation_date=datetime.now(),
                    page_count=page_count,
                    keywords=keywords,
                    summary=summary,
                    document_type=file_name.split('.')[-1]. upper(),
                    doc_id=file_name
                )
                
                doc_metadata_list.append({
                    "filename": file_name,
                    "keywords": ", ".join(keywords),
                    "summary": summary,
                    "char_count": len(file_text),
                    "page_count": page_count
                })
                
                # Chunk document
                chunks = chunk_text_advanced(
                    file_text,
                    strategy=chunking_strategy,
                    max_tokens=chunk_size,
                    overlap_tokens=chunk_overlap
                )
                
                # Add chunks to documents
                for i, chunk in enumerate(chunks):
                    documents.append({
                        "text": chunk,
                        "doc_id": file_name,
                        "chunk_id": i,
                        "category": file_name.split('.')[-1].upper(),
                        "keywords": ", ".join(keywords[:3]),
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "metadata": doc_metadata
                    })
        
        except Exception as e:
            st.warning(f"⚠️ Failed to read {file_name}: {str(e)}")
    
    if not documents:
        return None, None, None
    
    # Build vector store with metadata
    vector_store, doc_chunks = build_vector_store_with_metadata(
        documents,
        embed_model,
        show_progress=True
    )
    
    return vector_store, doc_chunks, doc_metadata_list

# ========== Build Button Logic ==========
if build_button:
    if not uploaded_files:
        st.warning("⚠️ Please upload at least one file first.")
    else:
        with st.spinner("🔄 Building vector store from uploaded files..."):
            try:
                vector_store, doc_chunks, doc_metadata = build_index_from_files(
                    uploaded_files,
                    embed_model,
                    chunking_strategy,
                    chunk_size,
                    chunk_overlap
                )
                
                if vector_store is not None:
                    # Save to session state
                    st.session_state.vector_store = vector_store
                    st.session_state.doc_chunks = doc_chunks
                    st.session_state.doc_metadata = doc_metadata
                    st.session_state.total_chunks = len(doc_chunks)
                    
                    # Save to disk with metadata
                    save_path = "./Memory"
                    save_vector_store_with_metadata(
                        vector_store, 
                        doc_chunks, 
                        save_path, 
                        "rag_index"
                    )
                    
                    st.success(f"✅ Vector store built and saved!  ({len(doc_chunks)} chunks)")
                    
                    # Show document summary
                    with st.expander("📋 Document Summary", expanded=True):
                        summary_df = pd.DataFrame(doc_metadata)
                        st.dataframe(summary_df, use_container_width=True)
                else:
                    st.error("❌ No valid content found in uploaded files.")
            
            except Exception as e:
                st.error(f"❌ Error building index: {str(e)}")
                st.exception(e)

# ========== Auto-load Existing Vector Store ==========
if st.session_state.vector_store is None:
    try:
        if os.path.exists("./Memory/rag_index"):
            with st.spinner("📂 Loading existing vector store..."):
                vector_store, metadata = load_vector_store_with_metadata(
                    "./Memory", 
                    "rag_index", 
                    embed_model
                )
                st.session_state.vector_store = vector_store
                st. session_state.total_chunks = len(metadata)
                st.info(f"✅ Existing vector store loaded ({len(metadata)} chunks)")
    except Exception as e:
        pass  # No existing store

# ========== Chat Interface ==========
st.markdown("---")
st.header("💬 Ask Questions")

query = st.text_input(
    "Enter your question",
    placeholder="e.g., What is this document about?"
)

ask_button = st.button("🔍 Ask", use_container_width=False)

# ========== Query Processing ==========
answer_text = None
retrieved_results = []

if ask_button and query:
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please build the index first from the sidebar.")
    else:
        with st.spinner("🔎 Searching and generating answer..."):
            try:
                # Search with reranking
                if enable_reranking:
                    results = search_vector_store_with_reranking(
                        st.session_state.vector_store,
                        query=query,
                        embed_model=embed_model,
                        k=top_k,
                        rerank_top_k=rerank_top_k,
                        rerank_method=rerank_method
                    )
                else:
                    # Simple search without reranking
                    search_results = st.session_state.vector_store.similarity_search_with_score(query, k=top_k)
                    results = []
                    for doc, score in search_results:
                        results.append({
                            "text": doc.page_content,
                            "metadata": doc.metadata,
                            "score": float(score),
                            "combined_score": float(score)
                        })
                
                if results:
                    # Convert to DocChunk for RAG
                    retrieved_docs = [
                        DocChunk(
                            doc_id=r["metadata"]["doc_id"],
                            chunk_id=r["metadata"]["chunk_id"],
                            text=r["text"],
                            meta=r["metadata"]
                        )
                        for r in results
                    ]
                    
                    # Generate answer
                    answer_text = answer_with_rag(query, retrieved_docs, chat_model)
                    retrieved_results = results
                else:
                    answer_text = "❌ No relevant documents found."
            
            except Exception as e:
                st.error(f"Error during search: {str(e)}")
                st.exception(e)

# ========== Display Results ==========
if answer_text or retrieved_results:
    col1, col2 = st. columns([1, 1])
    
    with col1:
        st.subheader("💡 Answer")
        if answer_text:
            st. markdown(f"<div class='answer-box'>{answer_text}</div>", unsafe_allow_html=True)
        else:
            st. caption("Answer will appear here.")
    
    with col2:
        st.subheader("📄 Retrieved Snippets")
        if retrieved_results:
            snippet_rows = []
            for rank, r in enumerate(retrieved_results, 1):
                # Show different scores if reranking is enabled
                if enable_reranking and 'original_score' in r:
                    score_display = f"🎯 {r['combined_score']:.4f} (S:{r['original_score']:.3f} K:{r['keyword_score']:.3f})"
                else:
                    score_display = f"{r. get('combined_score', r['score']):.4f}"
                
                snippet_rows. append({
                    "Rank": rank,
                    "Document": r["metadata"]["doc_id"],
                    "Score": score_display,
                    "Keywords": r["metadata"]["keywords"],
                    "Snippet": r["text"][:250] + ("..." if len(r["text"]) > 250 else "")
                })
            
            df_snippets = pd.DataFrame(snippet_rows)
            st.dataframe(df_snippets, use_container_width=True)
            
            # Download button
            csv_data = df_snippets.to_csv(index=False). encode('utf-8')
            st.download_button(
                label="⬇️ Download Retrieved Snippets (CSV)",
                data=csv_data,
                file_name=f"retrieved_snippets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.caption("Retrieved context will appear here.")

# ========== Reranking Info ==========
if enable_reranking and retrieved_results:
    with st.expander("ℹ️ Reranking Details"):
        st.markdown(f"""
        **Reranking Method:** `{rerank_method}`
        
        - **Initial Retrieval:** {rerank_top_k} candidates
        - **Final Results:** {top_k} documents
        - **Scoring:**
            - `S` = Semantic similarity (vector distance)
            - `K` = Keyword overlap score
            - `🎯` = Combined score (70% S + 30% K for hybrid)
        """)