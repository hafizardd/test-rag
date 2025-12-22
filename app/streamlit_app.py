import os
import io
import sys
from typing import List, Dict, Any
from datetime import datetime
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag. rag_core import (
    extract_elements_from_pdf,
    extract_elements_from_docx,
    summarize_text_with_groq,
    summarize_table_with_groq,
    summarize_image_with_groq,
    extract_keywords_simple,
    generate_document_summary,
    build_vector_store_with_metadata,
    save_vector_store_with_metadata,
    load_vector_store_with_metadata,
    search_vector_store_with_reranking,
    answer_with_rag,
    DocChunk,
    DocumentMetadata,
    get_groq_client,
)

from rag.history_chat import (
    process_user_query,
    save_assistant_answer,
    load_chat_history,
)

load_dotenv()

# ========== Helper Function ==========
def extract_references(chunks: List[DocChunk]) -> List[Dict[str, Any]]:
    """Extract unique references from retrieved chunks"""
    references = {}
    
    for chunk in chunks:
        doc_id = chunk.doc_id
        page_num = chunk.page_number if hasattr(chunk, 'page_number') else None
        
        if doc_id not in references:
            references[doc_id] = set()
        
        if page_num: 
            references[doc_id]. add(page_num)
    
    # Format references
    formatted_refs = []
    for doc_id, pages in references.items():
        if pages:
            sorted_pages = sorted(list(pages))
            formatted_refs.append({
                "document": doc_id,
                "pages": sorted_pages,
                "display":  f"{doc_id}, halaman {', '.join(map(str, sorted_pages))}"
            })
        else:
            formatted_refs. append({
                "document": doc_id,
                "pages": [],
                "display": doc_id
            })
    
    return formatted_refs

# ========== Page Configuration ==========
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== Custom CSS Styling ==========
def add_custom_css():
    st.markdown("""
        <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Dark theme colors */
        .main {
            background-color: #212121;
            color: #ececec;
        }
        
        /* Hide default elements */
        [data-testid="stSidebar"] {
            display:  none;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        /* Top navigation bar */
        .top-nav {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 60px;
            background-color: #212121;
            border-bottom: 1px solid #444;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
            z-index: 1000;
        }
        
        /* Model selector (left side) */
        .model-selector {
            display: flex;
            align-items: center;
            gap: 8px;
            background-color: #2f2f2f;
            padding: 8px 16px;
            border-radius:  8px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        
        .model-selector:hover {
            background-color: #3f3f3f;
        }
        
        .model-name {
            color: #ececec;
            font-weight: 500;
            font-size: 14px;
        }
        
        /* Chat container */
        .chat-container {
            margin-top: 80px;
            margin-bottom: 150px;
            padding: 20px;
            max-width: 900px;
            margin-left:  auto;
            margin-right:  auto;
        }
        
        /* Welcome message */
        .welcome-message {
            text-align: center;
            margin-top: 150px;
            margin-bottom: 50px;
        }
        
        .welcome-title {
            font-size: 32px;
            font-weight: 600;
            color: #ececec;
            margin-bottom:  30px;
        }
        
        /* Message bubbles */
        .user-message {
            background-color: #2f2f2f;
            color: #ececec;
            padding: 16px 20px;
            border-radius: 20px;
            margin:  12px 0;
            max-width: 75%;
            margin-left: auto;
            word-wrap: break-word;
            line-height: 1.6;
        }
        
        . assistant-message {
            background-color: #2f2f2f;
            color: #ececec;
            padding: 16px 20px;
            border-radius:  20px;
            margin:  12px 0;
            max-width: 75%;
            word-wrap: break-word;
            line-height: 1.6;
        }
        
        /* Input container */
        .input-container {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            background-color: #212121;
            padding: 20px;
            z-index: 999;
        }
        
        . input-wrapper {
            max-width: 900px;
            margin:  0 auto;
            position: relative;
        }
        
        /* Plus button */
        .plus-button {
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            background:  transparent;
            border: none;
            color: #ececec;
            font-size: 24px;
            cursor: pointer;
            z-index: 10;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 6px;
            transition: background-color 0.2s;
        }
        
        . plus-button:hover {
            background-color: #3f3f3f;
        }
        
        /* Popup menu */
        .popup-menu {
            position: absolute;
            bottom: 60px;
            left: 12px;
            background-color: #2f2f2f;
            border-radius: 12px;
            padding: 8px;
            min-width: 250px;
            box-shadow:  0 4px 20px rgba(0, 0, 0, 0.5);
            z-index: 1001;
        }
        
        .popup-menu-item {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 12px 16px;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s;
            color: #ececec;
        }
        
        .popup-menu-item:hover {
            background-color: #3f3f3f;
        }
        
        .popup-menu-icon {
            font-size: 18px;
        }
        
        /* Text input styling */
        .stTextInput > div > div > input {
            background-color: #2f2f2f ! important;
            color: #ececec !important;
            border: 1px solid #444 !important;
            border-radius: 24px !important;
            padding:  14px 100px 14px 50px !important;
            font-size: 15px !important;
        }
        
        .stTextInput > div > div > input: focus {
            border-color: #666 !important;
            box-shadow: none !important;
        }
        
        /* Send button */
        .send-button {
            position: absolute;
            right: 8px;
            top: 50%;
            transform:  translateY(-50%);
            background-color: #ececec;
            border: none;
            color: #212121;
            width: 36px;
            height: 36px;
            border-radius:  50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            transition: background-color 0.2s;
        }
        
        .send-button:hover {
            background-color: #d0d0d0;
        }
        
        . send-button:disabled {
            background-color: #3f3f3f;
            color: #666;
            cursor: not-allowed;
        }
        
        /* Modal overlay */
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom:  0;
            background-color: rgba(0, 0, 0, 0.85);
            z-index: 9998;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        .modal-content {
            background-color: #2f2f2f;
            padding: 30px;
            border-radius:  16px;
            max-width:  600px;
            width: 90%;
            max-height: 80vh;
            overflow-y:  auto;
            border: 1px solid #444;
            z-index: 9999;
            position: relative;
        }
        
        .modal-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .modal-title {
            font-size: 20px;
            font-weight:  600;
            color: #ececec;
        }
        
        .close-button {
            background: transparent;
            border: none;
            color: #ececec;
            font-size: 24px;
            cursor: pointer;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 6px;
            transition: background-color 0.2s;
        }
        
        .close-button:hover {
            background-color: #3f3f3f;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #ececec ! important;
            color: #212121 !important;
            border-radius: 8px !important;
            border: none !important;
            padding: 10px 20px !important;
            font-weight: 600 !important;
            transition: background-color 0.2s ! important;
        }
        
        .stButton > button:hover {
            background-color: #d0d0d0 ! important;
        }
        
        /* File uploader */
        [data-testid="stFileUploader"] {
            background-color: #3f3f3f;
            border: 2px dashed #666;
            border-radius: 12px;
            padding: 20px;
        }
        
        /* Selectbox */
        .stSelectbox > div > div {
            background-color: #3f3f3f ! important;
            color: #ececec !important;
            border: 1px solid #666 !important;
            border-radius: 8px !important;
        }
        
        /* Slider */
        .stSlider > div > div > div {
            background-color: #3f3f3f !important;
        }
        
        /* Checkbox */
        .stCheckbox {
            color: #ececec !important;
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: #212121;
        }
        
        ::-webkit-scrollbar-thumb {
            background: #444;
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
        
        /* Success/Warning/Error messages */
        .stSuccess, .stWarning, .stError {
            background-color: #3f3f3f ! important;
            color: #ececec !important;
            border-radius: 8px !important;
        }
        
        /* Reference box */
        .reference-box {
            background-color: #3f3f3f;
            padding: 12px;
            border-radius:  8px;
            border-left: 3px solid #666;
            margin-top: 12px;
            font-size: 0.9em;
            color: #b0b0b0;
        }
        </style>
    """, unsafe_allow_html=True)

add_custom_css()

# ========== Session State Initialization ==========
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = []
if "doc_metadata" not in st.session_state:
    st.session_state.doc_metadata = None
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())
if "references" not in st.session_state:
    st.session_state.references = []
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False
if "show_upload_menu" not in st.session_state:
    st.session_state.show_upload_menu = False
if "show_model_selector" not in st.session_state:
    st.session_state.show_model_selector = False
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_model" not in st.session_state:
    st.session_state.chat_model = "llama-3.1-8b-instant"
if "enable_reranking" not in st.session_state:
    st.session_state.enable_reranking = True
if "rerank_method" not in st.session_state:
    st.session_state.rerank_method = "hybrid"
if "rerank_top_k" not in st.session_state:
    st.session_state.rerank_top_k = 20
if "top_k" not in st.session_state:
    st.session_state.top_k = 5
if "chunking_strategy" not in st. session_state:
    st. session_state.chunking_strategy = "recursive"
if "chunk_size" not in st.session_state:
    st.session_state.chunk_size = 1000
if "chunk_overlap" not in st.session_state:
    st.session_state.chunk_overlap = 100

# ========== Load Embedding Model (Cached) ==========
@st.cache_resource
def load_embedding_model():
    """Load SentenceTransformer model (cached)"""
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')

embed_model = load_embedding_model()
embed_model_name = 'sentence-transformers/all-mpnet-base-v2'

# ========== Build Index Function ==========
def build_multimodal_index_from_files(
    files: List[io.BytesIO],
    embed_model: SentenceTransformer,
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int
):
    """Build vector store from uploaded files with multimodal support and page tracking"""
    documents = []
    doc_metadata_list = []
    groq_client = get_groq_client()
    
    for f in files:
        file_name = getattr(f, "name", "uploaded")
        file_ext = file_name.split('.')[-1]. lower()
        
        # Save file temporarily
        temp_path = f"./temp_{file_name}"
        with open(temp_path, "wb") as temp_file:
            temp_file.write(f.read())
        
        try:
            # Extract elements based on file type
            if file_ext == "pdf":
                elements, images_base64, page_map = extract_elements_from_pdf(temp_path)
            elif file_ext == "docx": 
                elements, images_base64, page_map = extract_elements_from_docx(temp_path)
            else:
                st.warning(f"⚠️ Unsupported file type: {file_name}")
                continue
            
            if not elements:
                st.warning(f"⚠️ No content extracted from {file_name}")
                continue
            
            # Process text and tables
            texts = []
            tables = []
            
            for idx, element in enumerate(elements):
                element_type = str(type(element))
                if "Table" in element_type: 
                    tables.append((element, page_map. get(idx)))
                elif "CompositeElement" in element_type: 
                    texts.append((element, page_map.get(idx)))
            
            # Create metadata
            all_text = " ".join([str(el) for el, _ in texts])
            keywords = extract_keywords_simple(all_text, top_n=5)
            summary = generate_document_summary(all_text, max_length=150)
            
            doc_metadata = DocumentMetadata(
                filename=file_name,
                file_size=len(all_text),
                creation_date=datetime.now(),
                page_count=len(elements),
                keywords=keywords,
                summary=summary,
                document_type=file_ext. upper(),
                doc_id=file_name
            )
            
            doc_metadata_list.append({
                "filename": file_name,
                "keywords": ", ".join(keywords),
                "summary": summary,
                "char_count": len(all_text),
                "page_count":  len(elements),
                "content_types": f"Text:  {len(texts)}, Tables: {len(tables)}, Images: {len(images_base64)}"
            })
            
            chunk_id = 0
            
            # Process text chunks with page numbers
            for text_elem, page_num in texts: 
                text_content = str(text_elem)
                if text_content. strip():
                    text_summary = summarize_text_with_groq(text_content, groq_client)
                    
                    documents.append({
                        "text": text_summary,
                        "doc_id": file_name,
                        "chunk_id": chunk_id,
                        "category": "TEXT",
                        "keywords": ", ".join(keywords[: 3]),
                        "chunk_size": chunk_size,
                        "chunk_overlap": chunk_overlap,
                        "content_type": "text",
                        "metadata": doc_metadata,
                        "page_number": page_num
                    })
                    chunk_id += 1
            
            # Process tables with page numbers
            for table_elem, page_num in tables: 
                if hasattr(table_elem. metadata, 'text_as_html'):
                    table_html = table_elem.metadata.text_as_html
                    table_summary = summarize_table_with_groq(table_html, groq_client)
                    
                    documents.append({
                        "text": table_summary,
                        "doc_id": file_name,
                        "chunk_id": chunk_id,
                        "category": "TABLE",
                        "keywords": ", ".join(keywords[:3]),
                        "chunk_size": chunk_size,
                        "chunk_overlap":  chunk_overlap,
                        "content_type": "table",
                        "metadata": doc_metadata,
                        "page_number": page_num
                    })
                    chunk_id += 1
            
            # Process images
            for img_idx, img_base64 in enumerate(images_base64):
                img_description = summarize_image_with_groq(img_base64, groq_client)
                
                documents.append({
                    "text": img_description,
                    "doc_id": file_name,
                    "chunk_id": chunk_id,
                    "category": "IMAGE",
                    "keywords": ", ".join(keywords[:3]),
                    "chunk_size": chunk_size,
                    "chunk_overlap":  chunk_overlap,
                    "content_type": "image",
                    "metadata": doc_metadata,
                    "page_number": None
                })
                chunk_id += 1
        
        except Exception as e:
            st. warning(f"⚠️ Failed to process {file_name}: {str(e)}")
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    if not documents:
        return None, None, None
    
    # Build vector store with page numbers
    vector_store, doc_chunks = build_vector_store_with_metadata(
        documents,
        embed_model,
        show_progress=True
    )
    
    return vector_store, doc_chunks, doc_metadata_list

# ========== Auto-load Existing Vector Store ==========
if st.session_state.vector_store is None:
    try:
        if os.path.exists("./Memory/rag_multimodal_index"):
            vector_store, metadata = load_vector_store_with_metadata(
                "./Memory", 
                "rag_multimodal_index", 
                embed_model
            )
            st.session_state.vector_store = vector_store
            st.session_state.total_chunks = len(metadata)
    except Exception: 
        pass

# Load chat history from database
if not st.session_state.chat_messages:
    history = load_chat_history(st.session_state.session_id)
    st.session_state.chat_messages = history

# ========== Top Navigation Bar ==========
col_left, col_right = st. columns([1, 5])

with col_left:
    if st.button(f"🤖 {st.session_state.chat_model}", key="model_btn", help="Change model"):
        st.session_state.show_model_selector = not st.session_state.show_model_selector

# ========== Model Selector Modal ==========
if st.session_state.show_model_selector:
    @st.dialog("🤖 Model & Settings")
    def show_model_settings():
        # Model selection
        chat_model = st.selectbox(
            "Chat Model",
            options=["llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            index=0 if st.session_state.chat_model == "llama-3.1-8b-instant" else 1,
            key="model_select"
        )
        st.session_state.chat_model = chat_model
        
        st.markdown("---")
        
        # Retrieval settings
        st.markdown("#### 🎯 Retrieval Settings")
        st.session_state.enable_reranking = st.checkbox(
            "Enable Reranking", 
            value=st.session_state.enable_reranking
        )
        
        if st.session_state.enable_reranking:
            st. session_state.rerank_method = st.selectbox(
                "Reranking Method",
                options=["hybrid", "keyword", "semantic"],
                index=["hybrid", "keyword", "semantic"]. index(st.session_state. rerank_method)
            )
            st.session_state.rerank_top_k = st.slider(
                "Initial Retrieval", 
                10, 50, st.session_state.rerank_top_k, 5
            )
        
        st.session_state.top_k = st.slider(
            "Final Results", 
            1, 10, st.session_state.top_k
        )
        
        st. markdown("---")
        
        # Chunking settings
        st.markdown("#### ⚙️ Chunking Settings")
        st.session_state.chunking_strategy = st. selectbox(
            "Strategy",
            options=["recursive", "semantic", "paragraph", "simple"],
            index=["recursive", "semantic", "paragraph", "simple"].index(st.session_state.chunking_strategy)
        )
        st.session_state.chunk_size = st.slider(
            "Chunk Size (chars)", 
            500, 2000, st.session_state.chunk_size, 100
        )
        st.session_state.chunk_overlap = st.slider(
            "Chunk Overlap (chars)", 
            50, 500, st.session_state.chunk_overlap, 50
        )
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✓ Apply", use_container_width=True):
                st.session_state.show_model_selector = False
                st. rerun()
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_messages = []
                st.session_state.vector_store = None
                st. session_state.doc_chunks = []
                st.session_state. total_chunks = 0
                st.session_state.show_model_selector = False
                st. rerun()
    
    show_model_settings()

# ========== Upload Settings Modal ==========
if st.session_state.show_settings:
    @st.dialog("📤 Upload Documents")
    def show_upload_dialog():
        uploaded_files = st.file_uploader(
            "Select PDF or DOCX files",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files: 
            st.info(f"✓ {len(uploaded_files)} file(s) selected")
            
            col_build, col_cancel = st.columns(2)
            with col_build:
                if st.button("🔨 Build Index", use_container_width=True):
                    with st.spinner("🔄 Building vector store..."):
                        try:
                            vector_store, doc_chunks, doc_metadata = build_multimodal_index_from_files(
                                uploaded_files,
                                embed_model,
                                st.session_state.chunking_strategy,
                                st.session_state.chunk_size,
                                st.session_state.chunk_overlap
                            )
                            
                            if vector_store is not None:
                                st. session_state.vector_store = vector_store
                                st. session_state.doc_chunks = doc_chunks
                                st. session_state. doc_metadata = doc_metadata
                                st.session_state.total_chunks = len(doc_chunks)
                                
                                save_path = "./Memory"
                                save_vector_store_with_metadata(
                                    vector_store,
                                    doc_chunks,
                                    save_path,
                                    "rag_multimodal_index"
                                )
                                
                                st.success(f"✅ Index built!  ({len(doc_chunks)} chunks)")
                                st.session_state.show_settings = False
                                st.rerun()
                            else:
                                st. error("❌ No valid content found")
                        except Exception as e: 
                            st.error(f"❌ Error:  {str(e)}")
            
            with col_cancel: 
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_settings = False
                    st.rerun()
        else:
            if st.button("Close", use_container_width=True):
                st.session_state.show_settings = False
                st.rerun()
    
    show_upload_dialog()

# ========== Chat Display ==========
if not st.session_state.chat_messages:
    # Welcome screen
    st.markdown("""
        <div class="welcome-message">
            <div class="welcome-title">Apa yang bisa saya bantu? </div>
        </div>
    """, unsafe_allow_html=True)
else:
    # Chat messages
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.chat_messages:
        if message["role"] == "user": 
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st. markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ========== Input Area with Plus Button ==========
st.markdown("---")

# Create columns for input layout
col_plus = st.columns([1, 20, 1])

# Plus button column
with col_plus[0]: 
    if st.button("➕", key="plus_btn", help="Upload files"):
        st.session_state.show_settings = True
        st.rerun()

# Input field
with col_plus[1]: 
    query = st.text_input(
        "Message",
        placeholder="Tanyakan apa saja...",
        label_visibility="collapsed",
        key="query_input"
    )

# Send button
with col_plus[2]:
    ask_button = st.button("↑", key="send_btn", help="Send", disabled=not query)

# ========== Query Processing ==========
if ask_button and query:
    if st.session_state.vector_store is None:
        st.warning("⚠️ Please upload and build index first via + button")
    else:
        # Add user message to chat
        st.session_state.chat_messages.append({"role": "user", "content": query})
        
        with st.spinner("💭 Thinking..."):
            try:
                session_id = st.session_state.session_id
                
                # Process query with history
                standalone_query = process_user_query(
                    session_id=session_id,
                    user_query=query
                )
                
                # Search vector store
                if st.session_state.enable_reranking:
                    results = search_vector_store_with_reranking(
                        st. session_state.vector_store,
                        query=standalone_query,
                        embed_model=embed_model,
                        k=st.session_state.top_k,
                        rerank_top_k=st.session_state.rerank_top_k,
                        rerank_method=st.session_state.rerank_method
                    )
                else:
                    search_results = st.session_state.vector_store.similarity_search_with_score(
                        standalone_query, k=st.session_state.top_k
                    )
                    results = []
                    for doc, score in search_results: 
                        results.append({
                            "text": doc.page_content,
                            "metadata": doc.metadata,
                            "score": float(score),
                            "combined_score": float(score)
                        })
                
                if results:
                    retrieved_docs = [
                        DocChunk(
                            doc_id=r["metadata"]["doc_id"],
                            chunk_id=r["metadata"]["chunk_id"],
                            text=r["text"],
                            meta=r["metadata"],
                            content_type=r["metadata"]. get("content_type", "text"),
                            page_number=r["metadata"].get("page_number")
                        )
                        for r in results
                    ]
                    
                    # Generate answer
                    answer_text = answer_with_rag(query, retrieved_docs, st.session_state.chat_model)
                    
                    # Extract references
                    references = extract_references(retrieved_docs)
                    
                    # Format answer with references
                    if references:
                        ref_text = "\n\n📚 **Referensi:**\n"
                        for ref in references: 
                            if ref['pages']:
                                pages_str = ", ".join(map(str, ref['pages']))
                                ref_text += f"- {ref['document']}, halaman {pages_str}\n"
                            else:
                                ref_text += f"- {ref['document']}\n"
                        answer_text += ref_text
                    
                    # Save answer
                    save_assistant_answer(
                        session_id=session_id,
                        answer=answer_text
                    )
                    
                    # Add assistant message to chat
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": answer_text
                    })
                else:
                    answer_text = "❌ Tidak menemukan konten yang relevan."
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": answer_text
                    })
                
                st.rerun()
                
            except Exception as e: 
                error_msg = f"❌ Error: {str(e)}"
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
                st.rerun()