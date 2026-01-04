import os
import io
import sys
from typing import List, Dict, Any
from datetime import datetime
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.rag_core import (
    extract_elements_from_pdf,
    extract_elements_from_docx,
    summarize_text_with_groq,
    summarize_table_with_groq,
    summarize_image_with_groq,
    extract_keywords_simple,
    generate_document_summary,
    search_vector_store_with_reranking,
    answer_with_rag,
    results_to_doc_chunks,
    DocChunk,
    DocumentMetadata,
    get_groq_client,
    SupabaseVectorStore,
)

from rag.history_chat import (
    process_user_query,
    save_assistant_answer,
    load_chat_history,
    clear_chat_history,
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
            references[doc_id].add(page_num)

    formatted_refs = []
    for doc_id, pages in references.items():
        if pages:
            sorted_pages = sorted(list(pages))
            formatted_refs.append({
                "document":  doc_id,
                "pages": sorted_pages,
                "display":  f"{doc_id}, halaman {', '.join(map(str, sorted_pages))}"
            })
        else:
            formatted_refs. append({
                "document": doc_id,
                "pages":  [],
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
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
        . main { background-color: #212121; color: #ececec; }
        [data-testid="stSidebar"] { display: none; }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        .welcome-message { text-align: center; margin-top: 150px; margin-bottom: 50px; }
        .welcome-title { font-size: 32px; font-weight: 600; color: #ececec; margin-bottom: 30px; }
        
        .user-message {
            background-color: #2f2f2f; color: #ececec; padding: 16px 20px;
            border-radius: 20px; margin:  12px 0; max-width: 75%;
            margin-left: auto; word-wrap: break-word; line-height: 1.6;
        }
        
        . assistant-message {
            background-color: #2f2f2f; color:  #ececec; padding: 16px 20px;
            border-radius: 20px; margin: 12px 0; max-width: 75%;
            word-wrap: break-word; line-height: 1.6;
        }
        
        . stTextInput > div > div > input {
            background-color: #2f2f2f !important; color: #ececec !important;
            border: 1px solid #444 !important; border-radius: 24px !important;
            padding: 14px 100px 14px 50px !important; font-size: 15px !important;
        }
        
        .stButton > button {
            background-color: #ececec !important; color: #212121 !important;
            border-radius: 8px !important; border: none !important;
            padding: 10px 20px !important; font-weight: 600 !important;
        }
        
        [data-testid="stFileUploader"] {
            background-color:  #3f3f3f; border: 2px dashed #666;
            border-radius: 12px; padding: 20px;
        }
        
        . status-badge {
            display: inline-block; padding: 4px 12px; border-radius: 20px;
            font-size: 12px; margin-left: 10px;
        }
        . status-connected { background-color: #065f46; color: #34d399; }
        .status-disconnected { background-color:  #7f1d1d; color: #fca5a5; }
        </style>
    """, unsafe_allow_html=True)

add_custom_css()


# ========== Session State Initialization ==========
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0
if "session_id" not in st.session_state:
    # Use a persistent session ID (you could also use user auth here)
    st.session_state.session_id = "default_user_session"
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False
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
if "supabase_connected" not in st.session_state:
    st.session_state.supabase_connected = False


# ========== Load Embedding Model (Cached) ==========
@st.cache_resource
def load_embedding_model():
    """Load SentenceTransformer model (cached)"""
    return SentenceTransformer('sentence-transformers/all-mpnet-base-v2', device='cpu')


@st.cache_resource
def get_vector_store(_embed_model):
    """Get Supabase vector store (cached)"""
    try:
        store = SupabaseVectorStore(_embed_model)
        return store
    except Exception as e: 
        st.error(f"❌ Failed to connect to Supabase: {str(e)}")
        return None


embed_model = load_embedding_model()


# ========== Initialize Supabase Vector Store ==========
if st.session_state.vector_store is None:
    try:
        vector_store = get_vector_store(embed_model)
        if vector_store: 
            st.session_state. vector_store = vector_store
            st.session_state.total_chunks = vector_store.get_document_count()
            st.session_state.supabase_connected = True
            print(f"✅ Connected to Supabase with {st.session_state.total_chunks} chunks")
    except Exception as e:
        st.session_state.supabase_connected = False
        print(f"❌ Failed to connect to Supabase: {str(e)}")


# ========== Load Chat History from Database on Startup ==========
if not st.session_state.chat_messages:
    history = load_chat_history(st.session_state.session_id)
    st.session_state.chat_messages = history
    if history:
        print(f"📚 Loaded {len(history)} messages from database")


# ========== Build Index Function ==========
def build_multimodal_index_from_files(
    files: List[io.BytesIO],
    embed_model: SentenceTransformer,
    vector_store:  SupabaseVectorStore,
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int
):
    """Build vector store from uploaded files"""
    documents = []
    doc_metadata_list = []
    groq_client = get_groq_client()
    
    for f in files:
        file_name = getattr(f, "name", "uploaded")
        file_ext = file_name.split('.')[-1].lower()
        
        # Check if document already exists
        if vector_store. document_exists(file_name):
            st.warning(f"⚠️ Document '{file_name}' already exists.  Skipping...")
            continue
        
        temp_path = f"./temp_{file_name}"
        with open(temp_path, "wb") as temp_file:
            temp_file.write(f.read())
        
        try:
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
            
            # Process elements
            texts = []
            tables = []
            
            for idx, element in enumerate(elements):
                element_type = str(type(element))
                if hasattr(element, 'element_type'):
                    if element.element_type == "table":
                        tables.append((element, page_map. get(idx)))
                    else:
                        texts.append((element, page_map.get(idx)))
                elif "Table" in element_type:
                    tables.append((element, page_map.get(idx)))
                elif "CompositeElement" in element_type:
                    texts.append((element, page_map.get(idx)))
                else:
                    texts.append((element, page_map.get(idx)))
            
            all_text = " ".join([str(el) for el, _ in texts])
            keywords = extract_keywords_simple(all_text, top_n=5)
            
            doc_metadata_list.append({
                "filename": file_name,
                "keywords": ", ".join(keywords),
                "content_types": f"Text:  {len(texts)}, Tables: {len(tables)}, Images: {len(images_base64)}"
            })
            
            chunk_id = 0
            
            # Process text chunks
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
                        "filename": file_name,
                        "page_number":  page_num
                    })
                    chunk_id += 1
            
            # Process tables
            for table_elem, page_num in tables:
                table_content = str(table_elem)
                if hasattr(table_elem, 'metadata') and hasattr(table_elem.metadata, 'text_as_html'):
                    table_content = table_elem.metadata.text_as_html
                
                if table_content.strip():
                    table_summary = summarize_table_with_groq(table_content, groq_client)
                    
                    documents.append({
                        "text": table_summary,
                        "doc_id": file_name,
                        "chunk_id": chunk_id,
                        "category": "TABLE",
                        "keywords": ", ".join(keywords[:3]),
                        "chunk_size": chunk_size,
                        "chunk_overlap":  chunk_overlap,
                        "content_type": "table",
                        "filename": file_name,
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
                    "content_type": "image",
                    "filename": file_name,
                    "page_number":  None
                })
                chunk_id += 1
        
        except Exception as e:
            st. warning(f"⚠️ Failed to process {file_name}: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    if not documents:
        return 0, None
    
    # Add documents to Supabase
    total_added = vector_store.add_documents(documents, show_progress=True)
    
    return total_added, doc_metadata_list


# ========== Top Navigation Bar ==========
col_left, col_right = st.columns([3, 1])

with col_left:
    if st.button(f"🤖 {st.session_state.chat_model}", key="model_btn"):
        st.session_state.show_model_selector = not st.session_state.show_model_selector

with col_right:
    if st.session_state.supabase_connected:
        st.markdown(f'<span class="status-badge status-connected">🟢 {st.session_state.total_chunks} docs</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-disconnected">🔴 Offline</span>', unsafe_allow_html=True)


# ========== Model Selector Modal ==========
if st.session_state.show_model_selector:
    @st.dialog("🤖 Model & Settings")
    def show_model_settings():
        chat_model = st.selectbox(
            "Chat Model",
            options=["llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            index=0 if st.session_state.chat_model == "llama-3.1-8b-instant" else 1,
        )
        st.session_state.chat_model = chat_model
        
        st.markdown("---")
        st.markdown("#### 🎯 Retrieval Settings")
        
        st.session_state.enable_reranking = st.checkbox("Enable Reranking", value=st.session_state.enable_reranking)
        
        if st.session_state.enable_reranking:
            st. session_state.rerank_method = st.selectbox(
                "Reranking Method",
                options=["hybrid", "keyword", "semantic"],
                index=["hybrid", "keyword", "semantic"]. index(st.session_state. rerank_method)
            )
            st.session_state.rerank_top_k = st.slider("Initial Retrieval", 10, 50, st.session_state.rerank_top_k, 5)
        
        st.session_state.top_k = st.slider("Final Results", 1, 10, st.session_state.top_k)
        
        st.markdown("---")
        st.markdown("#### 🗄️ Vector Store")
        
        if st.session_state.vector_store:
            doc_count = st.session_state.vector_store.get_document_count()
            st.info(f"📊 Total chunks: {doc_count}")
            
            doc_ids = st.session_state.vector_store.get_all_doc_ids()
            if doc_ids:
                st. markdown("**Indexed Documents:**")
                for doc_id in doc_ids[: 5]: 
                    st.markdown(f"- {doc_id}")
                if len(doc_ids) > 5:
                    st.markdown(f"_... and {len(doc_ids) - 5} more_")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("✓ Apply", use_container_width=True):
                st.session_state.show_model_selector = False
                st. rerun()
        with col2:
            if st. button("🗑️ Clear Chat", use_container_width=True):
                # Clear chat history from database
                clear_chat_history(st.session_state.session_id)
                st.session_state.chat_messages = []
                st.session_state. show_model_selector = False
                st. rerun()
        with col3:
            if st.button("🧹 Clear Docs", use_container_width=True, type="secondary"):
                if st.session_state.vector_store:
                    with st.spinner("Clearing... "):
                        st.session_state.vector_store.delete_all()
                        st.session_state.total_chunks = 0
                    st.success("✅ Cleared!")
                    st.session_state.show_model_selector = False
                    st.rerun()
    
    show_model_settings()


# ========== Upload Modal ==========
if st.session_state.show_settings:
    @st.dialog("📤 Upload Documents")
    def show_upload_dialog():
        if not st.session_state.supabase_connected:
            st.error("❌ Not connected to Supabase.")
            if st.button("Close"):
                st.session_state. show_settings = False
                st.rerun()
            return
        
        # Show existing documents
        if st.session_state.vector_store:
            doc_ids = st.session_state.vector_store.get_all_doc_ids()
            if doc_ids:
                st.info(f"📚 Existing documents: {', '.join(doc_ids[: 3])}{'...' if len(doc_ids) > 3 else ''}")
        
        uploaded_files = st.file_uploader(
            "Select PDF or DOCX files",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files: 
            st.info(f"✓ {len(uploaded_files)} file(s) selected")
            
            col1, col2 = st. columns(2)
            with col1:
                if st.button("🔨 Build Index", use_container_width=True):
                    with st.spinner("🔄 Processing..."):
                        try:
                            total_added, doc_metadata = build_multimodal_index_from_files(
                                uploaded_files,
                                embed_model,
                                st.session_state.vector_store,
                                st.session_state. chunking_strategy,
                                st.session_state.chunk_size,
                                st.session_state.chunk_overlap
                            )
                            
                            if total_added > 0:
                                st.session_state.total_chunks = st.session_state.vector_store.get_document_count()
                                st.success(f"✅ Added {total_added} chunks!")
                                st.session_state.show_settings = False
                                st.rerun()
                            else:
                                st. warning("⚠️ No new content added (files may already exist)")
                        except Exception as e: 
                            st.error(f"❌ Error: {str(e)}")
            
            with col2:
                if st.button("Cancel", use_container_width=True):
                    st.session_state.show_settings = False
                    st.rerun()
        else:
            if st.button("Close", use_container_width=True):
                st.session_state. show_settings = False
                st.rerun()
    
    show_upload_dialog()


# ========== Chat Display ==========
if not st.session_state.chat_messages:
    st.markdown("""
        <div class="welcome-message">
            <div class="welcome-title">Apa yang bisa saya bantu? </div>
        </div>
    """, unsafe_allow_html=True)
else:
    for message in st.session_state. chat_messages:
        if message["role"] == "user": 
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)


# ========== Input Area ==========
st.markdown("---")

col_plus, col_input, col_send = st.columns([1, 18, 1])

with col_plus:
    if st.button("➕", key="plus_btn", help="Upload files"):
        st.session_state. show_settings = True
        st.rerun()

with col_input:
    query = st.text_input(
        "Message",
        placeholder="Tanyakan apa saja...  (Anda bisa merujuk ke pertanyaan sebelumnya)",
        label_visibility="collapsed",
        key="query_input"
    )

with col_send:
    ask_button = st.button("↑", key="send_btn", disabled=not query)


# ========== Query Processing ==========
if ask_button and query:
    if st.session_state.vector_store is None:
        st. warning("⚠️ Not connected to Supabase.  Please check your configuration.")
    elif st.session_state.total_chunks == 0:
        st.warning("⚠️ No documents indexed.  Please upload documents first via + button.")
    else:
        # Add user message to chat UI immediately
        st.session_state.chat_messages.append({"role": "user", "content":  query})
        
        with st.spinner("💭 Thinking..."):
            try:
                session_id = st.session_state.session_id
                
                # Get current chat history BEFORE processing
                # This is for passing to LLM for context understanding
                # Exclude the message we just added
                current_history = st.session_state.chat_messages[:-1]
                
                # Process query with history awareness
                # This will reformulate "berikan contohnya" -> "berikan contoh tentang order dalam Holland Schema"
                standalone_query = process_user_query(
                    session_id=session_id,
                    user_query=query
                )
                
                print(f"🔍 Searching with query: '{standalone_query}'")
                
                # Search vector store using the reformulated query
                if st. session_state.enable_reranking:
                    results = search_vector_store_with_reranking(
                        st. session_state.vector_store,
                        query=standalone_query,
                        embed_model=embed_model,
                        k=st.session_state.top_k,
                        rerank_top_k=st.session_state.rerank_top_k,
                        rerank_method=st.session_state. rerank_method
                    )
                else:
                    results = st.session_state.vector_store.similarity_search(
                        query=standalone_query,
                        k=st.session_state.top_k
                    )
                
                print(f"📊 Retrieved {len(results)} documents")
                
                if results:
                    # Convert search results to DocChunk objects
                    retrieved_docs = results_to_doc_chunks(results)
                    
                    # Generate answer with: 
                    # 1. Reformulated query (standalone_query)
                    # 2. Retrieved documents (retrieved_docs)  
                    # 3. Chat history for context (current_history)
                    answer_text = answer_with_rag(
                        query=standalone_query,
                        retrieved=retrieved_docs,
                        chat_model=st.session_state.chat_model,
                        chat_history=current_history  # <-- This is the key addition!
                    )
                    
                    # Extract references from retrieved documents
                    references = extract_references(retrieved_docs)
                    
                    # Add references if not already in answer
                    if references:
                        has_refs = any(marker in answer_text for marker in ["Referensi:", "References:", "📚"])
                        if not has_refs:
                            ref_text = "\n\n📚 **Referensi:**\n"
                            for ref in references: 
                                if ref['pages']: 
                                    pages_str = ", ".join(map(str, ref['pages']))
                                    ref_text += f"- {ref['document']}, halaman {pages_str}\n"
                                else:
                                    ref_text += f"- {ref['document']}\n"
                            answer_text += ref_text
                    
                    # Save assistant answer to database
                    save_assistant_answer(session_id=session_id, answer=answer_text)
                    
                    # Add to chat UI
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content": answer_text
                    })
                else:
                    # No relevant documents found
                    answer_text = "❌ Tidak menemukan konten yang relevan dalam dokumen.  Coba ajukan pertanyaan dengan kata kunci yang lebih spesifik."
                    save_assistant_answer(session_id=session_id, answer=answer_text)
                    st.session_state.chat_messages.append({
                        "role": "assistant",
                        "content":  answer_text
                    })
                
                # Refresh the page to show new messages
                st.rerun()
                
            except Exception as e: 
                error_msg = f"❌ Error: {str(e)}"
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content":  error_msg
                })
                import traceback
                print(traceback.format_exc())
                st.rerun()