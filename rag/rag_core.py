import os
import json
import faiss
import time
import random
import base64
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm
from collections import Counter
from io import BytesIO
from PIL import Image

import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community. docstore.in_memory import InMemoryDocstore
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

# Import PDF libraries with fallback
UNSTRUCTURED_AVAILABLE = False
try:
    from unstructured.partition.pdf import partition_pdf
    from unstructured.partition. docx import partition_docx
    UNSTRUCTURED_AVAILABLE = True
    print("✅ Unstructured library available")
except ImportError:
    print("⚠️ Unstructured library not available, using fallback PDF processing")

# Fallback PDF libraries
try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False
    print("⚠️ pdfplumber not available")

try:
    from PyPDF2 import PdfReader
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    print("⚠️ PyPDF2 not available")

try:
    from docx import Document
    PYTHON_DOCX_AVAILABLE = True
except ImportError:
    PYTHON_DOCX_AVAILABLE = False
    print("⚠️ python-docx not available")

load_dotenv()

# ========== Configuration ==========
def get_groq_client() -> Groq:
    """Get GROQ API client"""
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set.  Provide it via env or a . env file.")
    return Groq(api_key=api_key)

# REMOVED: get_gemini_client() function - no longer needed

# ========== Document Metadata Dataclass ==========
@dataclass
class DocumentMetadata:
    """Metadata untuk dokumen"""
    filename: str
    file_size:  int
    creation_date: datetime
    page_count: int = 0
    keywords: Optional[List[str]] = None
    summary: str = ""
    document_type:  str = "general"
    doc_id: str = ""


@dataclass
class DocChunk:
    """Document chunk with metadata"""
    doc_id:  str
    chunk_id: int
    text: str
    meta: Dict
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    metadata: Optional[DocumentMetadata] = None
    content_type: str = "text"
    page_number: Optional[int] = None


# ========== Simple Element Class for Fallback ==========
class SimpleElement:  
    """Simple element class for fallback PDF processing"""
    def __init__(self, text: str, page_num: Optional[int] = None, element_type: str = "text"):
        self.text = text
        self.element_type = element_type
        self.metadata = type('obj', (object,), {
            'page_number': page_num,
            'text_as_html': text if element_type == "table" else None
        })()
    
    def __str__(self):
        return self.text

# ========== Multimodal Extraction Functions ==========
def extract_elements_from_pdf(file_path: str, output_path: str = "./content/") -> Tuple[List, List[str], Dict]:
    """
    Extract elements from PDF including text, tables, and images with fallback support
    Returns: (elements, image_base64_list, page_map)
    """
    print(f"🔍 Extracting from PDF: {file_path}")
    
    # Try Method 1: Unstructured (requires Poppler)
    if UNSTRUCTURED_AVAILABLE: 
        try:
            from unstructured.partition.pdf import partition_pdf
            
            print("📄 Trying unstructured library...")
            chunks = partition_pdf(
                filename=file_path,
                infer_table_structure=True,
                strategy="hi_res",
                extract_image_block_types=["Image"],
                extract_image_block_to_payload=True,
                chunking_strategy="by_title",
                max_characters=10000,
                combine_text_under_n_chars=2000,
                new_after_n_chars=6000,
                languages=["eng", "ind"]
            )
            
            images_base64 = []
            page_map = {}
            
            for idx, chunk in enumerate(chunks):
                if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'page_number'):
                    page_map[idx] = chunk.metadata.page_number
                
                if "CompositeElement" in str(type(chunk)):
                    chunk_els = chunk.metadata.orig_elements
                    for el in chunk_els:
                        if "Image" in str(type(el)):
                            if hasattr(el. metadata, 'image_base64') and el.metadata.image_base64:
                                images_base64.append(el.metadata. image_base64)
            
            print(f"✅ Extracted {len(chunks)} chunks with unstructured")
            return chunks, images_base64, page_map
        
        except Exception as e:
            print(f"⚠️ Unstructured failed: {str(e)}")
            print("📄 Falling back to pdfplumber...")
    else:
        print("⚠️ Unstructured not available, using fallback methods")
    
    # Method 2: Fallback with pdfplumber (NO POPPLER NEEDED)
    if PDFPLUMBER_AVAILABLE:
        try:
            import pdfplumber
            
            elements = []
            images_base64 = []
            page_map = {}
            
            with pdfplumber.open(file_path) as pdf:
                print(f"📖 Processing {len(pdf.pages)} pages with pdfplumber...")
                
                for page_num, page in enumerate(pdf.pages, start=1):
                    # Extract text
                    text = page.extract_text()
                    if text and text.strip():
                        element = SimpleElement(text, page_num, "text")
                        elements.append(element)
                        page_map[len(elements) - 1] = page_num
                    
                    # Extract tables
                    tables = page.extract_tables()
                    if tables: 
                        for table in tables:
                            if table:  
                                # Convert table to text format
                                table_text = "\n".join([" | ".join([str(cell) if cell else "" for cell in row]) for row in table])
                                table_html = f"<table>{table_text}</table>"
                                table_element = SimpleElement(table_html, page_num, "table")
                                elements.append(table_element)
                                page_map[len(elements) - 1] = page_num
            
            print(f"✅ Extracted {len(elements)} elements with pdfplumber")
            return elements, images_base64, page_map
        
        except Exception as e:
            print(f"⚠️ pdfplumber failed: {str(e)}")
    
    # Method 3: Last fallback with PyPDF2
    if PYPDF2_AVAILABLE: 
        try:
            from PyPDF2 import PdfReader
            
            print("📄 Using PyPDF2 as last fallback...")
            elements = []
            images_base64 = []
            page_map = {}
            
            reader = PdfReader(file_path)
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    element = SimpleElement(text, page_num, "text")
                    elements.append(element)
                    page_map[len(elements) - 1] = page_num
            
            print(f"✅ Extracted {len(elements)} elements with PyPDF2")
            return elements, images_base64, page_map
        
        except Exception as e:
            print(f"❌ PyPDF2 failed: {str(e)}")
    
    # If all methods fail
    print(f"❌ All PDF extraction methods failed for {file_path}")
    return [], [], {}

def extract_elements_from_docx(file_path: str) -> Tuple[List, List[str], Dict]:
    """
    Extract elements from DOCX with fallback support
    Returns: (elements, image_base64_list, page_map)
    """
    print(f"🔍 Extracting from DOCX: {file_path}")
    
    # Method 1: Unstructured
    if UNSTRUCTURED_AVAILABLE:
        try:
            from unstructured.partition. docx import partition_docx
            
            print("📄 Trying unstructured library...")
            chunks = partition_docx(
                filename=file_path,
                chunking_strategy="by_title",
                max_characters=10000,
                combine_text_under_n_chars=2000,
                new_after_n_chars=6000,
            )
            
            images_base64 = []
            page_map = {}
            
            for idx, chunk in enumerate(chunks):
                if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'page_number'):
                    page_map[idx] = chunk.metadata. page_number
            
            print(f"✅ Extracted {len(chunks)} chunks with unstructured")
            return chunks, images_base64, page_map
        
        except Exception as e:
            print(f"⚠️ Unstructured failed: {str(e)}")
    else:
        print("⚠️ Unstructured not available, using fallback methods")
    
    # Method 2: Fallback with python-docx
    if PYTHON_DOCX_AVAILABLE: 
        try:
            from docx import Document
            
            print("📄 Using python-docx fallback...")
            doc = Document(file_path)
            elements = []
            page_map = {}
            
            for para_idx, para in enumerate(doc.paragraphs):
                if para.text. strip():
                    element = SimpleElement(para.text, None, "text")
                    elements.append(element)
            
            # Extract tables
            for table_idx, table in enumerate(doc. tables):
                table_text = ""
                for row in table.rows:
                    row_text = " | ".join([cell.text for cell in row. cells])
                    table_text += row_text + "\n"
                
                if table_text.strip():
                    table_element = SimpleElement(f"<table>{table_text}</table>", None, "table")
                    elements.append(table_element)
            
            print(f"✅ Extracted {len(elements)} elements with python-docx")
            return elements, [], page_map
        
        except Exception as e:
            print(f"❌ python-docx failed: {str(e)}")
    
    print(f"❌ All DOCX extraction methods failed for {file_path}")
    return [], [], {}

def summarize_image_with_groq(image_base64: str, groq_client: Groq) -> str:
    """
    Summarize image using Groq's Llama 4 Scout Vision model
    Model: meta-llama/llama-4-scout-17b-16e-instruct
    """
    try:
        prompt = """Describe this image in detail. 
Focus on the key visual elements, text content, charts, diagrams, or any important information.
Be specific and concise."""
        
        # Create message with image
        response = groq_client.chat. completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            temperature=1,
            max_tokens=500,
            top_p=1,
            stream=False,
            stop=None
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error summarizing image with Groq: {str(e)}")
        return "Image content could not be processed."

def summarize_text_with_groq(text: str, groq_client: Groq, max_retries: int = 3) -> str:
    """
    Summarize text using Groq with auto-retry for connection errors.
    """
    # If text is too short, don't waste an API call
    if len(text.strip()) < 50:
        return text

    for attempt in range(max_retries):
        try:
            prompt = f"""You are an assistant tasked with summarizing text content.
Give a concise summary of the following text. 
Respond only with the summary, no additional comment. 

Text: {text}
"""
            response = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=300
            )
            
            return response.choices[0].message.content

        except Exception as e:
            error_msg = str(e)
            print(f"⚠️ Attempt {attempt + 1}/{max_retries} failed: {error_msg}")
            
            # If it's a connection error or rate limit, wait and try again
            if "Connection error" in error_msg or "429" in error_msg:
                # Exponential backoff: Wait 2s, then 4s, then 8s...
                sleep_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(sleep_time)
            else:
                # If it's a logic error (e.g., bad request), stop trying
                break

    print("❌ Failed to summarize after retries. Using fallback.")
    return text[:500]  # Fallback

def summarize_table_with_groq(table_html: str, groq_client:  Groq) -> str:
    """Summarize table using Groq"""
    try:
        prompt = f"""You are an assistant tasked with summarizing tables.
Give a concise summary of the following table. 
Respond only with the summary, no additional comment.  

Table: {table_html}
"""
        
        response = groq_client.chat. completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=300
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error summarizing table: {str(e)}")
        return table_html[:500]


# ========== Advanced Chunking Strategies ==========
def chunk_text(
    text: str,
    strategy: str = "recursive",
    max_tokens: int = 1000,
    overlap_tokens: int = 100,
) -> List[str]:
    """
    Advanced chunking dengan berbagai strategi
    """
    if not text:
        return []

    if strategy == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n", ". ", " ", ""],
            chunk_size=max_tokens,
            chunk_overlap=overlap_tokens,
            length_function=len,
        )
        return text_splitter.split_text(text)

    elif strategy == "semantic":
        sentences = text.split(". ")
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk + sentence) < max_tokens:
                current_chunk += sentence + ". "
            else:
                if current_chunk: 
                    chunks.append(current_chunk. strip())
                current_chunk = sentence + ". "

        if current_chunk: 
            chunks.append(current_chunk.strip())
        return chunks

    elif strategy == "paragraph":
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for para in paragraphs:
            if len(current_chunk + para) < max_tokens:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(current_chunk. strip())
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk. strip())
        return chunks

    else:  # simple
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words: 
            word_length = len(word) + 1
            if current_length + word_length > max_tokens and current_chunk:
                chunks. append(" ".join(current_chunk))
                overlap_words = (
                    current_chunk[-overlap_tokens // 10:]
                    if len(current_chunk) > overlap_tokens // 10
                    else []
                )
                current_chunk = overlap_words + [word]
                current_length = sum(len(w) + 1 for w in current_chunk)
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk: 
            chunks.append(" ".join(current_chunk))

        return chunks


# ========== Document Metadata Helpers ==========
def extract_keywords_simple(text: str, top_n: int = 10) -> List[str]:
    """Extract simple keywords from text based on word frequency"""
    words = text. lower().split()
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
    filtered_words = [w for w in words if w not in stopwords and len(w) > 3]
    
    word_counts = Counter(filtered_words)
    return [word for word, count in word_counts.most_common(top_n)]


def generate_document_summary(text: str, max_length: int = 200) -> str:
    """Generate simple document summary from first sentences"""
    sentences = text.split('.  ')
    summary = ''
    for sentence in sentences[: 3]:
        if len(summary + sentence) < max_length:
            summary += sentence + '. '
        else:
            break
    return summary. strip() or text[: max_length] + "..."


# ========== LangChain Embeddings Wrapper ==========
class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper untuk SentenceTransformer agar kompatibel dengan LangChain"""
    
    def __init__(self, model:  SentenceTransformer):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed list of documents"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """Embed single query"""
        embedding = self.model.encode([text], convert_to_numpy=True)
        return embedding[0].tolist()


# ========== Vector Store Operations ==========
def build_vector_store_with_metadata(
    documents: List[Dict[str, Any]],
    embed_model: SentenceTransformer,
    show_progress: bool = True
) -> Tuple[FAISS, List[DocChunk]]:
    """
    Build LangChain FAISS vector store with proper metadata
    """
    if not documents:
        raise ValueError("No documents provided to build vector store")
    
    embeddings = SentenceTransformerEmbeddings(embed_model)
    sample_vec = embed_model.encode(["test"], convert_to_numpy=True)
    dimension = sample_vec.shape[1]
    
    index = faiss.IndexFlatIP(dimension)
    
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    
    texts = []
    metadatas = []
    doc_chunks = []
    
    for doc in tqdm(documents, desc="Building vector store", disable=not show_progress):
        texts.append(doc["text"])
        
        meta_dict = {
            "doc_id": doc. get("doc_id", "unknown"),
            "chunk_id": doc.get("chunk_id", 0),
            "category": doc.get("category", "unknown"),
            "keywords": doc.get("keywords", ""),
            "content_type": doc.get("content_type", "text"),
        }
        metadatas.append(meta_dict)
        
        doc_chunk = DocChunk(
            doc_id=doc. get("doc_id", "unknown"),
            chunk_id=doc.get("chunk_id", 0),
            text=doc["text"],
            meta=meta_dict,
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            chunk_size=doc.get("chunk_size", 1000),
            chunk_overlap=doc.get("chunk_overlap", 100),
            metadata=doc. get("metadata", None),
            content_type=doc.get("content_type", "text")
        )
        doc_chunks.append(doc_chunk)
    
    vector_store.add_texts(texts, metadatas=metadatas)
    
    return vector_store, doc_chunks


def save_vector_store_with_metadata(
    vector_store: FAISS, 
    chunks: List[DocChunk], 
    save_path: str, 
    index_name: str
):
    """Save vector store beserta metadata lengkap"""
    os.makedirs(save_path, exist_ok=True)
    
    full_path = os.path.join(save_path, index_name)
    vector_store.save_local(full_path)
    
    metadata_path = os.path.join(save_path, f"{index_name}_metadata.json")
    
    chunks_data = []
    for chunk in chunks:
        chunk_data = {
            "doc_id": chunk.doc_id,
            "chunk_id":  chunk.chunk_id,
            "text": chunk.text[: 500],
            "embedding_model": chunk.embedding_model,
            "chunk_size": chunk.chunk_size,
            "chunk_overlap": chunk.chunk_overlap,
            "meta": chunk.meta,
            "content_type": chunk.content_type,
        }
        
        if chunk.metadata:
            chunk_data["metadata"] = {
                "filename": chunk.metadata.filename,
                "file_size": chunk.metadata.file_size,
                "creation_date": chunk.metadata.creation_date. isoformat(),
                "page_count": chunk.metadata.page_count,
                "keywords": chunk.metadata.keywords,
                "summary": chunk.metadata.summary,
                "document_type": chunk.metadata. document_type,
            }
        
        chunks_data.append(chunk_data)
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Vector store and metadata saved to:  {full_path}")


def load_vector_store_with_metadata(
    save_path:  str, 
    index_name: str, 
    embed_model: SentenceTransformer
) -> Tuple[FAISS, List[Dict]]:
    """Load vector store beserta metadata"""
    full_path = os. path.join(save_path, index_name)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Vector store not found at: {full_path}")
    
    embeddings = SentenceTransformerEmbeddings(embed_model)
    
    vector_store = FAISS.load_local(
        full_path, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    metadata_path = os. path.join(save_path, f"{index_name}_metadata. json")
    
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = []
    
    print(f"✅ Vector store and metadata loaded from: {full_path}")
    return vector_store, metadata


# ========== Reranking Functions ==========
def rerank_results(
    query:  str,
    results: List[Dict[str, Any]],
    embed_model: SentenceTransformer,
    rerank_method: str = "hybrid"
) -> List[Dict[str, Any]]:
    """Rerank search results"""
    if not results:
        return results
    
    query_words = set(query.lower().split())
    scored_results = []
    
    for result in results:
        text = result["text"]
        original_score = result["score"]
        
        doc_words = set(text.lower().split())
        keyword_overlap = len(query_words.intersection(doc_words)) / max(1, len(query_words))
        keyword_score = keyword_overlap
        
        if rerank_method == "hybrid":
            combined_score = 0.7 * original_score + 0.3 * keyword_score
        elif rerank_method == "keyword":
            combined_score = keyword_score
        else:
            combined_score = original_score
        
        scored_results.append({
            **result,
            "original_score": float(original_score),
            "keyword_score": float(keyword_score),
            "combined_score": float(combined_score)
        })
    
    scored_results.sort(key=lambda x: x["combined_score"], reverse=True)
    
    return scored_results


def search_vector_store_with_reranking(
    vector_store:  FAISS,
    query: str,
    embed_model: SentenceTransformer,
    k: int = 5,
    rerank_top_k: int = 20,
    rerank_method:  str = "hybrid",
    score_threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """Search vector store dengan reranking"""
    initial_results = vector_store.similarity_search_with_score(query, k=rerank_top_k)
    
    results = []
    for doc, score in initial_results:
        if score >= score_threshold:
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })
    
    reranked_results = rerank_results(query, results, embed_model, rerank_method)
    
    return reranked_results[: k]

def detect_language(text: str) -> str:
    """
    Simple language detection based on character patterns
    Returns:  'indonesian' or 'english'
    """
    # Indonesian-specific words
    indonesian_words = ['yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'pada', 'adalah', 'dalam', 
                        'dengan', 'ini', 'itu', 'atau', 'juga', 'akan', 'telah', 'ada', 
                        'apa', 'siapa', 'bagaimana', 'mengapa', 'kapan', 'dimana']
    
    # English-specific words
    english_words = ['the', 'is', 'are', 'was', 'were', 'what', 'who', 'how', 'why', 'when', 
                     'where', 'which', 'this', 'that', 'these', 'those', 'have', 'has', 'had']
    
    text_lower = text.lower()
    words = text_lower.split()
    
    indonesian_count = sum(1 for word in words if word in indonesian_words)
    english_count = sum(1 for word in words if word in english_words)
    
    # Return language with higher count, default to indonesian
    if english_count > indonesian_count: 
        return 'english'
    else: 
        return 'indonesian'


def create_enhanced_context(chunks: List[DocChunk]) -> str:
    """Create enhanced context from document chunks with page references"""
    parts = []
    for ch in chunks:
        content_type = ch.meta.get('content_type', 'text')
        page_info = f", Halaman {ch.page_number}" if ch.page_number else ""
        header = f"[{ch. doc_id}{page_info} | {ch.meta.get('category', 'unknown')} | Tipe: {content_type}]\n"
        parts.append(header + ch.text. strip())
    return "\n\n---\n\n".join(parts)


def answer_with_rag(query: str, retrieved:  List[DocChunk], chat_model: str) -> str:
    """Generate an answer using the retrieved document snippets and GROQ chat model with auto language detection"""
    client = get_groq_client()
    
    # Auto-detect language from query
    language = detect_language(query)
    
    # System prompt based on detected language
    if language == "indonesian":
        system_prompt = """Anda adalah asisten AI untuk analisis dokumen. 
Tugas Anda adalah menjawab pertanyaan pengguna berdasarkan kutipan dokumen yang disediakan.
Kutipan dapat mencakup konten teks, ringkasan tabel, dan deskripsi gambar.  
Jawab HANYA berdasarkan konteks yang diberikan.  Berikan jawaban yang ringkas dan jangan mengarang fakta.
Jika jawabannya tidak ada dalam konteks, katakan "Informasi tidak ditemukan dalam dokumen."

PENTING: Setelah menjawab, SELALU cantumkan referensi dengan format:  

Referensi:
- [nama_file], halaman [nomor_halaman]
- [nama_file], halaman [nomor_halaman]

Jika nomor halaman tidak tersedia, cukup tulis:  
- [nama_file]"""
    else:
        system_prompt = """You are an AI assistant for document analysis. 
Your job is to answer the user's question strictly based on the provided document snippets. 
The snippets may include text content, table summaries, and image descriptions. 
Answer only based on the provided context. Be concise and do not invent facts.  
If the answer does not exist in the context, say "Not found in the document."

IMPORTANT: After answering, ALWAYS include references in this format:

References:
- [filename], page [page_number]
- [filename], page [page_number]

If page number is not available, just write: 
- [filename]"""
    
    # Create enhanced context
    context = create_enhanced_context(retrieved)
    
    # User prompt based on detected language
    if language == "indonesian":
        user_prompt = f"""Pertanyaan: {query}

Konteks dokumen yang relevan:  
{context}

Tolong jawab pertanyaan berdasarkan konteks di atas. Jika memungkinkan, sebutkan dokumen mana yang menjadi rujukan jawaban Anda."""
    else:
        user_prompt = f"""Question: {query}

Relevant document snippets:
{context}

Please answer the question based on the context above. If possible, mention which documents were used as references."""
    
    try:
        resp = client.chat.completions. create(
            model=chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content":  user_prompt},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message. content
    except Exception as e: 
        if language == "indonesian":
            return f"Kesalahan saat menghasilkan jawaban: {str(e)}"
        else:
            return f"Error generating answer: {str(e)}"