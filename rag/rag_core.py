import os
import json
import faiss
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm
from collections import Counter

import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community. docstore.in_memory import InMemoryDocstore
from langchain_core.embeddings import Embeddings
from dotenv import load_dotenv

load_dotenv()

# ========== Configuration ==========
def get_groq_client() -> Groq:
    """Get GROQ API client"""
    api_key = os.environ.get("GROQ_API_KEY", ""). strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set.  Provide it via env or a .env file.")
    return Groq(api_key=api_key)


# ========== Document Metadata Dataclass ==========
@dataclass
class DocumentMetadata:
    """Metadata untuk dokumen"""
    filename: str
    file_size: int
    creation_date: datetime
    page_count: int = 0
    keywords: Optional[List[str]] = None
    summary: str = ""
    document_type: str = "general"
    doc_id: str = ""


@dataclass
class DocChunk:
    """Document chunk with metadata"""
    doc_id: str
    chunk_id: int
    text: str
    meta: Dict
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    metadata: Optional[DocumentMetadata] = None


# ========== Advanced Chunking Strategies ==========
def chunk_text_advanced(
    text: str,
    strategy: str = "recursive",
    max_tokens: int = 1000,
    overlap_tokens: int = 100,
) -> List[str]:
    """
    Advanced chunking dengan berbagai strategi
    
    Args:
        text: Text to chunk
        strategy: 'recursive', 'semantic', 'paragraph', or 'simple'
        max_tokens: Maximum chunk size in characters
        overlap_tokens: Overlap size in characters
    
    Returns:
        List of text chunks
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
                    chunks.append(current_chunk.strip())
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk.strip())
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
                    current_chunk[-overlap_tokens // 10 :]
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
    words = text.lower().split()
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
    filtered_words = [w for w in words if w not in stopwords and len(w) > 3]
    
    word_counts = Counter(filtered_words)
    return [word for word, count in word_counts.most_common(top_n)]


def generate_document_summary(text: str, max_length: int = 200) -> str:
    """Generate simple document summary from first sentences"""
    sentences = text.split('.  ')
    summary = ''
    for sentence in sentences[:3]:
        if len(summary + sentence) < max_length:
            summary += sentence + '. '
        else:
            break
    return summary. strip() or text[:max_length] + "..."


# ========== LangChain Embeddings Wrapper ==========
class SentenceTransformerEmbeddings(Embeddings):
    """Wrapper untuk SentenceTransformer agar kompatibel dengan LangChain"""
    
    def __init__(self, model: SentenceTransformer):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed list of documents"""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings. tolist()
    
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
    
    Args:
        documents: List of dicts with 'text', 'doc_id', 'chunk_id', 'category', 'keywords', 'metadata'
        embed_model: SentenceTransformer model
        show_progress: Show progress bar
    
    Returns:
        (FAISS vector store, List of DocChunk objects)
    """
    if not documents:
        raise ValueError("No documents provided to build vector store")
    
    # Wrap SentenceTransformer for LangChain compatibility
    embeddings = SentenceTransformerEmbeddings(embed_model)
    
    # Get dimension from first embedding
    sample_vec = embed_model.encode(["test"], convert_to_numpy=True)
    dimension = sample_vec. shape[1]
    
    # Create FAISS index
    index = faiss.IndexFlatIP(dimension)
    
    # Create vector store
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    
    # Prepare texts, metadata, and DocChunk objects
    texts = []
    metadatas = []
    doc_chunks = []
    
    for doc in tqdm(documents, desc="Building vector store", disable=not show_progress):
        texts.append(doc["text"])
        
        # Prepare metadata for FAISS
        meta_dict = {
            "doc_id": doc. get("doc_id", "unknown"),
            "chunk_id": doc.get("chunk_id", 0),
            "category": doc.get("category", "unknown"),
            "keywords": doc. get("keywords", ""),
        }
        metadatas.append(meta_dict)
        
        # Create DocChunk object
        doc_chunk = DocChunk(
            doc_id=doc. get("doc_id", "unknown"),
            chunk_id=doc.get("chunk_id", 0),
            text=doc["text"],
            meta=meta_dict,
            embedding_model="sentence-transformers/all-mpnet-base-v2",
            chunk_size=doc.get("chunk_size", 1000),
            chunk_overlap=doc.get("chunk_overlap", 100),
            metadata=doc. get("metadata", None)
        )
        doc_chunks.append(doc_chunk)
    
    # Add documents to vector store
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
    
    # Save vector store (LangChain FAISS format)
    full_path = os.path.join(save_path, index_name)
    vector_store.save_local(full_path)
    
    # Save metadata chunks
    metadata_path = os. path.join(save_path, f"{index_name}_metadata. json")
    
    chunks_data = []
    for chunk in chunks:
        chunk_data = {
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "text": chunk.text[:500],  # Save first 500 chars for reference
            "embedding_model": chunk.embedding_model,
            "chunk_size": chunk.chunk_size,
            "chunk_overlap": chunk.chunk_overlap,
            "meta": chunk.meta,
        }
        
        # Add DocumentMetadata if available
        if chunk.metadata:
            chunk_data["metadata"] = {
                "filename": chunk.metadata.filename,
                "file_size": chunk.metadata.file_size,
                "creation_date": chunk.metadata.creation_date.isoformat(),
                "page_count": chunk.metadata. page_count,
                "keywords": chunk.metadata.keywords,
                "summary": chunk.metadata. summary,
                "document_type": chunk.metadata.document_type,
            }
        
        chunks_data.append(chunk_data)
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Vector store and metadata saved to: {full_path}")


def load_vector_store_with_metadata(
    save_path: str, 
    index_name: str, 
    embed_model: SentenceTransformer
) -> Tuple[FAISS, List[Dict]]:
    """Load vector store beserta metadata"""
    full_path = os.path.join(save_path, index_name)
    
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Vector store not found at: {full_path}")
    
    # Wrap embeddings
    embeddings = SentenceTransformerEmbeddings(embed_model)
    
    # Load vector store
    vector_store = FAISS.load_local(
        full_path, 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Load metadata
    metadata_path = os.path.join(save_path, f"{index_name}_metadata.json")
    
    if os.path. exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = []
    
    print(f"✅ Vector store and metadata loaded from: {full_path}")
    return vector_store, metadata


# ========== Reranking Functions ==========
def rerank_results(
    query: str,
    results: List[Dict[str, Any]],
    embed_model: SentenceTransformer,
    rerank_method: str = "hybrid"
) -> List[Dict[str, Any]]:
    """
    Rerank search results menggunakan berbagai metode
    
    Args:
        query: Search query
        results: List of search results with 'text', 'metadata', 'score'
        embed_model: SentenceTransformer model
        rerank_method: 'hybrid', 'keyword', or 'semantic'
    
    Returns:
        Reranked results
    """
    if not results:
        return results
    
    query_words = set(query.lower().split())
    scored_results = []
    
    for result in results:
        text = result["text"]
        original_score = result["score"]
        
        # Keyword overlap score
        doc_words = set(text.lower().split())
        keyword_overlap = len(query_words.intersection(doc_words)) / max(1, len(query_words))
        
        # BM25-like score (simplified)
        keyword_score = keyword_overlap
        
        # Combine scores based on method
        if rerank_method == "hybrid":
            # Hybrid: 70% semantic, 30% keyword
            combined_score = 0.7 * original_score + 0.3 * keyword_score
        elif rerank_method == "keyword":
            # Pure keyword matching
            combined_score = keyword_score
        else:  # semantic
            # Pure semantic similarity
            combined_score = original_score
        
        scored_results.append({
            **result,
            "original_score": float(original_score),
            "keyword_score": float(keyword_score),
            "combined_score": float(combined_score)
        })
    
    # Sort by combined score
    scored_results.sort(key=lambda x: x["combined_score"], reverse=True)
    
    return scored_results


def search_vector_store_with_reranking(
    vector_store: FAISS,
    query: str,
    embed_model: SentenceTransformer,
    k: int = 5,
    rerank_top_k: int = 20,
    rerank_method: str = "hybrid",
    score_threshold: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Search vector store dengan reranking
    
    Args:
        vector_store: FAISS vector store
        query: Search query
        embed_model: SentenceTransformer model
        k: Final number of results to return
        rerank_top_k: Number of results to retrieve before reranking (should be > k)
        rerank_method: 'hybrid', 'keyword', or 'semantic'
        score_threshold: Minimum score threshold
    
    Returns:
        List of reranked results
    """
    # Step 1: Retrieve more results than needed
    initial_results = vector_store.similarity_search_with_score(query, k=rerank_top_k)
    
    # Convert to dict format
    results = []
    for doc, score in initial_results:
        if score >= score_threshold:
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)
            })
    
    # Step 2: Rerank results
    reranked_results = rerank_results(query, results, embed_model, rerank_method)
    
    # Step 3: Return top-k after reranking
    return reranked_results[:k]


# ========== RAG Pipeline ==========
def make_context(chunks: List[DocChunk]) -> str:
    """Create context from document chunks"""
    parts = []
    for ch in chunks:
        header = f"[{ch. doc_id} | {ch.meta.get('category', 'unknown')}]\n"
        parts.append(header + ch.text. strip())
    return "\n\n---\n\n".join(parts)


def answer_with_rag(query: str, retrieved: List[DocChunk], chat_model: str) -> str:
    """
    Generate an answer using the retrieved document snippets and GROQ chat model
    
    Args:
        query: User question
        retrieved: List of retrieved document chunks
        chat_model: GROQ model name
    
    Returns:
        Generated answer
    """
    client = get_groq_client()
    
    system = """You are an AI assistant for document screening. 
Your job is to answer the user's question strictly based on the provided document snippets.  
Answer only based on the provided context. Be concise and do not invent facts.  
If the answer does not exist in the context, say "Not found in the document." """
    
    context = make_context(retrieved)
    user = f"Question: {query}\n\nRelevant document snippets:\n{context}"
    
    try:
        resp = client.chat.completions. create(
            model=chat_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message. content
    except Exception as e:
        return f"Error generating answer: {str(e)}"