import os
import json
import math
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np
import faiss

from groq import Groq

from dotenv import load_dotenv
load_dotenv()

# ---------- Config ----------

def get_groq_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set. Provide it via env or a .env file.")
    return Groq(api_key=api_key)

# ---------- Chunking ----------

def _approx_token_len(text: str) -> int:
    # cheap token proxy: ~4 chars per token
    return max(1, math.ceil(len(text) / 4))

def chunk_text(text: str, max_tokens: int = 400, overlap_tokens: int = 60) -> List[str]:
    if not text:
        return []
    words = text.split()
    chunks, cur, cur_len = [], [], 0
    for w in words:
        w_len = _approx_token_len(w + " ")
        if cur_len + w_len > max_tokens and cur:
            chunks.append(" ".join(cur))
            # overlap
            while cur and cur_len > overlap_tokens:
                popped = cur.pop(0)
                cur_len -= _approx_token_len(popped + " ")
        cur.append(w)
        cur_len += w_len
    if cur:
        chunks.append(" ".join(cur))
    return chunks

# ---------- Embeddings / FAISS ----------

def embed_texts(texts: List[str], embed_model: str, batch_size: int = 96, show_progress: bool = True) -> np.ndarray:
    """Embed a list of texts using the selected embedding model."""
    client = get_groq_client()
    vecs = []
    for i in tqdm(range(0, len(texts), batch_size), disable=not show_progress, desc="Embedding"):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=embed_model, input=batch)
        for d in resp.data:
            vecs.append(d.embedding)
        time.sleep(0.01)  # gentle pacing
    return np.array(vecs, dtype="float32")

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

def search_index(index: faiss.IndexFlatIP, query_vec: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    faiss.normalize_L2(query_vec)
    D, I = index.search(query_vec, k)
    return D, I

# ---------- RAG Pipeline ----------

@dataclass
class DocChunk:
    doc_id: str
    chunk_id: int
    text: str
    meta: Dict

def make_context(chunks: List[DocChunk]) -> str:
    # Short context with minimal boilerplate to stay under token budgets
    parts = []
    for ch in chunks:
        header = f"[{ch.doc_id} | {ch.meta.get('category', 'unknown')}]\n"
        parts.append(header + ch.text.strip())
    return "\n\n---\n\n".join(parts)

def answer_with_rag(query: str, retrieved: List[DocChunk], chat_model: str) -> str:
    """Generate an answer using the retrieved resume snippets and the chosen chat model."""
    client = get_groq_client()
    system = (
        "You are an HR assistant chatbot. Answer the user's question using ONLY the provided resume snippets. "
        "If the answer can't be found, say so briefly. Be concise. Where helpful, extract names, skills, years of experience."
    )
    context = make_context(retrieved)
    user = f"Question: {query}\n\nRelevant resume snippets:\n{context}"
    resp = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content
