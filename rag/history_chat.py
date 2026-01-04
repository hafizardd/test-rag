# chat_history.py
import os
from pathlib import Path
from typing import List, Dict, Optional
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    ForeignKey,
    DateTime,
)
from sqlalchemy.orm import (
    sessionmaker,
    relationship,
    declarative_base,
)
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime

# GROQ client from rag_core
from rag. rag_core import get_groq_client, detect_language
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ". env"

# Load . env file explicitly
load_dotenv(dotenv_path=ENV_PATH)

# ===============================
# Database Configuration
# ===============================
def get_database_url() -> str:
    """Build and validate database URL from environment variables"""
    
    database_url = os.getenv("DATABASE_URL")
    if database_url: 
        return database_url
    
    host = os.getenv("SUPABASE_HOST")
    port = os.getenv("SUPABASE_PORT", "5432")
    db_name = os.getenv("SUPABASE_DB", "postgres")
    user = os.getenv("SUPABASE_USER", "postgres")
    password = os.getenv("SUPABASE_PASSWORD")
    
    if not host:
        raise ValueError("Missing SUPABASE_HOST.  Please set it in your .env file.")
    if not password:
        raise ValueError("Missing SUPABASE_PASSWORD. Please set it in your .env file.")
    
    return f"postgresql://{user}:{password}@{host}:{port}/{db_name}"


DATABASE_URL = get_database_url()
print(f"Connecting to:  {DATABASE_URL. split('@')[1] if '@' in DATABASE_URL else 'Invalid URL'}")

Base = declarative_base()

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
)
SessionLocal = sessionmaker(bind=engine)


# ===============================
# ORM Models
# ===============================
class ChatSession(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    session_id = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.id",
    )


class ChatMessage(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("sessions.id"), nullable=False)
    role = Column(String(50), nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    session = relationship("ChatSession", back_populates="messages")


Base.metadata.create_all(engine)


# ===============================
# DB Helpers
# ===============================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _get_or_create_session(db, session_id: str) -> ChatSession:
    session = (
        db.query(ChatSession)
        .filter(ChatSession.session_id == session_id)
        .first()
    )
    if not session: 
        session = ChatSession(session_id=session_id)
        db.add(session)
        db.commit()
        db.refresh(session)
    return session


# ===============================
# Public API
# ===============================
def save_message(session_id: str, role: str, content: str) -> None:
    """Persist one chat message"""
    db = next(get_db())
    try:
        session = _get_or_create_session(db, session_id)
        msg = ChatMessage(
            session_id=session. id,
            role=role,
            content=content,
        )
        db.add(msg)
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Database error: {e}")
    finally:
        db.close()


def load_chat_history(session_id: str, limit: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Load chat history in OpenAI/GROQ-compatible format. 
    
    Args:
        session_id: The session identifier
        limit: Maximum number of messages to return (most recent). None for all.
    
    Returns:
        List of message dicts with 'role' and 'content' keys
    """
    db = next(get_db())
    history:  List[Dict[str, str]] = []

    try:
        session = (
            db.query(ChatSession)
            .filter(ChatSession.session_id == session_id)
            .first()
        )
        if session:
            messages = session.messages
            
            # Apply limit if specified (get most recent messages)
            if limit and len(messages) > limit:
                messages = messages[-limit:]
            
            for msg in messages: 
                history.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                    }
                )
    finally:
        db.close()

    return history


def clear_chat_history(session_id: str) -> bool:
    """Clear all chat history for a session"""
    db = next(get_db())
    try:
        session = (
            db.query(ChatSession)
            .filter(ChatSession.session_id == session_id)
            .first()
        )
        if session:
            db.delete(session)
            db.commit()
            return True
        return False
    except SQLAlchemyError as e: 
        db.rollback()
        print(f"Database error:  {e}")
        return False
    finally:
        db.close()


def get_all_sessions() -> List[Dict[str, any]]:
    """Get all chat sessions with metadata"""
    db = next(get_db())
    sessions = []
    
    try:
        all_sessions = db.query(ChatSession).order_by(ChatSession.updated_at.desc()).all()
        for session in all_sessions:
            sessions.append({
                "session_id": session.session_id,
                "created_at":  session.created_at,
                "updated_at": session.updated_at,
                "message_count": len(session.messages),
            })
    finally:
        db.close()
    
    return sessions


# ===============================
# History-Aware Question Rewriting
# ===============================
def reformulate_question(
    chat_history: List[Dict[str, str]],
    user_query: str,
) -> str:
    """
    Reformulate follow-up questions into standalone questions.
    This helps the retrieval system find relevant documents even when
    the user refers to previous conversation context.
    """
    # If no history, return original query
    if not chat_history: 
        print(f"📝 No history, using original query: '{user_query}'")
        return user_query
    
    # Limit history to last 10 messages for context (5 exchanges)
    recent_history = chat_history[-10: ] if len(chat_history) > 10 else chat_history
    
    client = get_groq_client()
    language = detect_language(user_query)

    if language == "indonesian": 
        system_prompt = """Anda adalah asisten AI yang membantu mereformulasi pertanyaan. 

Tugas Anda: 
1. Lihat riwayat percakapan dan pertanyaan terakhir pengguna
2. Jika pertanyaan mengacu pada konteks sebelumnya (misalnya "itu", "tersebut", "hal itu", dll), ubah menjadi pertanyaan lengkap dan mandiri
3. Jika pertanyaan sudah jelas dan mandiri, kembalikan pertanyaan asli tanpa perubahan

Contoh: 
- Riwayat: "Apa itu Holland Schema?" -> "Holland Schema adalah teori karir..."
- Pertanyaan: "Siapa yang menciptakannya?"
- Reformulasi: "Siapa yang menciptakan Holland Schema?"

PENTING: 
- Hanya kembalikan pertanyaan yang sudah direformulasi
- JANGAN menjawab pertanyaan
- JANGAN menambahkan penjelasan
- Pertahankan bahasa asli pertanyaan"""
    else:
        system_prompt = """You are an AI assistant that helps reformulate questions.

Your task:
1. Look at the chat history and the user's latest question
2. If the question refers to previous context (e.g., "it", "that", "this", etc.), rewrite it as a complete standalone question
3. If the question is already clear and standalone, return it unchanged

Example:
- History: "What is Holland Schema?" -> "Holland Schema is a career theory..."
- Question: "Who created it?"
- Reformulated: "Who created the Holland Schema?"

IMPORTANT: 
- Only return the reformulated question
- DO NOT answer the question
- DO NOT add explanations
- Keep the original language of the question"""

    messages = [{"role": "system", "content":  system_prompt}]
    messages.extend(recent_history)
    messages.append({"role": "user", "content": f"Reformulasi pertanyaan ini: {user_query}" if language == "indonesian" else f"Reformulate this question: {user_query}"})

    try:
        response = client.chat.completions. create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=0.1,  # Low temperature for more deterministic output
            max_tokens=200,
        )
        reformulated = response.choices[0].message.content.strip()
        
        # Clean up the response - remove quotes if present
        reformulated = reformulated.strip('"\'')
        
        # If the reformulated query is too different or seems like an answer, use original
        if len(reformulated) > len(user_query) * 3 or ":" in reformulated:
            print(f"📝 Reformulation too long, using original:  '{user_query}'")
            return user_query
        
        print(f"📝 Original:  '{user_query}' -> Reformulated: '{reformulated}'")
        return reformulated
        
    except Exception as e:
        print(f"⚠️ Reformulation failed: {e}")
        return user_query


# ===============================
# Main Entry for RAG
# ===============================
def process_user_query(
    session_id: str,
    user_query: str,
) -> str:
    """
    Process user query with chat history context.
    
    - Load history
    - Reformulate query (history-aware)
    - Save user message
    - Return standalone query for retrieval
    """
    # Load recent history (last 10 messages)
    history = load_chat_history(session_id, limit=10)
    
    print(f"📚 Loaded {len(history)} messages from history")
    
    # Reformulate if there's history
    standalone_query = reformulate_question(history, user_query)
    
    # Save the original user message
    save_message(session_id, "user", user_query)
    
    return standalone_query


def save_assistant_answer(
    session_id: str,
    answer: str,
) -> None:
    """Save assistant answer after RAG generation"""
    save_message(session_id, "assistant", answer)