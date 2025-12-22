# chat_history.py
import os
from typing import List, Dict
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import (
    sessionmaker,
    relationship,
    declarative_base,
)
from sqlalchemy.exc import SQLAlchemyError

# GROQ client from rag_core 
from rag.rag_core import get_groq_client, detect_language

# ===============================
# Database Configuration
# ===============================
DATABASE_URL = "sqlite:///chat_history.db"
Base = declarative_base()

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)


# ===============================
# ORM Models
# ===============================
class ChatSession(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, nullable=False)
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
    role = Column(String, nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)

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
            session_id=session.id,
            role=role,
            content=content,
        )
        db.add(msg)
        db.commit()
    except SQLAlchemyError:
        db.rollback()
    finally:
        db.close()


def load_chat_history(session_id: str) -> List[Dict[str, str]]:
    """
    Load chat history in OpenAI/GROQ-compatible format:
    [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
    """
    db = next(get_db())
    history: List[Dict[str, str]] = []

    try:
        session = (
            db.query(ChatSession)
            .filter(ChatSession.session_id == session_id)
            .first()
        )
        if session:
            for msg in session.messages:
                history.append(
                    {
                        "role": msg.role,
                        "content": msg.content,
                    }
                )
    finally:
        db.close()

    return history


# ===============================
# History-Aware Question Rewriting
# ===============================
def reformulate_question(
    chat_history: List[Dict[str, str]],
    user_query: str,
) -> str:
    """
    Reformulate follow-up questions into standalone questions
    (equivalent to LangChain create_history_aware_retriever)
    """
    if not chat_history:
        return user_query

    client = get_groq_client()
    language = detect_language(user_query)

    if language == "indonesian":
        system_prompt = (
            "Anda adalah asisten AI.\n"
            "Berdasarkan riwayat percakapan dan pertanyaan terakhir pengguna, "
            "ubah pertanyaan tersebut menjadi pertanyaan mandiri "
            "yang dapat dipahami tanpa konteks sebelumnya.\n"
            "JANGAN menjawab pertanyaannya."
        )
    else:
        system_prompt = (
            "You are an AI assistant.\n"
            "Given the chat history and the latest user question, "
            "rewrite the question so it can be understood on its own.\n"
            "DO NOT answer the question."
        )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_query})

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            temperature=2,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        # Fallback: return original query
        return user_query


# ===============================
# Main Entry for RAG
# ===============================
def process_user_query(
    session_id: str,
    user_query: str,
) -> str:
    """
    - Load history
    - Reformulate query (history-aware)
    - Save user message
    - Return standalone query for FAISS retrieval
    """
    history = load_chat_history(session_id)
    standalone_query = reformulate_question(history, user_query)

    save_message(session_id, "user", user_query)

    return standalone_query


def save_assistant_answer(
    session_id: str,
    answer: str,
) -> None:
    """Save assistant answer after RAG generation"""
    save_message(session_id, "assistant", answer)
