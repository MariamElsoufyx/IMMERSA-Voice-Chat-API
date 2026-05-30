import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.database import Base

EMBEDDING_DIM = 384  # BAAI/bge-small-en-v1.5 local model dimensions


class FAQ(Base):
    __tablename__ = "frequently_asked_questions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    character_id: Mapped[str] = mapped_column(String(10), nullable=False, index=True)  # s1 / s2 / p1 (always lowercase)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    audio_url: Mapped[str | None] = mapped_column(Text, nullable=True)   # Supabase Storage URL
    tag: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="en")  # 'en' or 'ar'
    embedding: Mapped[list[float] | None] = mapped_column(Vector(384), nullable=True)
    tag = mapped_column(String(50), nullable=True)  # optional tag for categorization
    emotion: Mapped[str | None] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<FAQ id={self.id} character={self.character_id} question={self.question[:40]!r}>"


class HistoryChunk(Base):
    """A chunk of source college-history text used for RAG retrieval.

    Unlike FAQ (pre-written Q&A pairs returned verbatim), these are passages of
    source material. On an FAQ miss the top-K most similar chunks are retrieved
    and injected into the LLM prompt as grounding context.
    """
    __tablename__ = "RAG_data"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)  # the chunk — embedded + injected into the prompt
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)
    # NULL = global history shared by every character; otherwise scoped to one character's era/department
    character_id: Mapped[str | None] = mapped_column(String(10), nullable=True, index=True)
    source_doc: Mapped[str | None] = mapped_column(Text, nullable=True)   # origin file/document, for citations + re-indexing
    chunk_index: Mapped[int | None] = mapped_column(Integer, nullable=True)  # position within source_doc
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="en")  # 'en' or 'ar'
    tag: Mapped[str | None] = mapped_column(String(50), nullable=True)   # topic, e.g. 'founding', 'faculty'
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    def __repr__(self) -> str:
        return f"<HistoryChunk id={self.id} character={self.character_id} content={self.content[:40]!r}>"


class PastQuestion(Base):
    __tablename__ = "past_questions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    character_id: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    audio_url: Mapped[str | None] = mapped_column(Text, nullable=True)   # response audio (TTS or FAQ cached)
    question_audio_url: Mapped[str | None] = mapped_column(Text, nullable=True)  # original client-recorded question audio
    source: Mapped[str] = mapped_column(String(10), nullable=False, default="llm")  # 'faq' or 'llm'
    faq_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    emotion: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Timing in seconds (nullable — may be absent if stage was skipped)
    preprocess_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    stt_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    faq_lookup_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    llm_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    content_filter_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    content_filter_pass: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    content_filter_flagged: Mapped[str | None] = mapped_column(Text, nullable=True)
    verifier_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    verifier_pass: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    verifier_historical_accuracy: Mapped[str | None] = mapped_column(Text, nullable=True)
    verifier_appropriateness: Mapped[str | None] = mapped_column(Text, nullable=True)
    verifier_modern_references: Mapped[str | None] = mapped_column(Text, nullable=True)
    verifier_in_character: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Replacement reply produced by the verifier when it rejects the original answer.
    # NULL when the verifier passes (or fails without producing a correction).
    verifier_corrected_answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    verifier_corrected_emotion: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # Tier 1 — regex anachronism (future years / modern terms / URLs / emails)
    anachronism_pass: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    anachronism_reasons: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Tier 2 — OpenAI Moderation API on the question (run parallel to FAQ lookup)
    moderation_q_pass: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    moderation_q_categories: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Tier 2 — OpenAI Moderation API on the answer (run parallel to TTS)
    moderation_a_pass: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    moderation_a_categories: Mapped[str | None] = mapped_column(Text, nullable=True)
    tts_first_chunk_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    tts_total_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    time_to_first_audio_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    total_s: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )

    def __repr__(self) -> str:
        return f"<PastQuestion id={self.id} character={self.character_id} question={self.question[:40]!r}>"
