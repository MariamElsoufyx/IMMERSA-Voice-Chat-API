import uuid
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from pgvector.sqlalchemy import Vector
from app.core.config import FAQ_SIMILARITY_THRESHOLD
from app.db.models import FAQ
from sqlalchemy import text



async def get_all_faqs(db: AsyncSession) -> list[FAQ]:
    result = await db.execute(select(FAQ).order_by(FAQ.created_at.desc()))
    return result.scalars().all()


async def get_faq_by_id(db: AsyncSession, faq_id: uuid.UUID) -> FAQ | None:
    result = await db.execute(select(FAQ).where(FAQ.id == faq_id))
    return result.scalar_one_or_none()


async def get_faqs_by_character(db: AsyncSession, character_id: str) -> list[FAQ]:
    result = await db.execute(
        select(FAQ).where(FAQ.character_id == character_id).order_by(FAQ.created_at.desc())
    )
    return result.scalars().all()


async def create_faq(db: AsyncSession, faq_data: dict) -> FAQ:
    faq_data = {**faq_data, "character_id": faq_data["character_id"].lower()}
    faq = FAQ(**faq_data)
    db.add(faq)
    await db.commit()
    await db.refresh(faq)
    return faq


async def update_faq(db: AsyncSession, faq_id: uuid.UUID, updates: dict) -> FAQ | None:
    faq = await get_faq_by_id(db, faq_id)
    if not faq:
        return None
    for key, value in updates.items():
        setattr(faq, key, value)
    await db.commit()
    await db.refresh(faq)
    return faq


async def delete_faq(db: AsyncSession, faq_id: uuid.UUID) -> bool:
    result = await db.execute(delete(FAQ).where(FAQ.id == faq_id))
    await db.commit()
    return result.rowcount > 0


async def delete_all_faqs(db: AsyncSession) -> int:
    result = await db.execute(delete(FAQ))
    await db.commit()
    return result.rowcount


async def search_similar_faq(
    db: AsyncSession,
    embedding: list[float],
    character_id: str,
    threshold: float = FAQ_SIMILARITY_THRESHOLD,
    limit: int = 1,
) -> FAQ | None:
    


    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    # Fetch all columns in one query — avoids a second get_faq_by_id round trip.
    query = text("""
        SELECT id, character_id, question, answer, audio_url, tag, language,
               emotion, created_at, updated_at,
               1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM frequently_asked_questions
        WHERE character_id = :character_id
          AND embedding IS NOT NULL
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
    """)

    result = await db.execute(query, {
        "embedding": embedding_str,
        "character_id": character_id,
        "limit": limit,
    })
    rows = result.mappings().all()

    if not rows:
        return None

    row = rows[0]
    similarity = float(row["similarity"])
    print(f"   ↳ best match similarity: {similarity:.4f} (threshold: {threshold})")
    if similarity < threshold:
        print(f"   ↳ below threshold — no FAQ match")
        return None


    faq = FAQ()
    for col in ("id", "character_id", "question", "answer", "audio_url", "tag", "language", "emotion", "created_at", "updated_at"):
        setattr(faq, col, row[col])
    return faq
