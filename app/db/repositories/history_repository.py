import uuid

from sqlalchemy import select, delete, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import HISTORY_TOP_K, HISTORY_SIMILARITY_THRESHOLD
from app.db.models import HistoryChunk
from app.utils.log import log


async def get_all_chunks(db: AsyncSession) -> list[HistoryChunk]:
    result = await db.execute(select(HistoryChunk).order_by(HistoryChunk.created_at.desc()))
    return result.scalars().all()


async def get_chunk_by_id(db: AsyncSession, chunk_id: uuid.UUID) -> HistoryChunk | None:
    result = await db.execute(select(HistoryChunk).where(HistoryChunk.id == chunk_id))
    return result.scalar_one_or_none()


async def update_chunk(db: AsyncSession, chunk_id: uuid.UUID, updates: dict) -> HistoryChunk | None:
    chunk = await get_chunk_by_id(db, chunk_id)
    if not chunk:
        return None
    if "character_id" in updates and updates["character_id"]:
        updates = {**updates, "character_id": updates["character_id"].lower()}
    for key, value in updates.items():
        setattr(chunk, key, value)
    await db.commit()
    await db.refresh(chunk)
    return chunk


async def delete_chunk(db: AsyncSession, chunk_id: uuid.UUID) -> bool:
    result = await db.execute(delete(HistoryChunk).where(HistoryChunk.id == chunk_id))
    await db.commit()
    return result.rowcount > 0


async def create_chunk(db: AsyncSession, chunk_data: dict) -> HistoryChunk:
    cid = chunk_data.get("character_id")
    chunk_data = {**chunk_data, "character_id": cid.lower() if cid else None}
    chunk = HistoryChunk(**chunk_data)
    db.add(chunk)
    await db.commit()
    await db.refresh(chunk)
    return chunk


async def delete_chunks_by_source(db: AsyncSession, source_doc: str) -> int:

    result = await db.execute(delete(HistoryChunk).where(HistoryChunk.source_doc == source_doc))
    await db.commit()
    return result.rowcount


async def delete_all_chunks(db: AsyncSession) -> int:
    result = await db.execute(delete(HistoryChunk))
    await db.commit()
    return result.rowcount


async def search_history_chunks(
    db: AsyncSession,
    embedding: list[float],
    character_id: str | None,
    threshold: float = HISTORY_SIMILARITY_THRESHOLD,
    limit: int = HISTORY_TOP_K,
) -> list[HistoryChunk]:
 
    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
    cid = character_id.lower() if character_id else None

    query = text("""
        SELECT id, content, character_id, source_doc, chunk_index, language, tag,
               created_at, updated_at,
               1 - (embedding <=> CAST(:embedding AS vector)) AS similarity
        FROM "RAG_data"
        WHERE embedding IS NOT NULL
          AND (character_id = :character_id OR character_id IS NULL)
        ORDER BY embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
    """)

    result = await db.execute(query, {
        "embedding": embedding_str,
        "character_id": cid,
        "limit": limit,
    })
    rows = result.mappings().all()


    if rows:
        best = float(rows[0]["similarity"])
        scores = ", ".join(f"{float(r['similarity']):.4f}" for r in rows)
        if best < threshold:
            log.info("RAG", f"no chunk cleared threshold {threshold} — closest similarity: {best:.4f}")
        else:
            log.info("RAG", f"best similarity: {best:.4f} (threshold: {threshold})")
        log.detail(f"top-{len(rows)} scores: [{scores}]")
    else:
        log.info("RAG", f"no chunks in scope (character={cid or 'global'})")

    chunks: list[HistoryChunk] = []
    for row in rows:
        similarity = float(row["similarity"])
        if similarity < threshold:
            continue
        chunk = HistoryChunk()
        for col in ("id", "content", "character_id", "source_doc", "chunk_index", "language", "tag", "created_at", "updated_at"):
            setattr(chunk, col, row[col])
        chunk.similarity = similarity 
        chunks.append(chunk)
    return chunks
