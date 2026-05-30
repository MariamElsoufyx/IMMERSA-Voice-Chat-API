"""In-memory history-chunk vector index (RAG grounding).

Mirrors FAQMemoryCache. Loaded once at startup from the DB; every retrieval runs
as a numpy matrix–vector multiply — no network round trip, no DB connection in
the request hot path. This matters because FAQ lookups hit the in-memory cache,
so RAG was previously the ONLY thing touching the remote DB per request — and a
cold/stale Supabase connection there could blow past the lookup timeout.

Typical numbers:
  • Load  : one DB SELECT at startup
  • Search: < 1ms for any realistic chunk count (pure numpy)

Scope: a search for character X considers X's own chunks AND global chunks
(character_id IS NULL), matching the SQL `character_id = :cid OR IS NULL`.

Thread-safety: reads are lock-free (numpy arrays are immutable after build).
"""
import json
import time
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

import app.core.config as config

# Internal key under which global (character_id IS NULL) chunks are stored.
_GLOBAL_KEY = "__global__"


class HistoryMemoryCache:
    def __init__(self):
        # key (lowercased character_id, or _GLOBAL_KEY) → list of (_HistoryResult, unit-normalised ndarray)
        self._data: dict[str, list[tuple]] = {}
        self._loaded: bool = False
        self._total: int = 0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    async def load(self, db: AsyncSession) -> None:
        """Fetch every history chunk that has an embedding and build the index."""
        from sqlalchemy import text

        result = await db.execute(text("""
            SELECT id, content, character_id, source_doc, chunk_index,
                   language, tag, embedding
            FROM "RAG_data"
            WHERE embedding IS NOT NULL
        """))
        rows = result.mappings().all()

        data: dict[str, list[tuple]] = {}
        skipped = 0

        for row in rows:
            raw = row["embedding"]
            if raw is None:
                skipped += 1
                continue

            # Raw SQL returns pgvector as a string '[-0.09, 0.03, ...]' — parse it.
            if isinstance(raw, str):
                raw = json.loads(raw)

            # Normalise to a unit vector so cosine similarity == dot product.
            emb = np.array(raw, dtype=np.float32)
            norm = float(np.linalg.norm(emb))
            if norm == 0:
                skipped += 1
                continue
            emb /= norm

            chunk = _HistoryResult(
                id=row["id"],
                content=row["content"],
                character_id=row["character_id"],
                source_doc=row["source_doc"],
                chunk_index=row["chunk_index"],
                language=row["language"],
                tag=row["tag"],
            )
            key = (row["character_id"] or "").lower() or _GLOBAL_KEY
            data.setdefault(key, []).append((chunk, emb))

        self._data = data
        self._total = sum(len(v) for v in data.values())
        self._loaded = True

        summary = ", ".join(f"{k}={len(v)}" for k, v in data.items()) or "empty"
        print(f"✅ [HISTORY CACHE] {self._total} chunks loaded into memory ({summary})"
              + (f" — {skipped} skipped (no embedding)" if skipped else ""))

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        character_id: str | None,
        threshold: float | None = None,
        limit: int | None = None,
        quiet: bool = False,
    ) -> list["_HistoryResult"]:
        """Return up to *limit* chunks (character-scoped + global) above *threshold*,
        sorted by similarity desc. The query embedding is normalised here."""
        if threshold is None:
            threshold = config.HISTORY_SIMILARITY_THRESHOLD
        if limit is None:
            limit = config.HISTORY_TOP_K

        cid = (character_id or "").lower()
        # Character-specific chunks + global chunks (mirrors the SQL OR character_id IS NULL).
        entries = self._data.get(cid, []) + self._data.get(_GLOBAL_KEY, [])
        if not entries:
            if not quiet:
                print(f"   ↳ [HISTORY CACHE] no chunks in scope (character={cid or 'global'})")
            return []

        q = np.array(query_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q /= norm

        matrix = np.stack([emb for _, emb in entries])   # (n, 384)
        similarities = matrix @ q                         # (n,)

        # Indices of the top-`limit` scores, highest first.
        order = np.argsort(similarities)[::-1][:limit]

        if not quiet:
            best = float(similarities[order[0]])
            top = ", ".join(f"{float(similarities[i]):.4f}" for i in order)
            if best < threshold:
                print(f"   ↳ [HISTORY CACHE] no chunk cleared threshold {threshold} — closest {best:.4f}  [top: {top}]")
            else:
                print(f"   ↳ [HISTORY CACHE] best {best:.4f} (threshold {threshold})  [top: {top}]")

        results: list[_HistoryResult] = []
        for idx in order:
            score = float(similarities[idx])
            if score < threshold:
                break  # sorted desc — nothing after this clears the threshold
            chunk = entries[int(idx)][0]
            chunk.similarity = score  # transient; harmless if overwritten by a later search
            results.append(chunk)
        return results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def size(self) -> int:
        return self._total


# ---------------------------------------------------------------------------
# Lightweight result object (avoids detached-ORM-session issues)
# ---------------------------------------------------------------------------

class _HistoryResult:
    """Plain data container mirroring the HistoryChunk fields the pipeline reads.
    `content` and `source_doc` are what the prompt builder uses."""
    __slots__ = ("id", "content", "character_id", "source_doc", "chunk_index", "language", "tag", "similarity")

    def __init__(self, *, id, content, character_id, source_doc, chunk_index, language, tag):
        self.id = id
        self.content = content
        self.character_id = character_id
        self.source_doc = source_doc
        self.chunk_index = chunk_index
        self.language = language
        self.tag = tag
        self.similarity = None

    def __repr__(self) -> str:
        return f"<HistoryResult source={self.source_doc} content={self.content[:40]!r}>"
