"""In-memory FAQ vector index.

Loaded once at startup from the DB. All similarity searches run as a numpy
matrix–vector multiply — no network round trip, no DB lock contention.

Typical numbers:
  • Load  : ~50–200ms for 100–500 FAQs (one DB SELECT)
  • Search: < 1ms for any realistic FAQ count (pure numpy)

Thread-safety: reads are lock-free (numpy arrays are immutable after build).
"""
import json
import time
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

import app.core.config as config


class FAQMemoryCache:
    def __init__(self):
        # character_id (lowercase) → list of (FAQ object, unit-normalised embedding ndarray)
        self._data: dict[str, list[tuple]] = {}
        self._loaded: bool = False
        self._total: int = 0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    async def load(self, db: AsyncSession) -> None:
        """Fetch every FAQ row that has an embedding and build the index."""
        from sqlalchemy import text

        result = await db.execute(text("""
            SELECT id, character_id, question, answer, audio_url, tag, language,
                   emotion, created_at, updated_at, embedding
            FROM frequently_asked_questions
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
            # The ORM mapper would handle this automatically, but we use raw SQL
            # here to avoid loading the full ORM model (faster, no session binding).
            if isinstance(raw, str):
                raw = json.loads(raw)

            # Normalise to a unit vector so cosine similarity == dot product
            # (avoids a division on every search call)
            emb = np.array(raw, dtype=np.float32)
            norm = float(np.linalg.norm(emb))
            if norm == 0:
                skipped += 1
                continue
            emb /= norm

            cid = (row["character_id"] or "").lower()

            # Reconstruct a lightweight FAQ-like object (avoids ORM session issues)
            faq = _FAQResult(
                id=row["id"],
                character_id=row["character_id"],
                question=row["question"],
                answer=row["answer"],
                audio_url=row["audio_url"],
                tag=row["tag"],
                language=row["language"],
                emotion=row["emotion"],
            )
            data.setdefault(cid, []).append((faq, emb))

        self._data = data
        self._total = sum(len(v) for v in data.values())
        self._loaded = True

        char_summary = ", ".join(f"{cid}={len(v)}" for cid, v in data.items())
        print(f"✅ [FAQ CACHE] {self._total} FAQs loaded into memory ({char_summary})"
              + (f" — {skipped} skipped (no embedding)" if skipped else ""))

    # ------------------------------------------------------------------
    # Searching
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        character_id: str,
        threshold: float | None = None,
    ) -> "_FAQResult | None":
        """Return the best-matching FAQ for *character_id*, or None.

        The query embedding does NOT need to be pre-normalised — we normalise
        it here. All stored embeddings are pre-normalised at load time.
        """
        if threshold is None:
            threshold = config.SIMILARITY_THRESHOLD

        cid = (character_id or "").lower()
        entries = self._data.get(cid, [])
        if not entries:
            print(f"   ↳ [FAQ CACHE] no FAQs for character '{cid}'")
            return None

        t0 = time.perf_counter()
        q = np.array(query_embedding, dtype=np.float32)
        norm = float(np.linalg.norm(q))
        if norm > 0:
            q /= norm

        # Batch dot product — O(n · d) but d=384 is tiny; n<500 → sub-ms
        matrix = np.stack([emb for _, emb in entries])   # (n, 384)
        similarities = matrix @ q                         # (n,)

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if best_score < threshold:
            print(f"   ↳ [FAQ CACHE] best match {best_score:.4f} (threshold {threshold}) — no match ({elapsed_ms:.1f}ms)")
            return None

        faq = entries[best_idx][0]
        print(f"   ↳ [FAQ CACHE] best match {best_score:.4f} (threshold {threshold}) — matched {faq.question[:60]!r} ({elapsed_ms:.1f}ms)")
        return faq

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
# Lightweight FAQ result object (avoids detached-ORM-session issues)
# ---------------------------------------------------------------------------

class _FAQResult:
    """Plain data container that mirrors the FAQ ORM fields the pipeline reads."""
    __slots__ = ("id", "character_id", "question", "answer", "audio_url", "tag", "language", "emotion")

    def __init__(self, *, id, character_id, question, answer, audio_url, tag, language, emotion=None):
        self.id = id
        self.character_id = character_id
        self.question = question
        self.answer = answer
        self.audio_url = audio_url
        self.tag = tag
        self.language = language
        self.emotion = emotion

    def __repr__(self) -> str:
        return f"<FAQResult character={self.character_id} question={self.question[:40]!r}>"
