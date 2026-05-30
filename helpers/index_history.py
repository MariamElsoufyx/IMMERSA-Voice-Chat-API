"""
Indexing helpers for the college-history RAG store (RAG_data).

This is a library module — it has no CLI entry point. Use it through the
interactive manager:

    python -m helpers.history_manager   →  "Index / re-index a file"

Each source file in data/history/ is split into passages, embedded in batch, and
inserted. Indexing is idempotent per file — a file's old chunks are deleted
before its new ones land, so editing a file and re-indexing never duplicates.

Filename convention (optional):
    <character>__<topic>.txt   e.g. s1__founding.txt
      → character_id = "s1", tag = "founding"
    <topic>.txt                e.g. campus_history.txt
      → character_id = NULL (global, shared by all characters), tag = "campus_history"
(.md files are also accepted if you have any.)
"""

import os
import re

from app.db.repositories.history_repository import (
    create_chunk,
    delete_chunks_by_source,
)
from app.services.embedding_service import generate_embeddings_batch

HISTORY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "history"
)

# Chunk sizing — see the chunking discussion: ~200-400 words per passage.
TARGET_MAX_WORDS = 350   # accumulate paragraphs up to this, then start a new chunk
MIN_CHUNK_WORDS = 20     # passages shorter than this get merged forward (avoid tiny fragments)

_SENTENCE_SPLIT = re.compile(r'(?<=[.!?؟])\s+')


def _word_count(text: str) -> int:
    return len(text.split())


def _last_sentence(text: str) -> str:
    """The trailing sentence of a chunk — carried into the next chunk as overlap so a
    fact straddling a boundary survives in both."""
    sentences = [s for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]
    return sentences[-1] if sentences else ""


def chunk_text(text: str) -> list[str]:
    """Split source text into ~200-400 word passages on paragraph boundaries,
    carrying one sentence of overlap between adjacent chunks."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = _word_count(para)
        # Start a new chunk when the current one is full — but only if it already
        # has content, so a single oversized paragraph still becomes one chunk.
        if current and current_words + para_words > TARGET_MAX_WORDS:
            chunk = "\n\n".join(current)
            chunks.append(chunk)
            overlap = _last_sentence(chunk)
            current = [overlap] if overlap else []
            current_words = _word_count(overlap) if overlap else 0
        current.append(para)
        current_words += para_words

    if current:
        chunks.append("\n\n".join(current))

    # Merge any too-small trailing fragment back into the previous chunk.
    if len(chunks) >= 2 and _word_count(chunks[-1]) < MIN_CHUNK_WORDS:
        chunks[-2] = chunks[-2] + "\n\n" + chunks[-1]
        chunks.pop()

    return chunks


def parse_filename(filename: str) -> tuple[str | None, str]:
    """Derive (character_id, tag) from the filename convention.
    '<character>__<topic>.ext' → (character, topic); '<topic>.ext' → (None, topic)."""
    stem = os.path.splitext(filename)[0]
    if "__" in stem:
        character_id, topic = stem.split("__", 1)
        return character_id.lower(), topic
    return None, stem


def list_source_files() -> list[str]:
    """Return the .txt/.md filenames currently sitting in data/history/."""
    if not os.path.isdir(HISTORY_DIR):
        return []
    return sorted(f for f in os.listdir(HISTORY_DIR) if f.lower().endswith((".txt", ".md")))


async def index_file(db, path: str) -> int:
    """Chunk, embed, and (idempotently) insert one source file. Returns chunk count."""
    filename = os.path.basename(path)
    character_id, tag = parse_filename(filename)

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    if not chunks:
        print(f"  ⚠️  {filename}: no content — skipped")
        return 0

    # Idempotent: clear this file's previous chunks before inserting fresh ones.
    removed = await delete_chunks_by_source(db, filename)
    if removed:
        print(f"  ↻ {filename}: removed {removed} stale chunk(s)")

    print(f"  ⏳ {filename}: embedding {len(chunks)} chunk(s)...")
    embeddings = generate_embeddings_batch(chunks)

    for i, (content, embedding) in enumerate(zip(chunks, embeddings)):
        await create_chunk(db, {
            "content": content,
            "embedding": embedding,
            "character_id": character_id,
            "source_doc": filename,
            "chunk_index": i,
            "tag": tag,
        })

    scope = character_id or "global"
    print(f"  ✅ {filename}: indexed {len(chunks)} chunk(s) (scope={scope}, tag={tag})")
    return len(chunks)


async def index_all(db) -> int:
    """(Re)index every source file in data/history/. Returns total chunk count."""
    files = list_source_files()
    if not files:
        print(f"  ❌ No .txt files found in {HISTORY_DIR}")
        return 0
    total = 0
    for filename in files:
        total += await index_file(db, os.path.join(HISTORY_DIR, filename))
    print(f"\n  Done — {total} chunk(s) indexed across {len(files)} file(s).")
    return total
