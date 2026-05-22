"""
Interactive manager for the college-history RAG store (college_history_chunks).

Add, index, list, view, update, delete history chunks — and test retrieval to
see what a question would actually pull back (with similarity scores) so you can
tune HISTORY_TOP_K / HISTORY_SIMILARITY_THRESHOLD.

Usage:
    python -m helpers.history_manager
"""

import asyncio
import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

# Windows consoles default to cp1252, which can't encode the status emojis below.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.characters import characters_info
from app.core import config
from app.db.database import get_engine, get_session_factory
from app.db.repositories.history_repository import (
    create_chunk,
    delete_all_chunks,
    delete_chunk,
    delete_chunks_by_source,
    get_all_chunks,
    get_chunk_by_id,
    search_history_chunks,
    update_chunk,
)
from app.services.embedding_service import generate_embedding, generate_query_embedding
from helpers.index_history import HISTORY_DIR, index_all, index_file, list_source_files


# ── UI helpers ─────────────────────────────────────────────────────────────────

def inp(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"  {label}{suffix}: ").strip()
    return val or default


def inp_multiline(label: str) -> str:
    print(f"  {label} (end with a single '.' on its own line):")
    lines = []
    while True:
        line = input("  ")
        if line.strip() == ".":
            break
        lines.append(line)
    return "\n".join(lines).strip()


def choose(label: str, options: list[str]) -> str:
    print(f"\n  {label}")
    for i, opt in enumerate(options, 1):
        print(f"    {i}. {opt}")
    while True:
        raw = input("  Choice: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("  Invalid choice.")


def confirm(label: str) -> bool:
    return input(f"\n  {label} (y/n): ").strip().lower() == "y"


def header(title: str):
    print("\n" + "━" * 55)
    print(f"  {title}")
    print("━" * 55)


def divider():
    print("\n" + "─" * 55)


GLOBAL_LABEL = "(global — all characters)"


def choose_character(default: str | None = None) -> str | None:
    """Returns a character_id, or None for a global chunk."""
    available = list(characters_info.first_name.keys())
    display = [GLOBAL_LABEL] + [
        f"{c} — {characters_info.first_name[c]} {characters_info.last_name[c]}" for c in available
    ]
    picked = choose("Scope (which character can see this chunk?):", display)
    if picked == GLOBAL_LABEL:
        return None
    return picked.split(" — ")[0]


# ── Display ──────────────────────────────────────────────────────────────────

def print_chunk(chunk, index: int = None):
    prefix = f"[{index}] " if index is not None else ""
    scope = chunk.character_id or "global"
    sim = getattr(chunk, "similarity", None)
    print(f"\n  {prefix}ID        : {chunk.id}")
    print(f"     Scope     : {scope}")
    print(f"     Source    : {chunk.source_doc or '—'} (#{chunk.chunk_index if chunk.chunk_index is not None else '—'})")
    print(f"     Language  : {chunk.language}")
    print(f"     Tag       : {chunk.tag or '—'}")
    print(f"     Embedding : {'✅' if chunk.embedding is not None else '❌ none'}")
    if sim is not None:
        print(f"     Similarity: {sim:.4f}")
    preview = chunk.content.replace("\n", " ")
    print(f"     Content   : {preview[:120]}{'...' if len(preview) > 120 else ''}")


def _pick_chunk(chunks) -> uuid.UUID | None:
    for i, c in enumerate(chunks, 1):
        scope = c.character_id or "global"
        preview = c.content.replace("\n", " ")[:55]
        print(f"  [{i}] {scope:8s} | {c.source_doc or '—':24s} | {preview}")
    raw = inp("\nEnter number or chunk ID")
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(chunks):
            return chunks[idx].id
        print("  ❌ Number out of range.")
        return None
    try:
        return uuid.UUID(raw.strip())
    except ValueError:
        print("  ❌ Invalid input.")
        return None


# ── Actions ────────────────────────────────────────────────────────────────────

async def action_add(db):
    header("➕  ADD A HISTORY CHUNK")

    content = inp_multiline("Paste the history passage")
    if not content:
        print("  Content cannot be empty.")
        return

    character_id = choose_character()
    language = choose("Language:", ["en", "ar"])
    tag = inp("Tag (optional, e.g. 'founding', 'faculty')")
    source_doc = inp("Source label (optional, e.g. 'manual' or a document name)", default="manual")

    print("\n  ⏳ Generating embedding...")
    embedding = generate_embedding(content)
    print(f"  ✅ Embedding generated ({len(embedding)} dims)")

    chunk = await create_chunk(db, {
        "content": content,
        "embedding": embedding,
        "character_id": character_id,
        "source_doc": source_doc or None,
        "chunk_index": None,
        "language": language,
        "tag": tag or None,
    })

    header("✅  CHUNK SAVED")
    print_chunk(chunk)


async def action_index_file(db):
    header("📄  INDEX / RE-INDEX A FILE")
    files = list_source_files()
    if not files:
        print(f"  No .txt files found in {HISTORY_DIR}")
        return
    ALL = "(all files)"
    picked = choose("Which file?", [ALL] + files)
    if picked == ALL:
        await index_all(db)
    else:
        await index_file(db, os.path.join(HISTORY_DIR, picked))


async def action_list(db):
    header("📋  ALL HISTORY CHUNKS")
    chunks = await get_all_chunks(db)
    if not chunks:
        print("  No chunks in store.")
        return
    by_source: dict[str, int] = {}
    for c in chunks:
        key = c.source_doc or "(unknown)"
        by_source[key] = by_source.get(key, 0) + 1
    print(f"  {len(chunks)} chunk(s) across {len(by_source)} source(s):\n")
    for source, count in sorted(by_source.items()):
        print(f"    {source:40s} {count} chunk(s)")
    if confirm("\nShow full detail for every chunk?"):
        for i, c in enumerate(chunks, 1):
            print_chunk(c, index=i)
            divider()


async def action_view(db):
    header("🔍  VIEW A CHUNK")
    chunks = await get_all_chunks(db)
    if not chunks:
        print("  No chunks in store.")
        return
    chunk_id = _pick_chunk(chunks)
    if not chunk_id:
        return
    chunk = await get_chunk_by_id(db, chunk_id)
    if chunk:
        print_chunk(chunk)
        print(f"\n  Full content:\n  {chunk.content}")
    else:
        print("  ❌ Chunk not found.")


async def action_update(db):
    header("✏️   UPDATE A CHUNK")
    chunks = await get_all_chunks(db)
    if not chunks:
        print("  No chunks to update.")
        return
    chunk_id = _pick_chunk(chunks)
    if not chunk_id:
        return
    chunk = await get_chunk_by_id(db, chunk_id)
    if not chunk:
        print("  ❌ Chunk not found.")
        return
    print_chunk(chunk)

    field = choose("What do you want to update?", [
        "Content (regenerates embedding)",
        "Scope (character)",
        "Language",
        "Tag",
        "Regenerate embedding",
        "Cancel",
    ])
    if field == "Cancel":
        return

    updates = {}
    if field == "Content (regenerates embedding)":
        new_content = inp_multiline("New content")
        if not new_content:
            print("  Content cannot be empty.")
            return
        updates["content"] = new_content
        print("  ⏳ Regenerating embedding...")
        updates["embedding"] = generate_embedding(new_content)
        print(f"  ✅ Done ({len(updates['embedding'])} dims)")
    elif field == "Scope (character)":
        updates["character_id"] = choose_character()
    elif field == "Language":
        updates["language"] = choose("Language:", ["en", "ar"])
    elif field == "Tag":
        updates["tag"] = inp("New tag", default=chunk.tag or "") or None
    elif field == "Regenerate embedding":
        print("  ⏳ Generating embedding...")
        updates["embedding"] = generate_embedding(chunk.content)
        print(f"  ✅ Done ({len(updates['embedding'])} dims)")

    updated = await update_chunk(db, chunk_id, updates)
    if updated:
        print("\n  ✅ Updated.")
        print_chunk(updated)
    else:
        print("  ❌ Update failed.")


async def action_delete(db):
    header("🗑️   DELETE A CHUNK")
    chunks = await get_all_chunks(db)
    if not chunks:
        print("  No chunks to delete.")
        return
    chunk_id = _pick_chunk(chunks)
    if not chunk_id:
        return
    chunk = await get_chunk_by_id(db, chunk_id)
    if not chunk:
        print("  ❌ Chunk not found.")
        return
    print_chunk(chunk)
    if not confirm("Permanently delete this chunk?"):
        print("  Cancelled.")
        return
    ok = await delete_chunk(db, chunk_id)
    print("  ✅ Deleted." if ok else "  ❌ Failed to delete.")


async def action_delete_source(db):
    header("🗑️   DELETE ALL CHUNKS FROM A SOURCE")
    chunks = await get_all_chunks(db)
    if not chunks:
        print("  No chunks in store.")
        return
    sources = sorted({c.source_doc or "(unknown)" for c in chunks})
    source = choose("Which source?", sources)
    if not confirm(f"Delete all chunks from '{source}'?"):
        print("  Cancelled.")
        return
    count = await delete_chunks_by_source(db, source)
    print(f"  ✅ Deleted {count} chunk(s).")


async def action_delete_all(db):
    header("🗑️   DELETE ALL CHUNKS")
    chunks = await get_all_chunks(db)
    if not chunks:
        print("  No chunks in store.")
        return
    print(f"  This will permanently delete all {len(chunks)} chunk(s).")
    if not confirm("Are you sure? This cannot be undone"):
        print("  Cancelled.")
        return
    if not confirm("Really sure?"):
        print("  Cancelled.")
        return
    count = await delete_all_chunks(db)
    print(f"  ✅ Deleted {count} chunk(s).")


async def action_fill_missing_embeddings(db):
    header("🧠  FILL MISSING EMBEDDINGS")
    chunks = await get_all_chunks(db)
    missing = [c for c in chunks if c.embedding is None]
    if not missing:
        print("  ✅ All chunks already have embeddings.")
        return
    print(f"  Found {len(missing)} chunk(s) without embeddings.")
    if not confirm(f"Generate embeddings for all {len(missing)} chunk(s)?"):
        print("  Cancelled.")
        return
    success, failed = 0, 0
    for i, c in enumerate(missing, 1):
        print(f"  [{i}/{len(missing)}] {c.source_doc or '—'} #{c.chunk_index}")
        try:
            await update_chunk(db, c.id, {"embedding": generate_embedding(c.content)})
            success += 1
        except Exception as e:
            print(f"    ❌ {e}")
            failed += 1
    divider()
    print(f"\n  Done — ✅ {success} generated, ❌ {failed} failed.")


async def action_regenerate_all_embeddings(db):
    header("🔄  REGENERATE ALL EMBEDDINGS")
    print("  ⚠️  Overwrites EVERY embedding. Use after changing the embedding model —")
    print("     old vectors live in a different latent space and won't match queries.\n")
    chunks = await get_all_chunks(db)
    if not chunks:
        print("  No chunks found.")
        return
    if not confirm(f"Regenerate embeddings for all {len(chunks)} chunk(s)?"):
        print("  Cancelled.")
        return
    success, failed = 0, 0
    for i, c in enumerate(chunks, 1):
        print(f"  [{i}/{len(chunks)}] {c.source_doc or '—'} #{c.chunk_index}")
        try:
            await update_chunk(db, c.id, {"embedding": generate_embedding(c.content)})
            success += 1
        except Exception as e:
            print(f"    ❌ {e}")
            failed += 1
    divider()
    print(f"\n  Done — ✅ {success} regenerated, ❌ {failed} failed.")


async def action_test_retrieval(db):
    header("🔎  TEST RETRIEVAL")
    print(f"  Simulates what the pipeline retrieves on an FAQ miss.")
    print(f"  (configured: TOP_K={config.HISTORY_TOP_K}, threshold={config.HISTORY_SIMILARITY_THRESHOLD})\n")

    question = inp("Question to test")
    if not question:
        print("  Question cannot be empty.")
        return
    character_id = choose_character()

    emb = generate_query_embedding(question)
    # threshold=0.0 so you can see ALL top-K scores, including ones the pipeline would drop.
    results = await search_history_chunks(db, emb, character_id, threshold=0.0)

    if not results:
        print("\n  No chunks in scope for this character.")
        return

    threshold = config.HISTORY_SIMILARITY_THRESHOLD
    print(f"\n  Top {len(results)} by similarity (▶ = passes threshold {threshold}):")
    for c in results:
        sim = getattr(c, "similarity", 0.0)
        mark = "▶" if sim >= threshold else " "
        preview = c.content.replace("\n", " ")[:70]
        print(f"   {mark} {sim:.4f} | {c.source_doc or '—'} | {preview}")


# ── Main loop ──────────────────────────────────────────────────────────────────

async def main():
    engine = get_engine()
    session_factory = get_session_factory(engine)

    header("📚  HISTORY MANAGER (RAG)")

    actions = [
        ("Add a chunk (paste text)", action_add),
        ("Index / re-index a file", action_index_file),
        ("List all chunks", action_list),
        ("View a chunk", action_view),
        ("Update a chunk", action_update),
        ("Test retrieval (search)", action_test_retrieval),
        ("Delete a chunk", action_delete),
        ("Delete all chunks from a source", action_delete_source),
        ("Delete ALL chunks", action_delete_all),
        ("Fill missing embeddings (batch)", action_fill_missing_embeddings),
        ("Regenerate ALL embeddings (after model change)", action_regenerate_all_embeddings),
        ("Exit", None),
    ]

    try:
        while True:
            choice = choose("What would you like to do?", [label for label, _ in actions])
            handler = dict(actions)[choice]
            if handler is None:
                print("\n  Bye!\n")
                break
            async with session_factory() as db:
                await handler(db)
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
