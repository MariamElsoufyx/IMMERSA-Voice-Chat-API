"""
Interactive FAQ manager — seed, list, view, update, delete.

Usage:
    python -m scripts.faq_manager
"""

import asyncio
import io
import json
import os
import re
import sys
import uuid

import httpx
import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.characters import characters_info
from app.characters.build_prompt import build_narrator_prompts
from app.core import config
from app.core.clients import AIClients
from app.db.database import get_engine, get_session_factory
from app.db.repositories.faq_repository import (
    create_faq,
    delete_all_faqs,
    delete_faq,
    get_all_faqs,
    get_faq_by_id,
    update_faq,
)
from app.services.embedding_service import generate_embedding
from app.services.llm.groq_service import LLMGroqService

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
STORAGE_BUCKET = os.getenv("SUPABASE_AUDIO_BUCKET", "faq-audios")


# ── UI helpers ─────────────────────────────────────────────────────────────────

def inp(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"  {label}{suffix}: ").strip()
    return val or default


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


# ── Display ────────────────────────────────────────────────────────────────────

VALID_EMOTIONS = ["happy", "sad", "angry", "disgust", "surprise", "neutral"]


def print_faq(faq, index: int = None):
    prefix = f"[{index}] " if index is not None else ""
    print(f"\n  {prefix}ID        : {faq.id}")
    print(f"     Character : {faq.character_id}")
    print(f"     Language  : {faq.language}")
    print(f"     Tag       : {faq.tag or '—'}")
    print(f"     Emotion   : {faq.emotion or '—'}")
    print(f"     Question  : {faq.question}")
    print(f"     Answer    : {faq.answer[:80]}{'...' if len(faq.answer) > 80 else ''}")
    print(f"     Audio     : {'✅' if faq.audio_url else '❌ none'}")
    print(f"     Embedding : {'✅' if faq.embedding is not None else '❌ none'}")


# ── Audio helpers ──────────────────────────────────────────────────────────────

def generate_audio_bytes(text: str, character_id: str, elevenlabs_client) -> bytes | None:
    import librosa
    voice_id = characters_info.voices.get(character_id.lower())
    if not voice_id:
        print(f"  ⚠️  No voice ID configured for character {character_id}.")
        return None

    print("  ⏳ Generating audio via ElevenLabs...")
    mp3_chunks = []
    stream = elevenlabs_client.text_to_speech.convert(
        text=text,
        voice_id=voice_id,
        model_id=config.ELEVENLABS_MODEL_ID,
        output_format="mp3_44100_128",
        voice_settings=config.VOICE_SETTINGS,
    )
    for chunk in stream:
        if chunk:
            mp3_chunks.append(chunk)

    if not mp3_chunks:
        print("  ❌ ElevenLabs returned no audio.")
        return None

    mp3_bytes = b"".join(mp3_chunks)
    audio, sr = librosa.load(io.BytesIO(mp3_bytes), sr=None, mono=True)
    trimmed, _ = librosa.effects.trim(audio, top_db=35, frame_length=512, hop_length=128)
    if len(trimmed) == 0:
        trimmed = audio

    wav_buf = io.BytesIO()
    sf.write(wav_buf, trimmed, sr, format="WAV", subtype="PCM_16")
    print(f"  ✅ Audio generated ({len(trimmed)} samples @ {sr}Hz)")
    return wav_buf.getvalue()


def load_local_audio_bytes(path: str) -> bytes | None:
    """Read an already-prepared audio file from disk and normalise it to WAV PCM16."""
    import librosa
    path = path.strip().strip('"').strip("'")
    if not os.path.isfile(path):
        print(f"  ❌ File not found: {path}")
        return None
    try:
        audio, sr = librosa.load(path, sr=None, mono=True)
    except Exception as e:
        print(f"  ❌ Could not read audio file: {e}")
        return None
    if len(audio) == 0:
        print("  ❌ Audio file is empty.")
        return None
    wav_buf = io.BytesIO()
    sf.write(wav_buf, audio, sr, format="WAV", subtype="PCM_16")
    print(f"  ✅ Loaded local audio ({len(audio)} samples @ {sr}Hz)")
    return wav_buf.getvalue()


async def upload_audio(audio_bytes: bytes, character_id: str) -> str | None:
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("  ⚠️  SUPABASE_URL or SUPABASE_SERVICE_KEY not set.")
        return None

    filename = f"{character_id.lower()}_{uuid.uuid4().hex[:8]}.wav"
    url = f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{filename}"
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "audio/wav",
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, content=audio_bytes, headers=headers)

    if response.status_code in (200, 201):
        public_url = f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{filename}"
        print(f"  ✅ Uploaded → {public_url}")
        return public_url
    else:
        print(f"  ❌ Upload failed: {response.status_code} — {response.text}")
        return None


# ── LLM helper ─────────────────────────────────────────────────────────────────

def llm_generate_answer(question: str, character_id: str) -> dict:
    """Returns {"answer": str, "emotion": str | None}."""
    print("\n  ⏳ Generating answer with LLM...")
    clients = AIClients().get_all_clients()
    llm = LLMGroqService(client=clients["groq_client"])
    prompt_key = config.get_prompt_key_by_character_id(character_id)
    user_prompt, system_prompt = build_narrator_prompts(
        character_id=character_id,
        question=question,
        prompt_key=prompt_key,
    )
    raw = llm.generate_reply(user_prompt, system_prompt)
    try:
        parsed = json.loads(raw)
        return {"answer": parsed.get("answer", raw), "emotion": parsed.get("emotion")}
    except Exception:
        match = re.search(r'\{[\s\S]*\}', raw)
        if match:
            try:
                parsed = json.loads(match.group())
                return {"answer": parsed.get("answer", raw), "emotion": parsed.get("emotion")}
            except Exception:
                pass
    return {"answer": raw, "emotion": None}


# ── Resolve FAQ from list ──────────────────────────────────────────────────────

def _resolve_id(raw: str, faqs) -> uuid.UUID | None:
    if raw.isdigit():
        idx = int(raw) - 1
        if 0 <= idx < len(faqs):
            return faqs[idx].id
        print("  ❌ Number out of range.")
        return None
    try:
        return uuid.UUID(raw.strip())
    except ValueError:
        print("  ❌ Invalid input.")
        return None


def _pick_faq(faqs) -> uuid.UUID | None:
    for i, faq in enumerate(faqs, 1):
        print(f"  [{i}] {faq.character_id} | {faq.question[:60]}")
    raw = inp("\nEnter number or FAQ ID")
    return _resolve_id(raw, faqs)


# ── Actions ────────────────────────────────────────────────────────────────────

async def action_add(db):
    header("➕  ADD A FAQ")

    # Question
    question = inp("Question")
    if not question:
        print("  Question cannot be empty.")
        return

    # Character
    available = list(characters_info.first_name.keys())
    display = [f"{c} — {characters_info.first_name[c]} {characters_info.last_name[c]}" for c in available]
    character_id = choose("Character:", display).split(" — ")[0]

    # Language & tag
    language = choose("Language:", ["en", "ar"])
    tag = inp("Tag (optional, e.g. 'academics', 'personal')")

    # Answer
    emotion = None
    source = choose("Answer source:", ["Generate with LLM", "Type manually"])
    if source == "Generate with LLM":
        result = llm_generate_answer(question, character_id)
        answer = result["answer"]
        emotion = result["emotion"]
        print(f"\n  💬 LLM answer:\n  {answer}")
        print(f"  🎭 Emotion: {emotion or '—'}\n")
        if not confirm("Use this answer?"):
            answer = inp("Enter your answer")
            emotion = None
    else:
        answer = inp("Answer")

    if not answer:
        print("  Answer cannot be empty.")
        return

    # Emotion (allow override or manual set)
    emotion = choose("Emotion:", VALID_EMOTIONS + ["none"]) if emotion is None else emotion
    if emotion == "none":
        emotion = None

    # Audio
    audio_url = None
    audio_source = choose("Audio:", [
        "Generate from answer (ElevenLabs)",
        "Upload a ready audio file",
        "Skip",
    ])
    if audio_source == "Generate from answer (ElevenLabs)":
        clients = AIClients().get_all_clients()
        audio_bytes = generate_audio_bytes(answer, character_id, clients["elevenlabs_client"])
        if audio_bytes:
            audio_url = await upload_audio(audio_bytes, character_id)
    elif audio_source == "Upload a ready audio file":
        audio_bytes = load_local_audio_bytes(inp("Path to audio file"))
        if audio_bytes:
            audio_url = await upload_audio(audio_bytes, character_id)

    # Embedding
    print("\n  ⏳ Generating embedding...")
    embedding = generate_embedding(question)
    print(f"  ✅ Embedding generated ({len(embedding)} dims)")

    # Save
    print("\n  ⏳ Saving to database...")
    faq = await create_faq(db, {
        "character_id": character_id,
        "question": question,
        "answer": answer,
        "language": language,
        "tag": tag or None,
        "emotion": emotion,
        "audio_url": audio_url,
        "embedding": embedding,
    })

    header("✅  FAQ SAVED")
    print_faq(faq)


async def action_list(db):
    header("📋  ALL FAQs")
    faqs = await get_all_faqs(db)
    if not faqs:
        print("  No FAQs in database.")
        return
    for i, faq in enumerate(faqs, 1):
        print_faq(faq, index=i)
        divider()
    print(f"\n  Total: {len(faqs)} FAQ(s)")


async def action_view(db):
    header("🔍  VIEW A FAQ")
    faqs = await get_all_faqs(db)
    if not faqs:
        print("  No FAQs in database.")
        return
    faq_id = _pick_faq(faqs)
    if not faq_id:
        return
    faq = await get_faq_by_id(db, faq_id)
    if faq:
        print_faq(faq)
    else:
        print("  ❌ FAQ not found.")


async def action_update(db):
    header("✏️   UPDATE A FAQ")
    faqs = await get_all_faqs(db)
    if not faqs:
        print("  No FAQs to update.")
        return

    faq_id = _pick_faq(faqs)
    if not faq_id:
        return
    faq = await get_faq_by_id(db, faq_id)
    if not faq:
        print("  ❌ FAQ not found.")
        return

    print_faq(faq)

    field = choose("What do you want to update?", [
        "Question",
        "Answer",
        "Emotion",
        "Language",
        "Tag",
        "Regenerate embedding",
        "Regenerate audio",
        "Upload a ready audio file",
        "Update answer + regenerate embedding + regenerate audio",
        "Cancel",
    ])

    if field == "Cancel":
        return

    updates = {}
    clients_cache = {}

    def get_clients():
        if not clients_cache:
            clients_cache["c"] = AIClients().get_all_clients()
        return clients_cache["c"]

    if field == "Question":
        new_q = inp("New question", default=faq.question)
        updates["question"] = new_q
        if confirm("Regenerate embedding for new question?"):
            print("  ⏳ Generating embedding...")
            updates["embedding"] = generate_embedding(new_q)
            print(f"  ✅ Done ({len(updates['embedding'])} dims)")

    elif field == "Answer":
        source = choose("Answer source:", ["Generate with LLM", "Type manually"])
        if source == "Generate with LLM":
            result = llm_generate_answer(faq.question, faq.character_id)
            new_a = result["answer"]
            print(f"\n  💬 LLM answer:\n  {new_a}")
            print(f"  🎭 Emotion: {result['emotion'] or '—'}\n")
            if not confirm("Use this answer?"):
                new_a = inp("Enter your answer")
            elif result["emotion"]:
                updates["emotion"] = result["emotion"]
        else:
            new_a = inp("New answer", default=faq.answer)
        updates["answer"] = new_a
        if confirm("Regenerate audio for new answer?"):
            audio_bytes = generate_audio_bytes(new_a, faq.character_id, get_clients()["elevenlabs_client"])
            if audio_bytes:
                audio_url = await upload_audio(audio_bytes, faq.character_id)
                if audio_url:
                    updates["audio_url"] = audio_url

    elif field == "Emotion":
        current = faq.emotion or "none"
        print(f"  Current emotion: {current}")
        new_emotion = choose("New emotion:", VALID_EMOTIONS + ["none"])
        updates["emotion"] = None if new_emotion == "none" else new_emotion

    elif field == "Language":
        updates["language"] = choose("Language:", ["en", "ar"])

    elif field == "Tag":
        updates["tag"] = inp("New tag", default=faq.tag or "") or None

    elif field == "Regenerate embedding":
        print("  ⏳ Generating embedding...")
        updates["embedding"] = generate_embedding(faq.question)
        print(f"  ✅ Done ({len(updates['embedding'])} dims)")

    elif field == "Regenerate audio":
        audio_bytes = generate_audio_bytes(faq.answer, faq.character_id, get_clients()["elevenlabs_client"])
        if audio_bytes:
            audio_url = await upload_audio(audio_bytes, faq.character_id)
            if audio_url:
                updates["audio_url"] = audio_url

    elif field == "Upload a ready audio file":
        audio_bytes = load_local_audio_bytes(inp("Path to audio file"))
        if audio_bytes:
            audio_url = await upload_audio(audio_bytes, faq.character_id)
            if audio_url:
                updates["audio_url"] = audio_url

    elif field == "Update answer + regenerate embedding + regenerate audio":
        source = choose("Answer source:", ["Generate with LLM", "Type manually"])
        if source == "Generate with LLM":
            result = llm_generate_answer(faq.question, faq.character_id)
            new_a = result["answer"]
            print(f"\n  💬 LLM answer:\n  {new_a}")
            print(f"  🎭 Emotion: {result['emotion'] or '—'}\n")
            if not confirm("Use this answer?"):
                new_a = inp("Enter your answer")
            elif result["emotion"]:
                updates["emotion"] = result["emotion"]
        else:
            new_a = inp("New answer", default=faq.answer)
        updates["answer"] = new_a

        print("  ⏳ Generating embedding...")
        updates["embedding"] = generate_embedding(faq.question)
        print(f"  ✅ Embedding done")

        audio_bytes = generate_audio_bytes(new_a, faq.character_id, get_clients()["elevenlabs_client"])
        if audio_bytes:
            audio_url = await upload_audio(audio_bytes, faq.character_id)
            if audio_url:
                updates["audio_url"] = audio_url

    if not updates:
        print("  Nothing to update.")
        return

    updated = await update_faq(db, faq_id, updates)
    if updated:
        print("\n  ✅ Updated successfully.")
        print_faq(updated)
    else:
        print("  ❌ Update failed.")


async def action_delete(db):
    header("🗑️   DELETE A FAQ")
    faqs = await get_all_faqs(db)
    if not faqs:
        print("  No FAQs to delete.")
        return

    faq_id = _pick_faq(faqs)
    if not faq_id:
        return
    faq = await get_faq_by_id(db, faq_id)
    if not faq:
        print("  ❌ FAQ not found.")
        return

    print_faq(faq)
    if not confirm("Permanently delete this FAQ?"):
        print("  Cancelled.")
        return

    deleted = await delete_faq(db, faq_id)
    print("  ✅ Deleted." if deleted else "  ❌ Failed to delete.")


async def action_delete_all(db):
    header("🗑️   DELETE ALL FAQs")
    faqs = await get_all_faqs(db)
    if not faqs:
        print("  No FAQs in database.")
        return
    print(f"  This will permanently delete all {len(faqs)} FAQ(s).")
    if not confirm("Are you sure? This cannot be undone"):
        print("  Cancelled.")
        return
    if not confirm("Really sure?"):
        print("  Cancelled.")
        return
    count = await delete_all_faqs(db)
    print(f"  ✅ Deleted {count} FAQ(s).")


async def action_fill_missing_emotions(db):
    header("🎭  FILL MISSING EMOTIONS")
    faqs = await get_all_faqs(db)
    missing = [f for f in faqs if not f.emotion]

    if not missing:
        print("  ✅ All FAQs already have an emotion.")
        return

    print(f"  Found {len(missing)} FAQ(s) without emotion:\n")
    for i, faq in enumerate(missing, 1):
        print(f"  [{i}/{len(missing)}] {faq.character_id} | {faq.question[:60]}")

    source = choose("\nHow to assign emotions?", ["Generate with LLM (auto)", "Choose manually for each"])

    success, failed = 0, 0

    for i, faq in enumerate(missing, 1):
        print(f"\n  ── [{i}/{len(missing)}] {faq.question[:60]}")
        print(f"     Answer: {faq.answer[:80]}{'...' if len(faq.answer) > 80 else ''}")

        if source == "Generate with LLM (auto)":
            try:
                result = llm_generate_answer(faq.question, faq.character_id)
                emotion = result["emotion"]
                if emotion and emotion.lower() in VALID_EMOTIONS:
                    emotion = emotion.lower()
                    print(f"  🎭 LLM suggested: {emotion}")
                    if confirm("Save this emotion?"):
                        await update_faq(db, faq.id, {"emotion": emotion})
                        print(f"  ✅ Saved")
                        success += 1
                    else:
                        emotion = choose("  Pick emotion:", VALID_EMOTIONS + ["skip"])
                        if emotion != "skip":
                            await update_faq(db, faq.id, {"emotion": emotion})
                            success += 1
                else:
                    print(f"  ⚠️  LLM returned no valid emotion — pick manually")
                    emotion = choose("  Pick emotion:", VALID_EMOTIONS + ["skip"])
                    if emotion != "skip":
                        await update_faq(db, faq.id, {"emotion": emotion})
                        success += 1
            except Exception as e:
                print(f"  ❌ LLM failed: {e}")
                failed += 1
        else:
            emotion = choose("  Pick emotion:", VALID_EMOTIONS + ["skip"])
            if emotion != "skip":
                await update_faq(db, faq.id, {"emotion": emotion})
                print(f"  ✅ Saved")
                success += 1

    divider()
    print(f"\n  Done — ✅ {success} set, ❌ {failed} failed.")


async def action_fill_missing_audio(db):
    header("🎵  FILL MISSING AUDIO")
    faqs = await get_all_faqs(db)
    missing = [f for f in faqs if not f.audio_url]

    if not missing:
        print("  ✅ All FAQs already have audio.")
        return

    print(f"  Found {len(missing)} FAQ(s) without audio:\n")
    for i, faq in enumerate(missing, 1):
        print(f"  [{i}/{len(missing)}] {faq.character_id} | {faq.question[:60]}")

    if not confirm(f"\nGenerate and upload audio for all {len(missing)} FAQs?"):
        print("  Cancelled.")
        return

    clients = AIClients().get_all_clients()
    success, failed = 0, 0

    for i, faq in enumerate(missing, 1):
        print(f"\n  ── [{i}/{len(missing)}] {faq.question[:60]}")
        audio_bytes = generate_audio_bytes(faq.answer, faq.character_id, clients["elevenlabs_client"])
        if not audio_bytes:
            print(f"  ⚠️  Skipped — no voice configured for {faq.character_id}")
            failed += 1
            continue

        audio_url = await upload_audio(audio_bytes, faq.character_id)
        if audio_url:
            await update_faq(db, faq.id, {"audio_url": audio_url})
            success += 1
        else:
            failed += 1

    divider()
    print(f"\n  Done — ✅ {success} uploaded, ❌ {failed} failed.")


async def action_fill_missing_embeddings(db):
    header("🧠  FILL MISSING EMBEDDINGS")
    faqs = await get_all_faqs(db)
    missing = [f for f in faqs if f.embedding is None]

    if not missing:
        print("  ✅ All FAQs already have embeddings.")
        return

    print(f"  Found {len(missing)} FAQ(s) without embeddings:\n")
    for i, faq in enumerate(missing, 1):
        print(f"  [{i}/{len(missing)}] {faq.character_id} | {faq.question[:60]}")

    if not confirm(f"\nGenerate embeddings for all {len(missing)} FAQs?"):
        print("  Cancelled.")
        return

    success, failed = 0, 0

    for i, faq in enumerate(missing, 1):
        print(f"\n  ── [{i}/{len(missing)}] {faq.question[:60]}")
        try:
            embedding = generate_embedding(faq.question)
            await update_faq(db, faq.id, {"embedding": embedding})
            print(f"  ✅ Embedding saved ({len(embedding)} dims)")
            success += 1
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            failed += 1

    divider()
    print(f"\n  Done — ✅ {success} generated, ❌ {failed} failed.")


async def action_regenerate_all_embeddings(db):
    header("🔄  REGENERATE ALL EMBEDDINGS")
    print("  ⚠️  This overwrites EVERY existing embedding. Use after changing")
    print("     the embedding model — old vectors live in a different latent")
    print("     space and won't match new queries.\n")

    faqs = await get_all_faqs(db)
    if not faqs:
        print("  No FAQs found.")
        return

    print(f"  Found {len(faqs)} FAQ(s) to re-embed.\n")
    if not confirm(f"Regenerate embeddings for all {len(faqs)} FAQs?"):
        print("  Cancelled.")
        return

    success, failed = 0, 0

    for i, faq in enumerate(faqs, 1):
        print(f"\n  ── [{i}/{len(faqs)}] {faq.question[:60]}")
        try:
            embedding = generate_embedding(faq.question)
            await update_faq(db, faq.id, {"embedding": embedding})
            print(f"  ✅ Embedding regenerated ({len(embedding)} dims)")
            success += 1
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            failed += 1

    divider()
    print(f"\n  Done — ✅ {success} regenerated, ❌ {failed} failed.")


async def action_regenerate_all_audio(db):
    header("🔄  REGENERATE ALL AUDIO")
    print("  ⚠️  This overwrites existing audio by re-synthesizing every answer")
    print("     with the character's voice. Use after changing a voice_id or")
    print("     voice settings.\n")

    faqs = await get_all_faqs(db)
    if not faqs:
        print("  No FAQs found.")
        return

    # Scope by character (each character maps to one voice_id)
    chars = sorted({f.character_id for f in faqs})
    scope = choose("Which FAQs?", ["All characters"] + [f"Only {c}" for c in chars])
    if scope == "All characters":
        targets = faqs
    else:
        target_char = scope.replace("Only ", "", 1)
        targets = [f for f in faqs if f.character_id == target_char]

    if not targets:
        print("  No matching FAQs.")
        return

    print(f"\n  Found {len(targets)} FAQ(s) to regenerate audio for.\n")
    if not confirm(f"Regenerate audio for all {len(targets)} FAQs?"):
        print("  Cancelled.")
        return

    clients = AIClients().get_all_clients()
    success, failed = 0, 0

    for i, faq in enumerate(targets, 1):
        print(f"\n  ── [{i}/{len(targets)}] {faq.character_id} | {faq.question[:60]}")
        audio_bytes = generate_audio_bytes(faq.answer, faq.character_id, clients["elevenlabs_client"])
        if not audio_bytes:
            print(f"  ⚠️  Skipped — no voice configured for {faq.character_id}")
            failed += 1
            continue

        audio_url = await upload_audio(audio_bytes, faq.character_id)
        if audio_url:
            await update_faq(db, faq.id, {"audio_url": audio_url})
            success += 1
        else:
            failed += 1

    divider()
    print(f"\n  Done — ✅ {success} regenerated, ❌ {failed} failed.")


# ── Main loop ──────────────────────────────────────────────────────────────────

async def main():
    engine = get_engine()
    session_factory = get_session_factory(engine)

    header("📚  FAQ MANAGER")

    while True:
        action = choose("What would you like to do?", [
            "Add a FAQ",
            "List all FAQs",
            "View a FAQ",
            "Update a FAQ",
            "Delete a FAQ",
            "Delete ALL FAQs",
            "Fill missing audio (batch)",
            "Regenerate ALL audio (after voice change)",
            "Fill missing embeddings (batch)",
            "Regenerate ALL embeddings (after model change)",
            "Fill missing emotions (batch)",
            "Exit",
        ])

        async with session_factory() as db:
            if action == "Add a FAQ":
                await action_add(db)
            elif action == "List all FAQs":
                await action_list(db)
            elif action == "View a FAQ":
                await action_view(db)
            elif action == "Update a FAQ":
                await action_update(db)
            elif action == "Delete a FAQ":
                await action_delete(db)
            elif action == "Delete ALL FAQs":
                await action_delete_all(db)
            elif action == "Fill missing audio (batch)":
                await action_fill_missing_audio(db)
            elif action == "Regenerate ALL audio (after voice change)":
                await action_regenerate_all_audio(db)
            elif action == "Fill missing embeddings (batch)":
                await action_fill_missing_embeddings(db)
            elif action == "Regenerate ALL embeddings (after model change)":
                await action_regenerate_all_embeddings(db)
            elif action == "Fill missing emotions (batch)":
                await action_fill_missing_emotions(db)
            elif action == "Exit":
                print("\n  Bye!\n")
                break

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
