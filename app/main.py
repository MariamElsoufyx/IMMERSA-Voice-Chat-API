import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.api.websocket_routes import router as websocket_router
from app.api.faq_routes import router as faq_router
from app.core.clients import AIClients
from app.services.streaming.connection_manager import ConnectionManager
import app.core.config as config
from app.services.audio.preprocessor import AudioPreprocessor
from app.services.audio.local_preprocessor import LocalAudioPreprocessor
from app.services.stt.local_whisper import STTWhisperService
from app.services.llm.openai_service import LLMOpenAIService
from app.services.stt.groq_whisper import STTGroqWhisperService
from app.services.llm.groq_service import LLMGroqService
from app.services.llm.local_llm_service import LLMLocalService
from app.services.tts.elevenlabs_service import AudioGenerationElevenLabsService
from app.services.tts.local_tts_service import AudioGenerationLocalService
from app.services.pipeline.pipeline import Pipeline
from app.characters import characters_info
from app.db.database import get_engine, get_session_factory
from app.services.embedding_service import generate_embedding
from app.services.faq_memory_cache import FAQMemoryCache
from app.services.history_memory_cache import HistoryMemoryCache


def _print_startup_banner():
    """One-shot summary of which provider is wired in for each pluggable service."""
    mode = "LOCAL (from-scratch)" if config.USE_LOCAL_SERVICES else "API"
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🚀 [STARTUP] Loading models — service mode: {mode}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    rows = [
        ("Preprocessing", config.preprocessing_provider,
         "pure-numpy LocalAudioPreprocessor" if config.preprocessing_provider == "local"
         else "scipy + librosa + noisereduce AudioPreprocessor"),
        ("STT",           config.stt_provider,
         f"faster-whisper {config.whisper_model_size} on {config.whisper_device}" if config.stt_provider == "local"
         else f"Groq Whisper API ({config.groq_whisper_model})"),
        ("LLM",           config.llm_provider,
         f"LLMLocalService ({config.local_llm_backend} → {config.local_llm_model_name})" if config.llm_provider == "local"
         else f"Groq API ({config.groq_model_name})"),
        ("TTS",           config.tts_provider,
         f"local Piper ({len(config.local_tts_voices)} voices)" if config.tts_provider == "local"
         else f"ElevenLabs API ({config.ELEVENLABS_MODEL_ID})"),
        ("Moderation",    config.moderation_provider,
         f"local {config.local_moderation_model}" if config.moderation_provider == "local"
         else "OpenAI omni-moderation-latest"),
        ("LLM Judge",     config.llm_judge_provider,
         f"local LLM judge ({config.local_llm_judge_model_name})" if config.llm_judge_provider == "local"
         else f"OpenAI ({config.openai_verifier_model_name})"),
        ("Storage",       config.storage_provider,
         f"LocalStorageService → {config.local_storage_root}" if config.storage_provider == "local"
         else "Supabase Storage"),
    ]
    for label, provider, detail in rows:
        tag = "🏠 local" if provider == "local" else "☁️  api  "
        print(f"  {tag}  {label:<14} : {detail}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _print_startup_banner()
    await asyncio.to_thread(generate_embedding, "warmup")  # load model + run first inference to eliminate cold start
    models = AIClients().get_all_clients()

    # Pick STT service based on config
    if config.stt_provider == "groq":
        stt_service = STTGroqWhisperService(client=models["groq_client"])
    else:
        stt_service = STTWhisperService(model=models["whisper_model"])

    # Pick preprocessor based on config — same interface, different deps
    if config.preprocessing_provider == "local":
        audio_preprocessor = LocalAudioPreprocessor()
    else:
        audio_preprocessor = AudioPreprocessor()

    # Pick LLM service based on config
    if config.llm_provider == "local":
        llm_service = LLMLocalService()
    elif config.llm_provider == "openai":
        llm_service = LLMOpenAIService(client=models["openai_client"])
    else:
        llm_service = LLMGroqService(client=models["groq_client"])

    # Pick TTS service based on config
    if config.tts_provider == "local":
        tts_service = AudioGenerationLocalService()
    else:
        tts_service = AudioGenerationElevenLabsService(
            client=models["elevenlabs_client"],
            voices_ids=characters_info.voices,
        )

    # Set up DB session factory and pre-warm ALL pool connections.
    # The pool has pool_size=5 — fire 5 concurrent pings so every slot is
    # established at startup instead of lazily on the first real query.
    db_engine = get_engine()
    db_session_factory = get_session_factory(db_engine)
    from sqlalchemy import text

    async def _ping():
        async with db_session_factory() as db:
            await db.execute(text("SELECT 1"))

    results = await asyncio.gather(*[_ping() for _ in range(5)], return_exceptions=True)
    failures = [r for r in results if isinstance(r, Exception)]
    if failures:
        print(f"⚠️  [DB] Warm-up partial — {5 - len(failures)}/5 connections established. First error: {failures[0]}")
    else:
        print("✅ [DB] Connection pool warmed up (5/5 connections)")

    # Load all FAQ embeddings into memory — searches become numpy dot products (< 1ms, no DB hit)
    faq_memory_cache = FAQMemoryCache()
    try:
        async with db_session_factory() as db:
            await faq_memory_cache.load(db)
    except Exception as e:
        print(f"⚠️  [FAQ CACHE] Failed to load (non-fatal — will fall back to DB search): {e}")

    # Load all history chunks into memory — RAG retrieval becomes a numpy search (< 1ms),
    # keeping the remote DB out of the request hot path entirely.
    history_memory_cache = HistoryMemoryCache()
    try:
        async with db_session_factory() as db:
            await history_memory_cache.load(db)
    except Exception as e:
        print(f"⚠️  [HISTORY CACHE] Failed to load (non-fatal — will fall back to DB search): {e}")

    connection_manager = ConnectionManager()
    pipeline = Pipeline(
        connection_manager=connection_manager,
        audio_preprocessor=audio_preprocessor,
        stt_service=stt_service,
        llm_service=llm_service,
        elevenlabs_service=tts_service,
        db_session_factory=db_session_factory,
        faq_memory_cache=faq_memory_cache,
        history_memory_cache=history_memory_cache,
        openai_client=models["openai_client"],
    )

    app.state.connection_manager = connection_manager
    app.state.pipeline = pipeline
    pipeline.start()
    mode = "LOCAL (from-scratch)" if config.USE_LOCAL_SERVICES else "API"
    print(f"🎉 [STARTUP] Done — running in {mode} mode\n_____________________________________________\n")  # separator for clearer logs

    yield

    print("\n_____________________________________________\n🛑 [SHUTDOWN] Shutting down...")


app = FastAPI(title="Mohandeskhana Voice Chat WebSocket API", lifespan=lifespan)
app.include_router(websocket_router)
app.include_router(faq_router)


@app.get("/")
def root():
    return {"message": "Mohandeskhana WebSocket Voice Chat API is running 🗣️"}


@app.get("/health")
def health():
    return {"status": "ok"}
