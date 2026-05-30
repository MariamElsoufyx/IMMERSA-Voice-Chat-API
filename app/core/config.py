import os
from dotenv import load_dotenv
from elevenlabs import VoiceSettings
load_dotenv()






# ─── Master switch ──────────────────────────────────────────────────────────
# Flip ONE flag to route every external API (STT / LLM / TTS / moderation /
# LLM judge / audio storage) to its from-scratch local implementation. All the
# per-service `*_provider` values below are derived from this single boolean,
# so individual overrides aren't needed for the common case.
USE_LOCAL_SERVICES = True
# ────────────────────────────────────────────────────────────────────────────


#preprocessing
preprocessing_provider = "local" if USE_LOCAL_SERVICES else "library"  # "local" → pure-numpy LocalAudioPreprocessor | "library" → scipy/librosa/noisereduce AudioPreprocessor
audio_preprocessing_sample_rate = 16000
audio_noise_reduction_enabled = False   # disable noisereduce to save 50–200ms per batch



#SST
stt_provider = "local" if USE_LOCAL_SERVICES else "groq"  # "local" → faster-whisper on device | "groq" → Groq hosted Whisper API
whisper_model_size = "tiny.en"   # used only when stt_provider = "local"
whisper_device = "cpu"           # used only when stt_provider = "local"
whisper_compute_type = "int8"    # used only when stt_provider = "local"
groq_whisper_model = "whisper-large-v3-turbo"  # used only when stt_provider = "groq"
SST_language = "en"
SST_vad_filter = True
SST_beam_size = 2



#LLM
llm_provider = "local" if USE_LOCAL_SERVICES else "groq"  # "local" → LLMLocalService | "groq" → Groq | "openai" → OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
openAI_model_name="gpt-5-nano"
openai_max_tokens = 600
openai_verifier_model_name = "gpt-4.1-nano"
openai_verifier_max_tokens = 600                    # enough for all 5 JSON fields with notes + corrected_answer when verification fails
VERIFIER_TIMEOUT = 10.0   # seconds — if OpenAI doesn't respond in time, pass through to TTS
groq_model_name = "llama-3.1-8b-instant"
groq_max_completion_tokens = 1024

#Local LLM (used only when llm_provider = "local") — see app/services/llm/local_llm_service.py
local_llm_backend = "ollama"                # "ollama" → HTTP daemon | "llamacpp" → in-process
local_llm_model_name = "llama3.1:8b"        # ollama model tag, or llama.cpp model identifier
local_llm_model_path = ""                   # only used when backend = "llamacpp" — path to .gguf
local_llm_n_ctx = 4096
local_llm_n_threads = 8
local_llm_max_tokens = 1024
local_llm_temperature = 0.7
local_llm_timeout_s = 60.0
local_llm_ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
local_llm_judge_model_name = "llama3.1:8b"  # separate so the judge can be smaller/faster than the reply model


#TTS
tts_provider = "local" if USE_LOCAL_SERVICES else "elevenlabs"  # "local" → Piper | "elevenlabs" → ElevenLabs cloud API
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_MODEL_ID = "eleven_v3"  # lowest-latency model — sentence pipelining keeps quality high
VOICE_STABILITY = 0.2                     # low = more emotional range
VOICE_SIMILARITY_BOOST = 0.85             # high = closer to target voice, but less expressive
VOICE_STYLE = 0.75                         # push expressiveness
USER_SPEAKER_BOOST = True
VOICE_SETTINGS = VoiceSettings(

    stability=VOICE_STABILITY,
    similarity_boost=VOICE_SIMILARITY_BOOST,
    style=VOICE_STYLE,
    use_speaker_boost=USER_SPEAKER_BOOST,
)
TTS_FIRST_CHUNK_TIMEOUT = 7.0

#Local TTS (used only when tts_provider = "local") — see app/services/tts/local_tts_service.py
# One Piper .onnx voice file per character. Defaults are placeholder paths —
# point them at real models before flipping tts_provider to "local".
local_tts_voices = {
    "s1": os.getenv("PIPER_VOICE_S1", "data/piper_voices/s1.onnx"),
    "s2": os.getenv("PIPER_VOICE_S2", "data/piper_voices/s2.onnx"),
    "p1": os.getenv("PIPER_VOICE_P1", "data/piper_voices/p1.onnx"),
}




#storage
storage_provider = "local" if USE_LOCAL_SERVICES else "supabase"  # "local" → LocalStorageService | "supabase" → Supabase Storage
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
RESPONSES_AUDIO_BUCKET = os.getenv("SUPABASE_RESPONSES_BUCKET", "response-audios")
QUESTIONS_AUDIO_BUCKET = os.getenv("SUPABASE_QUESTIONS_BUCKET", "question-audios")

#Local storage (used only when storage_provider = "local") — see app/services/storage/local_storage_service.py
local_storage_root = os.getenv("LOCAL_STORAGE_ROOT", "data/storage")
local_storage_base_url = os.getenv("LOCAL_STORAGE_BASE_URL", "http://localhost:8000")

#db
FAQ_SIMILARITY_THRESHOLD = 0.8
FAQ_LOOKUP_TIMEOUT = 2.0   # seconds — if DB doesn't respond in time, skip FAQ and fall through to LLM

#conversation memory (per-WS-session multi-turn context passed to the LLM)
CONVERSATION_HISTORY_MAX_TURNS = 6  # keep last N (question, answer) pairs; older turns are dropped

#RAG (history retrieval — used on FAQ miss to ground the LLM)
HISTORY_TOP_K = 6                  # number of history chunks to inject into the prompt
HISTORY_SIMILARITY_THRESHOLD = 0.4 # looser than FAQ — we want context even on a partial match
HISTORY_LOOKUP_TIMEOUT = 2.0       # seconds — skip retrieval and answer without grounding if slow (in-memory search is ~1ms, so this is just a safety net)


#verification
moderation_provider = "local" if USE_LOCAL_SERVICES else "openai"  # "local" → toxic-bert | "openai" → omni-moderation-latest
llm_judge_provider = "local" if USE_LOCAL_SERVICES else "openai"   # "local" → LLMLocalService judge | "openai" → gpt-4.1-nano
MODERATION_ENABLED = True              # OpenAI Moderation API on questions and answers
ANACHRONISM_ENABLED = True             # regex check for future years / modern terms / URLs / emails

#Local moderation (used only when moderation_provider = "local") — see local_model_checks.py
local_moderation_model = "unitary/toxic-bert"   # HuggingFace text-classification model
local_moderation_threshold = 0.5                # score ≥ threshold flags the label

# Anachronism cutoff — any 4-digit year in the LLM answer that exceeds this
# year is flagged.
ANACHRONISM_DEFAULT_LATEST_YEAR = 1918


#functions 
def get_prompt_key_by_character_id(character_id):
    cid = character_id.lower()
    if cid.startswith("s"):
        return "mohandeskhana-student"
    elif cid.startswith("p"):
        return "mohandeskhana-professor"
