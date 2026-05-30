import os
from dotenv import load_dotenv
from elevenlabs import VoiceSettings
load_dotenv()






#preprocessing
audio_preprocessing_sample_rate = 16000
audio_noise_reduction_enabled = False   # disable noisereduce to save 50–200ms per batch



#SST 
stt_provider = "groq"            # "local" → faster-whisper on device | "groq" → Groq hosted Whisper API
whisper_model_size = "tiny.en"   # used only when stt_provider = "local"
whisper_device = "cpu"           # used only when stt_provider = "local"
whisper_compute_type = "int8"    # used only when stt_provider = "local"
groq_whisper_model = "whisper-large-v3-turbo"  # used only when stt_provider = "groq"
SST_language = "en"
SST_vad_filter = True
SST_beam_size = 2



#LLM 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
openAI_model_name="gpt-5-nano"
openai_max_tokens = 600 
openai_verifier_model_name = "gpt-4.1-nano"
openai_verifier_max_tokens = 600                    # enough for all 5 JSON fields with notes + corrected_answer when verification fails
VERIFIER_TIMEOUT = 10.0   # seconds — if OpenAI doesn't respond in time, pass through to TTS
groq_model_name = "llama-3.1-8b-instant"
groq_max_completion_tokens = 1024


#TTS 
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




#storage
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
RESPONSES_AUDIO_BUCKET = os.getenv("SUPABASE_RESPONSES_BUCKET", "response-audios")
QUESTIONS_AUDIO_BUCKET = os.getenv("SUPABASE_QUESTIONS_BUCKET", "question-audios")

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
MODERATION_ENABLED = True              # OpenAI Moderation API on questions and answers
ANACHRONISM_ENABLED = True             # regex check for future years / modern terms / URLs / emails

# Anachronism cutoff — any 4-digit year in the LLM answer that exceeds this
# year is flagged. Each character is actually checked against their OWN
# operation_year (see regex_checks.check_anachronism); this value is only the
# fallback used when a character has no year on file.
ANACHRONISM_DEFAULT_LATEST_YEAR = 1960


#functions 
def get_prompt_key_by_character_id(character_id):
    cid = character_id.lower()
    if cid.startswith("s"):
        return "mohandeskhana-student"
    elif cid.startswith("p"):
        return "mohandeskhana-professor"
