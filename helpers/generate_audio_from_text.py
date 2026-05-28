"""
generate_audio.py
-----------------
Generate a WAV file from user-supplied text using the exact same ElevenLabs
parameters configured in app/core/config.py.

Output file: <sanitized_input>.wav  (saved next to this script)

Usage:
    python generate_audio.py
    python generate_audio.py "Hello, how are you?"
"""

import io
import os
import re
import sys

import librosa
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from elevenlabs import ElevenLabs, VoiceSettings

load_dotenv()

# ── Same parameters as app/core/config.py ─────────────────────────────────────
ELEVENLABS_API_KEY   = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID             =os.getenv("ROGER_VOICE_ID") 
MODEL_ID             = "eleven_v3"
OUTPUT_FORMAT        = "mp3_44100_128"   # available on all tiers
SAMPLE_RATE          = 44100

VOICE_SETTINGS = VoiceSettings(
    stability=0.2,
    similarity_boost=0.85,
    style=0.75,
    use_speaker_boost=True,
)
# ──────────────────────────────────────────────────────────────────────────────


def sanitize(text: str, max_len: int = 40) -> str:
    """Turn arbitrary text into a safe filename stem."""
    slug = re.sub(r"[^\w\s-]", "", text.lower())
    slug = re.sub(r"[\s_-]+", "_", slug).strip("_")
    return slug[:max_len] or "output"


def generate(text: str) -> str:
    if not ELEVENLABS_API_KEY:
        raise EnvironmentError("ELEVENLABS_API_KEY is not set in .env")
    if not VOICE_ID:
        raise EnvironmentError("AHMAD_VOICE_ID is not set in .env")

    client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

    print(f"🔊 Generating audio for: \"{text}\"")
    print(f"   model={MODEL_ID}  voice={VOICE_ID}  format={OUTPUT_FORMAT}")

    mp3_stream = client.text_to_speech.convert(
        text=text,
        voice_id=VOICE_ID,
        model_id=MODEL_ID,
        output_format=OUTPUT_FORMAT,
        voice_settings=VOICE_SETTINGS,
    )

    mp3_bytes = b"".join(chunk for chunk in mp3_stream if chunk)

    # Decode MP3 → float32 numpy array at the target sample rate
    audio, sr = librosa.load(io.BytesIO(mp3_bytes), sr=SAMPLE_RATE, mono=True)

    filename = f"{sanitize(text)}.wav"
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
    os.makedirs(templates_dir, exist_ok=True)
    output_path = os.path.join(templates_dir, filename)
    sf.write(output_path, audio, samplerate=sr)

    print(f"✅ Saved → {output_path}  ({len(audio)} samples, {len(audio)/sr:.2f}s)")
    return output_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_text = " ".join(sys.argv[1:])
    else:
        user_text = input("Enter text to synthesize: ").strip()

    if not user_text:
        print("❌ No text provided.")
        sys.exit(1)

    generate(user_text)
