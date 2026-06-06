import io
import wave

import numpy as np

import app.core.config as config


class STTGroqWhisperService:

    def __init__(self, client=None):
        self.client = client
        self.language = config.SST_language
        self.model = config.groq_whisper_model
        self._warmup()

    def _warmup(self):
        print(f"⏳ [STT] Warming up Groq Whisper (model={self.model})...")
        silent = np.zeros(config.audio_preprocessing_sample_rate, dtype=np.float32)  # lahzet samt 3lashan el warm up 
        try:
            self.transcribe(silent)
        except Exception:
            pass  
        print(f"✅ [STT] Groq Whisper ready (model={self.model})")

    def _audio_array_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """Convert a float32 numpy array (16kHz mono) to raw WAV bytes for the API."""
        pcm = (audio * 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)   # 16-bit
            wf.setframerate(config.audio_preprocessing_sample_rate)
            wf.writeframes(pcm.tobytes())
        return buf.getvalue()

    def transcribe(self, audio: np.ndarray) -> str:
        wav_bytes = self._audio_array_to_wav_bytes(audio)
        transcription = self.client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes, "audio/wav"),
            model=self.model,
            language=self.language,
            response_format="text",
        )



        return transcription.strip() if isinstance(transcription, str) else transcription.text.strip()
