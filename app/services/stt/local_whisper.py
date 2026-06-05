import numpy as np
from faster_whisper import WhisperModel
import app.core.config as config


class STTWhisperService:
    def __init__(self, model=None):
        self.model = model
        self.SST_language = config.SST_language
        self.SST_vad_filter = config.SST_vad_filter
        self.SST_beam_size = config.SST_beam_size
        self._warmup()

    def _warmup(self):
      #Run a dummy transcription to trigger CTranslate2 model loading
        silent = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        segments, _ = self.model.transcribe(
            silent,
            language=self.SST_language,
            vad_filter=self.SST_vad_filter,
            beam_size=self.SST_beam_size,
        )
        list(segments)  # consume the generator to force full execution
        print("✅ [STT] Whisper warmed up")

    def transcribe(self, audio, language="en", vad_filter=True, beam_size=2):
        segments, info = self.model.transcribe(
            audio,
            language=self.SST_language,
            vad_filter=self.SST_vad_filter,
            beam_size=self.SST_beam_size
        )

        transcription = " ".join(segment.text for segment in segments)
        return transcription.strip()
