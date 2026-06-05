import io
from datetime import datetime

import soundfile as sf


from app.core import config




class AudioGenerationElevenLabsService:
    def __init__(self, voices_ids=None, client=None):
        self.client = client
        self.voices_ids = voices_ids
        self.model_id = config.ELEVENLABS_MODEL_ID
        self._warmup()

    def _warmup(self):
        print(f"⏳ [TTS] Warming up ElevenLabs (model={self.model_id}, {len(self.voices_ids)} voices)...")
        for character_id, voice_id in (self.voices_ids or {}).items():
            try:
                stream = self.client.text_to_speech.convert(
                    text="Hello.",
                    voice_id=voice_id,
                    model_id=self.model_id,
                    output_format="mp3_44100_128",
                )
                for _ in stream:
                    break  # first chunk is enough — connection and voice are warm
            except Exception:
                pass
        print(f"✅ [TTS] ElevenLabs ready (model={self.model_id})")

    def stream_audio(self, text, character_id, debug_output_path=None):
        """Stream MP3 chunks. Used for debug/save paths."""
        try:
            audio_stream = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voices_ids[str(character_id).lower()],
                model_id=self.model_id,
                output_format="mp3_44100_128",
                voice_settings=config.VOICE_SETTINGS,
            )

            output_file = open(debug_output_path, "wb") if debug_output_path else None
            try:
                for chunk in audio_stream:
                    if not chunk:
                        continue
                    if output_file:
                        output_file.write(chunk)
                    yield chunk
            finally:
                if output_file:
                    output_file.close()

        except Exception as e:
            print(f"[TTS STREAM ERROR] {repr(e)}")
            raise

    def stream_audio_pcm(self, text, character_id):

        try:
            audio_stream = self.client.text_to_speech.convert(
                text=text,
                voice_id=self.voices_ids[str(character_id).lower()],
                model_id=self.model_id,
                output_format="pcm_44100",   # raw signed 16-bit PCM, no container
                voice_settings=config.VOICE_SETTINGS,
            )
            for chunk in audio_stream:
                if chunk:
                    yield chunk
        except Exception as e:
            print(f"[TTS STREAM ERROR] {repr(e)}")
            raise
