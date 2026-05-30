"""Local TTS service — drop-in for AudioGenerationElevenLabsService.

Exposes the same stream_audio() / stream_audio_pcm() generators the pipeline
already consumes, so swapping providers is a one-line change in main.py.

Backend: Piper TTS (https://github.com/rhasspy/piper).
  - Runs entirely on CPU, ~real-time on a laptop.
  - One .onnx voice model per character, mapped via config.local_tts_voices.
  - Emits raw 22050 Hz mono PCM16 → we resample to 44100 Hz to match the
    ElevenLabs format the rest of the pipeline assumes.

The pipeline's _start_sentence_tts() loads whatever stream_audio() yields with
librosa, so emitting raw PCM wrapped in a tiny WAV header works the same as
ElevenLabs' MP3 chunks. stream_audio_pcm() emits naked PCM16 at 44100 Hz to
match the ElevenLabs PCM path byte-for-byte.
"""
import io
import struct
import wave

import numpy as np

import app.core.config as config


class AudioGenerationLocalService:
    PIPER_SAMPLE_RATE = 22050  # Piper voices output 22.05 kHz mono
    TARGET_SAMPLE_RATE = 44100  # match ElevenLabs pcm_44100

    def __init__(self, voices_paths: dict | None = None):
        """voices_paths: {character_id: "/abs/path/to/voice.onnx"}.
        Falls back to config.local_tts_voices if not provided.

        Missing .onnx (or its required .onnx.json sidecar) is logged but does
        NOT block startup — so you can wire one voice at a time. Any character
        without a loaded voice will raise at synth time."""
        import os
        from piper import PiperVoice  # imported lazily so the dep is optional

        self.voices_paths = voices_paths or config.local_tts_voices
        self.voices: dict[str, "PiperVoice"] = {}
        missing: list[tuple[str, str, str]] = []  # (cid, path, reason)
        for cid, path in (self.voices_paths or {}).items():
            if not os.path.exists(path):
                missing.append((cid, path, "model file missing"))
                continue
            if not os.path.exists(path + ".json"):
                missing.append((cid, path, "config sidecar missing (need <model>.onnx.json)"))
                continue
            try:
                self.voices[str(cid).lower()] = PiperVoice.load(path)
            except Exception as e:
                missing.append((cid, path, f"load failed: {e}"))

        if missing:
            print("⚠️  [TTS] Some Piper voices could not be loaded:")
            for cid, path, reason in missing:
                print(f"     - {cid}: {reason} ({path})")
            print("     Download voices: https://github.com/rhasspy/piper/blob/master/VOICES.md")
            print("     Each voice = TWO files: <name>.onnx AND <name>.onnx.json — place both side by side.")
            print("     Example (PowerShell):")
            print('       Invoke-WebRequest "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx" -OutFile "data\\piper_voices\\s1.onnx"')
            print('       Invoke-WebRequest "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json" -OutFile "data\\piper_voices\\s1.onnx.json"')

        self._warmup()

    def _warmup(self):
        """Synthesize a one-word clip per voice so the ONNX session compiles."""
        print(f"⏳ [TTS] Warming up local Piper ({len(self.voices)} voice(s))...")
        for cid, voice in self.voices.items():
            try:
                buf = io.BytesIO()
                with wave.open(buf, "wb") as wf:
                    voice.synthesize("Hello.", wf)
            except Exception:
                pass
        print(f"✅ [TTS] Local Piper ready ({list(self.voices)})")

    # --- internal: synth one sentence → raw PCM16 at TARGET_SAMPLE_RATE ---

    def _synth_pcm(self, text: str, character_id: str) -> bytes:
        voice = self.voices[str(character_id).lower()]
        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            voice.synthesize(text, wf)
        wav_buf.seek(0)

        with wave.open(wav_buf, "rb") as wf:
            src_sr = wf.getframerate()
            n_frames = wf.getnframes()
            pcm = wf.readframes(n_frames)

        # Resample to TARGET_SAMPLE_RATE if Piper's native rate differs.
        if src_sr != self.TARGET_SAMPLE_RATE:
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            ratio = self.TARGET_SAMPLE_RATE / src_sr
            new_len = int(round(len(audio) * ratio))
            x_src = np.linspace(0, 1, len(audio), endpoint=False)
            x_dst = np.linspace(0, 1, new_len, endpoint=False)
            resampled = np.interp(x_dst, x_src, audio)
            pcm = (np.clip(resampled, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

        return pcm

    @staticmethod
    def _wrap_pcm_as_wav(pcm: bytes, sample_rate: int = 44100, channels: int = 1) -> bytes:
        bits = 16
        data_len = len(pcm)
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + data_len, b"WAVE",
            b"fmt ", 16, 1, channels, sample_rate,
            sample_rate * channels * bits // 8,
            channels * bits // 8, bits,
            b"data", data_len,
        )
        return header + pcm

    # --- public API: matches AudioGenerationElevenLabsService ---

    def stream_audio(self, text: str, character_id: str, debug_output_path: str | None = None):
        """Yield WAV-wrapped chunks (one per sentence). The pipeline decodes with librosa."""
        try:
            pcm = self._synth_pcm(text, character_id)
            if not pcm:
                return
            wav_bytes = self._wrap_pcm_as_wav(pcm, sample_rate=self.TARGET_SAMPLE_RATE)
            if debug_output_path:
                with open(debug_output_path, "wb") as f:
                    f.write(wav_bytes)
            # Chunk the WAV into ~32 KB pieces so it feels streamed to the consumer.
            CHUNK = 32 * 1024
            for i in range(0, len(wav_bytes), CHUNK):
                yield wav_bytes[i:i + CHUNK]
        except Exception as e:
            print(f"[LOCAL TTS STREAM ERROR] {repr(e)}")
            raise

    def stream_audio_pcm(self, text: str, character_id: str):
        """Yield raw PCM16 LE at 44100 Hz — matches ElevenLabs pcm_44100 byte-for-byte."""
        try:
            pcm = self._synth_pcm(text, character_id)
            if not pcm:
                return
            CHUNK = 16 * 1024
            for i in range(0, len(pcm), CHUNK):
                yield pcm[i:i + CHUNK]
        except Exception as e:
            print(f"[LOCAL TTS STREAM ERROR] {repr(e)}")
            raise
