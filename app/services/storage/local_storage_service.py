"""Local filesystem audio storage — drop-in for the Supabase Storage uploads
inlined in pipeline.py (_upload_question_audio, _combine_and_upload_audio).

Writes WAV files to data/storage/{bucket}/{filename} and returns a URL that
FastAPI can serve via StaticFiles. The pipeline only cares that a string URL
comes back, so dropping this in is one find-and-replace away.

The on-disk layout intentionally mirrors Supabase's `/storage/v1/object/public/
{bucket}/{filename}` path so existing audio_url values in past_questions stay
valid if you switch back later.
"""
import io
import os
import uuid

import numpy as np
import soundfile as sf

import app.core.config as config


class LocalStorageService:
    def __init__(self, root_dir: str | None = None, base_url: str | None = None):
        """root_dir: where files land on disk. base_url: prefix returned in URLs
        (typically your FastAPI host + a mounted StaticFiles path)."""
        self.root_dir = root_dir or config.local_storage_root
        self.base_url = (base_url or config.local_storage_base_url).rstrip("/")
        os.makedirs(self.root_dir, exist_ok=True)

    # --- internal helpers ---

    def _bucket_dir(self, bucket: str) -> str:
        path = os.path.join(self.root_dir, bucket)
        os.makedirs(path, exist_ok=True)
        return path

    def _public_url(self, bucket: str, filename: str) -> str:
        # Mirrors Supabase's /storage/v1/object/public/{bucket}/{filename} layout.
        return f"{self.base_url}/storage/v1/object/public/{bucket}/{filename}"

    def _write(self, bucket: str, audio_bytes: bytes, character_id: str, suffix: str = "") -> str | None:
        if not audio_bytes:
            return None
        try:
            filename = f"{character_id.lower()}_{uuid.uuid4().hex[:8]}{suffix}.wav"
            full_path = os.path.join(self._bucket_dir(bucket), filename)
            with open(full_path, "wb") as f:
                f.write(audio_bytes)
            return self._public_url(bucket, filename)
        except Exception as e:
            print(f"[LOCAL STORAGE ERROR] {bucket} write failed: {e}")
            return None

    # --- public API: matches the inline Supabase calls in pipeline.py ---

    async def upload_question_audio(self, audio_bytes: bytes, character_id: str) -> str | None:
        """Mirrors Pipeline._upload_question_audio()."""
        return self._write(
            config.QUESTIONS_AUDIO_BUCKET, audio_bytes, character_id, suffix="_q"
        )

    async def combine_and_upload_audio(self, wav_chunks: list[bytes], character_id: str) -> str | None:
        """Mirrors Pipeline._combine_and_upload_audio() — concat WAV chunks, write one file."""
        if not wav_chunks:
            return None
        try:
            all_audio = []
            sr = None
            for chunk in wav_chunks:
                audio, file_sr = sf.read(io.BytesIO(chunk), dtype="float32", always_2d=False)
                if len(audio) > 0:
                    all_audio.append(audio)
                    sr = sr or file_sr
            if not all_audio or sr is None:
                return None

            combined = np.concatenate(all_audio)
            buf = io.BytesIO()
            sf.write(buf, combined, sr, format="WAV", subtype="PCM_16")
            return self._write(config.RESPONSES_AUDIO_BUCKET, buf.getvalue(), character_id)
        except Exception as e:
            print(f"[LOCAL STORAGE ERROR] combine_and_upload failed: {e}")
            return None
