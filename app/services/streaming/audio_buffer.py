from typing import List, Optional


class AudioBufferService:
    def __init__(self):
        self.chunks: List[str] = []

    def add_chunk(self, audio_chunk: str) -> None:
        self.chunks.append(audio_chunk)

    def get_chunk_count(self) -> int:
        return len(self.chunks)

    def get_all_chunks(self) -> List[str]:
        return self.chunks.copy()

    def get_latest_chunk(self) -> Optional[str]:
        if not self.chunks:
            return None
        return self.chunks[-1]

    def get_last_n_chunks(self, n: int) -> List[str]:
        if n <= 0:
            return []
        return self.chunks[-n:].copy()

    def has_chunks(self) -> bool:
        return len(self.chunks) > 0

    def merge_chunks(self, separator: str = "") -> str:
        merged_audio = separator.join(self.chunks)
        print(f"🧩 [BUFFER MERGE CHUNKS] total_chunks={len(self.chunks)} | merged_length={len(merged_audio)}")
        return merged_audio

    def clear(self) -> None:
        cleared_count = len(self.chunks)
        self.chunks.clear()
        print(f"🧹 [BUFFER CLEARED] cleared_chunks={cleared_count}")

    def reset(self) -> None:
        self.clear()
        print("♻️ [BUFFER RESET] Buffer reset completed")

    def to_dict(self) -> dict:
        return {
            "chunk_count": len(self.chunks),
            "has_chunks": self.has_chunks(),
        }
