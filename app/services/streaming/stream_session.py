import base64
from dataclasses import dataclass, field
from typing import Optional
import time

import app.core.config as config
from app.services.streaming.audio_buffer import AudioBufferService


@dataclass
class StreamSession:
    session_id: str
    character_id: Optional[str] = None
    sample_rate: int = config.audio_preprocessing_sample_rate
    audio_format: str = "wav"
    state: str = "CONNECTED"
    dead_time_start: Optional[float] = None
    dead_time_end: Optional[float] = None
    audio_buffer: AudioBufferService = field(default_factory=AudioBufferService)
    final_transcript: str = ""
    reply_text: str = ""
    emotion: str = ""
    partial_transcripts: list = field(default_factory=list)
    processed_chunk_count: int = 0
    wav_header: bytes = field(default_factory=bytes)
    # Multi-turn conversation memory for THIS websocket connection.
    # List of {"role": "user"|"assistant", "content": str}, oldest first.
    # Reset only when the connection ends (see close()), NOT between utterances.
    conversation_history: list = field(default_factory=list)

    @property
    def sid(self) -> str:
        """Short session id for readable logs."""
        return self.session_id[:8]

    def append_partial_transcript(self, text: str) -> None:
        if text.strip():
            self.partial_transcripts.append(text.strip())
            print(f"[STT PARTIAL] sid={self.sid} | partial={text[:60]}")

    def get_combined_transcript(self) -> str:
        return " ".join(self.partial_transcripts).strip()

    def touch(self) -> None:
        self.updated_at = time.time()

    def start_session(
        self,
        character_id: str,
        sample_rate: int = config.audio_preprocessing_sample_rate,
        audio_format: str = "wav",
    ) -> None:
        self.character_id = character_id
        self.sample_rate = sample_rate
        self.audio_format = audio_format
        self.state = "LISTENING"
        self.touch()

        print(
            f"🚀 [SESSION STARTED] sid={self.sid} | "
            f"character={self.character_id} | state={self.state}"
        )

    def add_audio_chunk(self, audio_chunk: str) -> None:
        self.audio_buffer.add_chunk(audio_chunk)
        self.dead_time_start = time.time()
        self.touch()

    def set_final_transcript(self, text: str) -> None:
        self.final_transcript = text
        self.state = "FINALIZING_TRANSCRIPT"
        self.touch()

        print(
            f"📝 [FINAL TRANSCRIPT SET] sid={self.sid} | "
            f"state={self.state} | text={self.final_transcript}"
        )

    def set_reply_text(self, parsed: dict) -> None:
        self.reply_text = parsed["answer"]
        emotion_val = parsed.get("emotion")
        self.emotion = emotion_val.lower() if emotion_val else None
        self.state = "GENERATING_REPLY"
        self.touch()

    def set_state(self, new_state: str) -> None:
        old_state = self.state
        self.state = new_state
        self.touch()

        print(
            f"🔄 [STATE CHANGED] sid={self.sid} | "
            f"from={old_state} | to={new_state}"
        )

    def append_turn(self, question: str, answer: str) -> None:
        """Record a completed (question, answer) exchange into conversation memory,
        trimming to the last config.CONVERSATION_HISTORY_MAX_TURNS pairs."""
        question = (question or "").strip()
        answer = (answer or "").strip()
        if not question or not answer:
            return
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})

        # Trim to the last N pairs (2 messages per turn).
        max_messages = config.CONVERSATION_HISTORY_MAX_TURNS * 2
        if len(self.conversation_history) > max_messages:
            self.conversation_history = self.conversation_history[-max_messages:]
        self.touch()

        print(
            f"💬 [HISTORY APPENDED] sid={self.sid} | "
            f"turns={len(self.conversation_history) // 2}"
        )

    def get_history(self) -> list:
        """Return a copy of the conversation history as chat messages (oldest first)."""
        return list(self.conversation_history)

    def clear_history(self) -> None:
        cleared = len(self.conversation_history) // 2
        self.conversation_history = []
        self.touch()
        print(
            f"🧽 [HISTORY CLEARED] sid={self.sid} | cleared_turns={cleared}"
        )

    def get_audio_chunk_count(self) -> int:
        return self.audio_buffer.get_chunk_count()

    def get_all_audio_chunks(self):
        return self.audio_buffer.get_all_chunks()

    def clear_audio_buffer(self) -> None:
        chunks_before_clear = self.audio_buffer.get_chunk_count()
        self.audio_buffer.clear()
        self.touch()

        print(
            f"🧹 [AUDIO BUFFER CLEARED] sid={self.sid} | "
            f"cleared_chunks={chunks_before_clear}"
        )

    def reset_for_next_utterance(self) -> None:
        chunks_before_reset = self.audio_buffer.get_chunk_count()

        self.audio_buffer.reset()
        self.final_transcript = ""
        self.reply_text = ""
        self.emotion = None
        self.partial_transcripts = []
        self.processed_chunk_count = 0
        self.state = "LISTENING"
        self.dead_time_start = None
        self.dead_time_end = None
        self.touch()

        print(
            f"♻️ [SESSION RESET FOR NEXT UTTERANCE] sid={self.sid} | "
            f"cleared_chunks={chunks_before_reset} | state={self.state}"
        )

    def close(self) -> None:
        self.state = "CLOSED"
        # Conversation memory lives only for the duration of the connection.
        self.conversation_history = []
        self.touch()

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "character_id": self.character_id,
            "sample_rate": self.sample_rate,
            "audio_format": self.audio_format,
            "state": self.state,
            "audio_chunk_count": self.audio_buffer.get_chunk_count(),
            "final_transcript": self.final_transcript,
            "reply_text": self.reply_text,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "dead_time_start": self.dead_time_start,
            "dead_time_end": self.dead_time_end,
        }
