import base64
import json
import struct
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.streaming.event_protocol import (
    build_ack_event,
    build_connection_established_event,
    build_error_event,
)

router = APIRouter()


def _extract_wav_header(wav_bytes: bytes) -> bytes:
    """Return the full WAV header (up to and including the 'data' size field).

    Standard PCM WAV has a 44-byte header, but files with extra chunks (LIST,
    JUNK, bext, etc.) push 'data' further in. Slicing to a fixed 44 bytes in
    those cases truncates the header mid-chunk, producing bytes that soundfile
    can't recognise as WAV (LibsndfileError 61).
    """
    idx = wav_bytes.find(b"data")
    if idx == -1 or idx + 8 > len(wav_bytes):
        # Fallback: no 'data' marker found — return whatever we have.
        return wav_bytes
    return wav_bytes[: idx + 8]  # 'data' (4 bytes) + data-chunk size (4 bytes)


def _wrap_pcm16_as_wav(raw_bytes: bytes, sample_rate: int, channels: int = 1) -> bytes:
    """Wrap raw PCM16 LE bytes in a valid WAV container so soundfile can read them."""
    bits_per_sample = 16
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(raw_bytes),
        b"WAVE",
        b"fmt ",
        16,
        1,  # PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        len(raw_bytes),
    )
    return header + raw_bytes


@router.websocket("/ws/voice-chat")
async def websocket_voice_chat(websocket: WebSocket):
    session_id = str(uuid.uuid4())
    manager = websocket.app.state.connection_manager
    pipeline = websocket.app.state.pipeline

    await websocket.accept()
    await manager.connect(session_id, websocket)
    await manager.send_json(session_id, build_connection_established_event(session_id))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                await manager.send_json(session_id, build_error_event("Invalid JSON"))
                continue

            msg_type = data.get("type")

            if msg_type == "start_session":
                session = manager.create_session(session_id)
                session.start_session(
                    character_id=str(data.get("character_id")).lower(),
                    sample_rate=data.get("sample_rate", 16000),
                    audio_format=data.get("audio_format", "wav"),
                )
                await manager.send_json(session_id, build_ack_event(
                    event="start_session",
                    message="Session started successfully",
                    session_id=session_id,
                    character_id=(session.character_id).lower(),
                    sample_rate=session.sample_rate,
                    audio_format=session.audio_format,
                    state=session.state,
                ))

            elif msg_type == "audio_chunk":
                session = manager.get_session(session_id)
                if not session:
                    await manager.send_json(session_id, build_error_event("No active session. Send start_session first."))
                    continue
                audio_data = data.get("audio")
                if not audio_data:
                    await manager.send_json(session_id, build_error_event("Missing audio data"))
                    continue
                session.add_audio_chunk(audio_data)
                total = session.get_audio_chunk_count()

                # For WAV format, cache the header from the first chunk so later
                # headerless batches can be reassembled into valid WAV data.
                if total == 1 and session.audio_format != "pcm16_base64_chunks":
                    session.wav_header = _extract_wav_header(base64.b64decode(audio_data))

                await manager.send_json(session_id, build_ack_event(
                    event="audio_chunk",
                    message="Audio chunk received",
                    chunk_index=data.get("chunk_index"),
                    total_chunks=total,
                ))

                # Trigger rolling STT every 5 chunks
                if total % 5 == 0:
                    batch = session.audio_buffer.get_last_n_chunks(5)
                    raw = b"".join(base64.b64decode(c) for c in batch)
                    if session.audio_format == "pcm16_base64_chunks":
                        audio_bytes = _wrap_pcm16_as_wav(raw, session.sample_rate)
                    elif total > 5:
                        audio_bytes = session.wav_header + raw
                    else:
                        audio_bytes = raw
                    session.processed_chunk_count = total
                    await pipeline.enqueue(session_id, audio_bytes, is_final=False)

            elif msg_type == "end_of_utterance":
                session = manager.get_session(session_id)
                if not session or not session.audio_buffer.has_chunks():
                    await manager.send_json(session_id, build_error_event("No audio to process"))
                    continue
                await manager.send_json(session_id, build_ack_event(
                    event="end_of_utterance",
                    message="Processing started",
                ))

                remaining = session.audio_buffer.get_all_chunks()[session.processed_chunk_count:]
                if remaining:
                    raw = b"".join(base64.b64decode(c) for c in remaining)
                    if session.audio_format == "pcm16_base64_chunks":
                        audio_bytes = _wrap_pcm16_as_wav(raw, session.sample_rate)
                    elif session.processed_chunk_count > 0:
                        audio_bytes = session.wav_header + raw
                    else:
                        audio_bytes = raw
                    await pipeline.enqueue(session_id, audio_bytes, is_final=True)
                else:
                    # Chunks count was exactly divisible by 5 — all already processed
                    await pipeline.enqueue_finalize(session_id)

            elif msg_type == "close_session":
                session = manager.get_session(session_id)
                if session:
                    session.close()
                await manager.send_json(session_id, build_ack_event(
                    event="close_session",
                    message="Session closed successfully",
                ))
                await websocket.close()
                break

            else:
                await manager.send_json(session_id, build_error_event(f"Unknown message type: {msg_type}"))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WS ERROR] session_id={session_id} | {e}")
        try:
            await manager.send_json(session_id, build_error_event(str(e)))
        except Exception:
            pass
    finally:
        manager.disconnect(session_id)
