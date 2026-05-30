import asyncio
import base64
import io
import json
import os
import queue
import tempfile
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf
import websockets


WS_URL = "wss://immersa-api-voice-chat.run.place/ws/voice-chat"
#WS_URL = "ws://127.0.0.1:8000/ws/voice-chat"

TEST_FILES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_files")

CHARACTER_ID = "s1"
SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"
CHUNK_DURATION = 0.5  # seconds
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

AUDIO_FORMAT = "pcm16_base64_chunks"
#AUDIO_FORMAT = "wav_base64_chunks"

audio_queue = queue.Queue()
stop_recording = threading.Event()
server_processing_done = asyncio.Event()

# Latency tracking: set when end_of_utterance is sent, read on first tts_audio_chunk
timing = {"end_of_utterance_sent_at": None, "first_chunk_received_at": None}


def mic_callback(indata, frames, time, status):
    if status:
        print("MIC STATUS:", status)

    audio_queue.put(indata.copy())


def wait_for_enter_to_stop():
    input("\n🎤 Recording started... press ENTER to stop\n")
    stop_recording.set()


def encode_chunk(chunk):
    """
    chunk: numpy array shape (samples, channels)
    returns: base64 string
    """
    if AUDIO_FORMAT == "pcm16_base64_chunks":
        raw_bytes = chunk.tobytes()
        return base64.b64encode(raw_bytes).decode("utf-8")

    elif AUDIO_FORMAT == "wav_base64_chunks":
        buffer = io.BytesIO()
        sf.write(buffer, chunk, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        wav_bytes = buffer.getvalue()
        return base64.b64encode(wav_bytes).decode("utf-8")

    else:
        raise ValueError(f"Unsupported AUDIO_FORMAT: {AUDIO_FORMAT}")


async def send_audio_chunk(websocket, chunk, chunk_index):
    chunk_b64 = encode_chunk(chunk)

    await websocket.send(json.dumps({
        "type": "audio_chunk",
        "chunk_index": chunk_index,
        "audio": chunk_b64
    }))
    print(f"SENT: audio_chunk {chunk_index}")


def pick_test_file() -> str:
    """List audio files in test_files/ and let the user choose one."""
    os.makedirs(TEST_FILES_DIR, exist_ok=True)
    files = sorted(
        f for f in os.listdir(TEST_FILES_DIR)
        if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))
    )
    if not files:
        raise FileNotFoundError(f"No audio files found in {TEST_FILES_DIR}")

    print(f"\nAudio files in {TEST_FILES_DIR}:")
    for i, name in enumerate(files, 1):
        print(f"  [{i}] {name}")

    while True:
        choice = input("Select file number: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(files):
            return os.path.join(TEST_FILES_DIR, files[int(choice) - 1])
        print(f"Please enter a number between 1 and {len(files)}.")


async def sender_file(websocket, file_path):
    """Read an audio file, convert to PCM16 chunks, and stream to the server."""
    audio, sr = sf.read(file_path, dtype="int16", always_2d=True)

    if sr != SAMPLE_RATE:
        import librosa
        audio_float = audio[:, 0].astype(np.float32) / 32768.0
        resampled = librosa.resample(audio_float, orig_sr=sr, target_sr=SAMPLE_RATE)
        audio = (resampled * 32768).astype(np.int16).reshape(-1, 1)

    total_chunks = (len(audio) + CHUNK_SAMPLES - 1) // CHUNK_SAMPLES
    print(f"📁 Sending {total_chunks} chunk(s) from file…")

    for i in range(0, len(audio), CHUNK_SAMPLES):
        chunk = audio[i:i + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            pad = np.zeros((CHUNK_SAMPLES - len(chunk), CHANNELS), dtype=np.int16)
            chunk = np.vstack([chunk, pad])
        await send_audio_chunk(websocket, chunk, i // CHUNK_SAMPLES)

    timing["end_of_utterance_sent_at"] = time.perf_counter()
    await websocket.send(json.dumps({"type": "end_of_utterance"}))
    print("SENT: end_of_utterance")


def text_to_chunks(text):
    """Convert text to PCM16 audio chunks via pyttsx3."""
    try:
        import pyttsx3
    except ImportError:
        raise ImportError("pyttsx3 is required for text mode: pip install pyttsx3")

    engine = pyttsx3.init()
    engine.setProperty("rate", 150)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    try:
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()

        audio, sr = sf.read(tmp_path, dtype="int16", always_2d=True)
    finally:
        os.unlink(tmp_path)

    if sr != SAMPLE_RATE:
        import librosa
        audio_float = audio[:, 0].astype(np.float32) / 32768.0
        resampled = librosa.resample(audio_float, orig_sr=sr, target_sr=SAMPLE_RATE)
        audio = (resampled * 32768).astype(np.int16).reshape(-1, 1)

    chunks = []
    for i in range(0, len(audio), CHUNK_SAMPLES):
        chunk = audio[i:i + CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            pad = np.zeros((CHUNK_SAMPLES - len(chunk), CHANNELS), dtype=np.int16)
            chunk = np.vstack([chunk, pad])
        chunks.append(chunk)

    return chunks


async def sender_text(websocket, text):
    chunks = text_to_chunks(text)
    print(f"📝 Converted text to {len(chunks)} audio chunk(s)")

    for i, chunk in enumerate(chunks):
        await send_audio_chunk(websocket, chunk, i)

    timing["end_of_utterance_sent_at"] = time.perf_counter()
    await websocket.send(json.dumps({"type": "end_of_utterance"}))
    print("SENT: end_of_utterance")


async def sender(websocket):
    chunk_index = 0

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=CHUNK_SAMPLES,
        callback=mic_callback
    )

    try:
        with stream:
            print("✅ Microphone stream ready")

            while not stop_recording.is_set() or not audio_queue.empty():
                try:
                    chunk = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

                await send_audio_chunk(websocket, chunk, chunk_index)
                chunk_index += 1
                await asyncio.sleep(0)

        print("🛑 Microphone stream stopped")

        timing["end_of_utterance_sent_at"] = time.perf_counter()
        await websocket.send(json.dumps({
            "type": "end_of_utterance"
        }))
        print("SENT: end_of_utterance")

    except websockets.ConnectionClosed as e:
        print(f"⚠️  Connection closed while sending (code={e.code}): {e.reason or 'no reason given'}")
        stop_recording.set()
        server_processing_done.set()


async def receiver(websocket):
    playback_stream = None
    playback_sample_rate = None
    playback_channels = None

    try:
        while True:
            msg = await websocket.recv()
            data = json.loads(msg)
            msg_type = data.get("type")

            if msg_type == "ack":
                print("RECV: ack →", data)

            elif msg_type == "partial_transcript":
                print("RECV: partial_transcript →", data.get("text"))

            elif msg_type == "final_transcript":
                print("RECV: final_transcript →", data.get("text"))

            elif msg_type == "llm_token":
                token = data.get("token", "")
                print(token, end="", flush=True)

            elif msg_type == "reply_text_done":
                print(f"\nRECV: reply_text_done | emotion={data.get('emotion')}")

            elif msg_type == "tts_audio_chunk":
                raw = base64.b64decode(data["audio"])
                audio, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)

                if audio.size == 0:
                    continue

                channels = audio.shape[1]

                # Measure latency from end_of_utterance → first audio chunk
                if timing["first_chunk_received_at"] is None and timing["end_of_utterance_sent_at"] is not None:
                    timing["first_chunk_received_at"] = time.perf_counter()
                    latency = timing["first_chunk_received_at"] - timing["end_of_utterance_sent_at"]
                    print(f"\n⏱  End-of-utterance → first audio chunk: {latency:.3f}s\n")

                print(
                    f"RECV: tts_audio_chunk index={data.get('chunk_index')} "
                    f"| {len(raw)} bytes | sr={sr} | channels={channels}"
                )

                if (
                    playback_stream is None
                    or playback_sample_rate != sr
                    or playback_channels != channels
                ):
                    if playback_stream is not None:
                        playback_stream.stop()
                        playback_stream.close()

                    playback_sample_rate = sr
                    playback_channels = channels

                    playback_stream = sd.OutputStream(
                        samplerate=sr,
                        channels=channels,
                        dtype="float32"
                    )
                    playback_stream.start()

                playback_stream.write(audio)

            elif msg_type == "tts_done":
                print("RECV: tts_done")
                server_processing_done.set()
                break

            elif msg_type == "error":
                print("RECV: error →", data)
                server_processing_done.set()
                break

            else:
                print("RECV:", data)

    except websockets.ConnectionClosed:
        print("Connection closed")
        server_processing_done.set()

    finally:
        if playback_stream is not None:
            playback_stream.stop()
            playback_stream.close()
            print("🔊 Playback finished")


async def run_turn(websocket) -> bool:
    """Run a single send/receive turn on an already-open connection.

    Returns False when the user chooses to quit (so the caller closes the
    connection), True to keep the connection open for another turn.
    """
    # reset per-turn latency tracking and the processing-done event
    timing["end_of_utterance_sent_at"] = None
    timing["first_chunk_received_at"] = None
    server_processing_done.clear()

    # choose input mode (or quit)
    # NOTE: all interactive prompts run via asyncio.to_thread so the blocking
    # input() never freezes the event loop. If it did, the websockets client
    # couldn't answer the server's keepalive pings while the user is typing, and
    # the server would time out the connection (abnormal close / code 1006).
    while True:
        mode = (await asyncio.to_thread(input, "\nInput mode — [m]icrophone / [t]ext / [f]ile / [q]uit: ")).strip().lower()
        if mode in ("m", "t", "f", "q"):
            break
        print("Please enter 'm', 't', 'f', or 'q'.")

    if mode == "q":
        return False

    if mode == "t":
        user_text = (await asyncio.to_thread(input, "Enter your message: ")).strip()
        if not user_text:
            print("❌ No text entered. Skipping turn.")
            return True

        receiver_task = asyncio.create_task(receiver(websocket))
        await sender_text(websocket, user_text)
        await server_processing_done.wait()
        await receiver_task

    elif mode == "f":
        file_path = await asyncio.to_thread(pick_test_file)
        print(f"\n▶️  Using: {file_path}\n")

        receiver_task = asyncio.create_task(receiver(websocket))
        await sender_file(websocket, file_path)
        await server_processing_done.wait()
        await receiver_task

    else:
        await asyncio.to_thread(input, "\n▶️ Press ENTER to start recording...\n")

        # reset state
        stop_recording.clear()
        while not audio_queue.empty():
            try:
                audio_queue.get_nowait()
            except queue.Empty:
                break

        stopper_thread = threading.Thread(target=wait_for_enter_to_stop, daemon=True)
        stopper_thread.start()

        sender_task = asyncio.create_task(sender(websocket))
        receiver_task = asyncio.create_task(receiver(websocket))

        await sender_task
        await server_processing_done.wait()
        await receiver_task

    return True


async def main():
    async with websockets.connect(WS_URL, max_size=None, ping_interval=None, ping_timeout=None) as websocket:
        # 1) connection_established
        msg = await websocket.recv()
        print("RECV:", msg)

        # 2) start_session
        await websocket.send(json.dumps({
            "type": "start_session",
            "character_id": CHARACTER_ID,
            "sample_rate": SAMPLE_RATE,
            "audio_format": AUDIO_FORMAT
        }))
        print("SENT: start_session")

        # first server response
        msg = await websocket.recv()
        print("RECV:", msg)

        # Keep the connection open across multiple turns so conversation memory
        # persists. The user types 'q' to close (which clears history server-side).
        try:
            while True:
                keep_going = await run_turn(websocket)
                if not keep_going:
                    break
        except websockets.ConnectionClosed as e:
            print(f"⚠️  Connection closed by server (code={e.code}): {e.reason or 'no reason given'}")

        print("👋 Closing connection.")


if __name__ == "__main__":
    asyncio.run(main())