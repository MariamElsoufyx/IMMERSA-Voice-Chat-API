# IMMERSA Voice Chat API

A real-time, low-latency voice chat backend powering historical character roleplay for the **IMMERSA** immersive experience. Users speak with AI characters set in Egypt's first engineering school (Mohandeskhana, 1917–1918), receiving voice responses in character — in under a second from end-of-utterance to first audio.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Characters](#characters)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Database Setup](#database-setup)
  - [Running Locally](#running-locally)
- [WebSocket API](#websocket-api)
  - [Connecting](#connecting)
  - [Client → Server Events](#client--server-events)
  - [Server → Client Events](#server--client-events)
- [Pipeline Internals](#pipeline-internals)
- [FAQ System](#faq-system)
- [Scripts](#scripts)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contact](#contact)

---

## Overview

IMMERSA Voice Chat API is the backend engine behind interactive, voice-driven conversations with AI characters placed in 1917–1918 Egypt. The system accepts real-time audio from a client (game, web app, or test script), transcribes it, retrieves a contextually accurate response (either from a pre-built FAQ or an LLM), synthesises speech, and streams it back — all while staying in historical character.

The pipeline is optimised at every stage for low latency: speculative embedding during STT, an in-memory vector index for FAQ search, sentence-level TTS pipelining, and async queue-based concurrency throughout.

---

## Features

- **Real-time WebSocket voice chat** — full duplex audio streaming with per-chunk ACKs
- **5-stage async pipeline** — Preprocess → STT → FAQ/LLM → TTS → Stream, all running concurrently
- **Two STT backends** — Groq hosted Whisper (default, fast) or faster-whisper on-device
- **FAQ vector search** — cosine similarity with `all-MiniLM-L6-v2` embeddings + pgvector HNSW index
- **In-memory FAQ index** — all FAQ embeddings loaded at startup; searches are numpy dot products (< 1 ms, zero DB round trips)
- **Speculative embedding** — embedding starts during STT so it overlaps rather than adds latency
- **LLM fallback** — if no FAQ matches, Groq `llama-3.1-8b-instant` generates an in-character reply
- **ElevenLabs TTS with sentence pipelining** — next sentence is prefetched while current is streamed
- **Fallback audio** — per-character fallback WAV plays on any TTS failure so the user never hears silence
- **Interaction logging** — every question, answer, timing breakdown, and audio URL saved to `past_questions`
- **Audio archiving** — TTS response audio uploaded to Supabase Storage for every interaction
- **Comprehensive latency reports** — per-stage timing printed and written to disk after each utterance
- **Connection pool warmup** — 5 concurrent DB connections established at startup, eliminating cold-start lag

---

## Architecture

```
Client (WebSocket)
       │
       │ audio_chunk  (base64 WAV / PCM16)
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                        5-Stage Pipeline                         │
│                                                                 │
│  ┌────────────┐   ┌─────────┐   ┌────────────────────────┐    │
│  │ Preprocess │──▶│   STT   │──▶│  FAQ Lookup            │    │
│  │            │   │  Groq / │   │  1. Exact-match cache  │    │
│  │ High-pass  │   │  Local  │   │  2. Memory vector index│    │
│  │ Trim sil.  │   │ Whisper │   │  3. DB fallback        │    │
│  │ Normalise  │   └─────────┘   └──────────┬─────────────┘    │
│  └────────────┘         ▲  speculative      │ hit / miss       │
│                         │  embedding        ▼                  │
│                         │           ┌──────────────┐           │
│                         └───────────│  LLM (Groq)  │           │
│                                     │  llama-3.1-  │           │
│                                     │  8b-instant  │           │
│                                     └──────┬───────┘           │
│                                            │                   │
│                                     ┌──────▼───────┐           │
│                                     │  TTS          │           │
│                                     │  ElevenLabs   │           │
│                                     │  (sentence    │           │
│                                     │  pipelining)  │           │
│                                     └──────┬───────┘           │
│                                            │                   │
│                              tts_audio_chunk (WAV)             │
└────────────────────────────────────────────┼────────────────────┘
                                             │
                                      Client receives audio
                                             │
                                  ┌──────────▼──────────┐
                                  │  Background Tasks    │
                                  │  - Upload audio to   │
                                  │    Supabase Storage  │
                                  │  - Save to           │
                                  │    past_questions    │
                                  └─────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn (async WebSocket) |
| Speech-to-Text | Groq Whisper API / faster-whisper (local) |
| Language Model | Groq `llama-3.1-8b-instant` / OpenAI `gpt-5-nano` |
| Text-to-Speech | ElevenLabs `eleven_turbo_v2_5` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dim, local CPU) |
| Vector Search | pgvector (HNSW index) + in-memory numpy index |
| Database | PostgreSQL (Supabase) + SQLAlchemy 2.0 async + asyncpg |
| Migrations | Alembic |
| Audio Storage | Supabase Storage |
| Audio Processing | librosa, soundfile, scipy, noisereduce, numpy |
| HTTP Client | httpx (async) |
| Deployment | Railway |

---

## Characters

The API serves three historical characters from Egypt's Mohandeskhana (School of Engineering), 1917–1918. Each has a fixed personality, internal conflict, and a unique ElevenLabs voice.

### Morad Ali El-Attar — `s1`
> *Irrigation Engineering student. From a wealthy family. Polite and ambitious, but overconfident and prone to procrastination.*

"Am I truly capable… or just here because of my family name?"

### Kareem Hassan Shawky — `s2`
> *Mechanical Engineering student. Top of his class. Calm, responsible, and helpful — but carrying the weight of his family's poverty.*

"If I fail, my family has nothing."

### Amin Saleh El-Shazly — `p1`
> *Mechanical Engineering professor. Deeply knowledgeable, patient, and inspires genuine respect.*

"Should Egypt follow Europe… or define its own engineering path?"

All characters speak with knowledge strictly bounded to 1917–1918 Egypt. Modern concepts, events, or language are never used.

---

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL database with the `pgvector` extension enabled (Supabase recommended)
- API keys for Groq, ElevenLabs, and optionally OpenAI
- Supabase project with two storage buckets: `faq-audios` and `response-audios`

### Installation

```bash
git clone https://github.com/your-org/IMMERSA-Voice-Chat-API.git
cd IMMERSA-Voice-Chat-API

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```env
# --- AI Providers ---
GROQ_API_KEY=gsk_...
ELEVENLABS_API_KEY=sk_...
OPENAI_API_KEY=sk-proj-...          # optional — only needed if switching LLM to OpenAI

# --- ElevenLabs Voice IDs (one per character) ---
AHMAD_VOICE_ID=...                  # character s1 (Morad)
ACHRAF_VOICE_ID=...                 # character s2 (Kareem)
# Add additional voice IDs for more characters as needed

# --- Database ---
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/postgres

# --- Supabase Storage ---
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
SUPABASE_AUDIO_BUCKET=faq-audios            # pre-generated FAQ audio
SUPABASE_RESPONSES_BUCKET=response-audios   # archived LLM/TTS responses
```

> **Tip for Railway / cloud deployments:** Use the Supabase **Session Pooler** URL (e.g. `aws-0-eu-central-1.pooler.supabase.com`) instead of the direct DB host to avoid DNS resolution failures.

### Database Setup

Make sure `pgvector` is enabled on your PostgreSQL instance, then run all migrations:

```bash
alembic upgrade head
```

This creates:
- `frequently_asked_questions` — FAQ entries with 384-dim embeddings and an HNSW index
- `past_questions` — interaction log with full timing breakdown

### Running Locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On startup you will see:

```
🚀 [STARTUP] Loading models...
✅ [EMBED] Embedding model ready
✅ [DB] Connection pool warmed up (5/5 connections)
✅ [FAQ CACHE] 47 FAQs loaded into memory (s1=15, s2=16, p1=16)
✅ [PIPELINE] All 5 workers started
🎉 [STARTUP] Done
```

---

## WebSocket API

### Connecting

```
ws://localhost:8000/ws/voice-chat
```

Production:

```
wss://immersa-voice-chat-api.up.railway.app/ws/voice-chat
```

### Client → Server Events

All messages are JSON unless noted.

#### `start_session`
Initialise a session for a specific character.

```json
{
  "type": "start_session",
  "character_id": "s1",
  "sample_rate": 16000,
  "audio_format": "wav_base64_chunks"
}
```

| Field | Values | Default |
|---|---|---|
| `character_id` | `"s1"`, `"s2"`, `"p1"` | required |
| `sample_rate` | `16000` | `16000` |
| `audio_format` | `"wav_base64_chunks"` · `"pcm16_base64_chunks"` | `"wav_base64_chunks"` |

#### `audio_chunk`
Send a chunk of recorded audio (base64-encoded). The server runs rolling STT every 5 chunks so transcription starts before end-of-utterance.

```json
{
  "type": "audio_chunk",
  "audio": "<base64-encoded WAV or PCM16 bytes>"
}
```

#### `end_of_utterance`
Signal that the user has finished speaking. Triggers STT finalisation and full pipeline execution.

```json
{
  "type": "end_of_utterance"
}
```

#### `close_session`
Cleanly close the session and free resources.

```json
{
  "type": "close_session"
}
```

---

### Server → Client Events

#### `connection_established`
```json
{ "type": "connection_established", "session_id": "abc123" }
```

#### `session_started`
```json
{ "type": "session_started", "character_id": "s1" }
```

#### `audio_chunk_received`
```json
{ "type": "audio_chunk_received", "chunk_count": 3 }
```

#### `final_transcript`
```json
{ "type": "final_transcript", "transcript": "What subjects do you study?" }
```

#### `reply_text_done`
Full reply text from FAQ or LLM (arrives before any audio).
```json
{ "type": "reply_text_done", "answer": "We study hydraulics and land surveying…" }
```

#### `tts_audio_chunk`
A WAV audio chunk (base64). Chunks arrive sequentially and should be played in order.
```json
{
  "type": "tts_audio_chunk",
  "chunk_index": 0,
  "audio": "<base64-encoded WAV>"
}
```

#### `tts_done`
All audio chunks have been sent for this utterance. The session returns to `LISTENING`.
```json
{ "type": "tts_done" }
```

#### `error`
```json
{ "type": "error", "message": "STT failed: ..." }
```

---

## Pipeline Internals

The pipeline runs as **5 independent async workers** connected by queues. Each stage produces output for the next without blocking the event loop.

```
preprocess_queue → stt_queue → llm_queue → tts_queue → send_queue
```

### Latency optimisations

| Optimisation | Saving |
|---|---|
| FAQ in-memory vector index | Eliminates DB round trip (~500ms → <1ms) |
| Speculative embedding during STT | Embedding runs in parallel with remaining STT batches |
| Rolling STT every 5 chunks | Transcription progresses before end-of-utterance |
| TTS sentence pipelining | Next sentence prefetched while current plays |
| Noise reduction disabled | Saves 50–200ms per batch |
| DB connection pool warmup | Eliminates first-query cold start |
| Embedding model warmup | Eliminates first-inference cold start |

### Latency report (printed after each utterance)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⏱  LATENCY REPORT  —  Apr 25, 2026  03:12:08 PM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Preprocess  (batch 1)  : 0.031s
  STT         (batch 1)  : 0.284s
  FAQ lookup  (HIT ✅)   : 0.102s
  TTS first chunk        : 0.843s
  TTS total              : 2.114s
  ─────────────────────────────────────
  Time to first audio    : 1.261s
  Total                  : 3.403s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## FAQ System

FAQs are pre-authored question–answer pairs with a pre-generated voice recording. When a user's question semantically matches a FAQ entry (cosine similarity ≥ 0.7), the pipeline:

1. Returns the stored answer text instantly
2. Streams the pre-recorded audio (if available) — skipping LLM and TTS entirely

This is the lowest-latency path: typically 100–200ms after STT completes.

### FAQ lookup path

```
transcript
    │
    ├─ Exact-match cache? → 0ms (skips embedding too)
    │
    ├─ Memory vector index (numpy dot product) → < 1ms
    │
    └─ DB pgvector fallback (if cache not loaded) → 50–500ms
```

### Managing FAQs

Use the interactive CLI:

```bash
python scripts/faq_manager.py
```

Options:
- **Add FAQ** — enter a question; the LLM auto-generates an in-character answer, or enter manually
- **List all FAQs** — paginated table per character
- **Update** — edit question, answer, embedding, or audio
- **Delete** — remove a FAQ entry
- **Fill missing audio** — batch-generate and upload ElevenLabs audio for entries without it
- **Fill missing embeddings** — batch-embed entries that lack a vector

---

## Scripts

### `scripts/microphone_test.py`
Live end-to-end test from your machine. Three input modes:

```bash
python scripts/microphone_test.py
# [m]icrophone  — speak and hear the character reply
# [t]ext        — type a question, hear the reply
# [f]ile        — send a WAV file, hear the reply
```

Measures and prints the latency from **end-of-utterance → first audio chunk received**.

### `scripts/basic_test.py`
Sends a pre-recorded WAV file over WebSocket and saves the TTS response to `test_result.wav`.

```bash
python scripts/basic_test.py
```

### `scripts/faq_manager.py`
Interactive CLI for managing the FAQ database (see [FAQ System](#faq-system)).

---

## Deployment

The API is deployed on **Railway**. All environment variables are set in the Railway → Variables tab.

### Required Railway environment variables

```
GROQ_API_KEY
ELEVENLABS_API_KEY
OPENAI_API_KEY          (optional)
DATABASE_URL            (use Supabase Session Pooler URL)
SUPABASE_URL
SUPABASE_SERVICE_KEY
SUPABASE_AUDIO_BUCKET
SUPABASE_RESPONSES_BUCKET
AHMAD_VOICE_ID
ACHRAF_VOICE_ID
```

### Start command

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

### STT on Railway

Set `stt_provider = "groq"` in `app/core/config.py` (default). Local Whisper requires GPU or a significantly larger container.

### Supabase storage buckets

Create two **public** buckets in your Supabase project:
- `faq-audios` — pre-recorded FAQ audio (uploaded via `faq_manager.py`)
- `response-audios` — archived TTS audio from every interaction (uploaded automatically by the pipeline)

---

## Project Structure

```
IMMERSA-Voice-Chat-API/
├── app/
│   ├── main.py                        # FastAPI app, lifespan startup/shutdown
│   ├── api/
│   │   └── websocket_routes.py        # WebSocket /ws/voice-chat endpoint
│   ├── characters/
│   │   ├── build_prompt.py            # Prompt assembly engine
│   │   ├── characters_info.py         # Character metadata & voice IDs
│   │   └── prompts.py                 # System & user prompt templates
│   ├── core/
│   │   ├── clients.py                 # AI client initialisation (Groq, ElevenLabs, OpenAI)
│   │   └── config.py                  # All configuration constants & env vars
│   ├── db/
│   │   ├── database.py                # Async engine & session factory
│   │   ├── models.py                  # ORM models: FAQ, PastQuestion
│   │   └── repositories/
│   │       ├── faq_repository.py      # FAQ CRUD + pgvector similarity search
│   │       └── past_questions_repository.py
│   ├── services/
│   │   ├── audio/
│   │   │   └── preprocessor.py        # High-pass filter, silence trim, normalise
│   │   ├── embedding_service.py       # BAAI/bge-small-en-v1.5 embeddings (local CPU)
│   │   ├── faq_memory_cache.py        # In-memory vector index (numpy, no DB)
│   │   ├── llm/
│   │   │   ├── groq_service.py        # Groq LLM
│   │   │   └── openai_service.py      # OpenAI LLM (fallback)
│   │   ├── pipeline/
│   │   │   └── pipeline.py            # 5-stage async queue pipeline
│   │   ├── streaming/
│   │   │   ├── audio_buffer.py        # Per-session chunk accumulator
│   │   │   ├── connection_manager.py  # WebSocket session registry
│   │   │   ├── event_protocol.py      # JSON event builders
│   │   │   └── stream_session.py      # Per-connection state dataclass
│   │   ├── stt/
│   │   │   ├── groq_whisper.py        # Groq Whisper API
│   │   │   └── local_whisper.py       # faster-whisper on-device
│   │   └── tts/
│   │       └── elevenlabs_service.py  # ElevenLabs streaming TTS
│   └── utils/
│       └── response_logger.py         # Saves LLM replies to disk
├── alembic/
│   └── versions/                      # Database migration history
├── scripts/
│   ├── microphone_test.py             # Live mic/text/file test client
│   ├── basic_test.py                  # WAV file WebSocket test
│   └── faq_manager.py                 # Interactive FAQ management CLI
├── data/
│   └── fallback_audios/               # Per-character fallback WAV files
│       ├── s1_fallback.wav
│       ├── s2_fallback.wav
│       └── p1_fallback.wav
├── .env                               # Local environment variables (not committed)
├── alembic.ini
└── requirements.txt
```

---

## Contact

**Mariam Elsoufyx** — mariamelsoufyx@gmail.com

---

<p align="center">Built for the IMMERSA immersive experience &mdash; Egypt, 1917</p>
