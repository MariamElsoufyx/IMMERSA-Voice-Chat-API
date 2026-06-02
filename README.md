# IMMERSA Voice Chat API

A real-time, low-latency voice chat backend powering historical character roleplay for the **IMMERSA** immersive experience. Users speak with AI characters set at **Al-Mohandeskhana** — the historical Egyptian engineering school (founded 1816 at the Cairo Citadel) that became the **Faculty of Engineering, Cairo University** — and receive in-character voice responses, with the first audio typically arriving about a second after they stop speaking.

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
  - [Session States](#session-states)
- [Pipeline Internals](#pipeline-internals)
- [FAQ System](#faq-system)
- [RAG / History Grounding](#rag--history-grounding)
- [Verification Layer](#verification-layer)
- [Conversation Memory](#conversation-memory)
- [REST Endpoints](#rest-endpoints)
- [Tooling & Scripts](#tooling--scripts)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contact](#contact)

---

## Overview

IMMERSA Voice Chat API is the backend engine behind interactive, voice-driven conversations with AI characters placed at Al-Mohandeskhana / Cairo University's Faculty of Engineering. The system accepts real-time audio from a client (game, web app, or test script), transcribes it, finds a contextually accurate response (from a pre-built FAQ or an LLM grounded in retrieved history), synthesises speech, and streams it back — all while staying in historical character.

Each character is anchored to a specific year via its `operation_year` (see [Characters](#characters)), and the LLM, prompts, and anachronism checks are all bounded to that era so a character never references anything from after its time.

The pipeline is optimised at every stage for low latency: speculative embedding during STT, in-memory vector indexes for both FAQ and history search, sentence-level TTS pipelining, and async queue-based concurrency throughout. Content verification runs **in parallel with** speech synthesis so it adds safety without adding latency on the happy path.

---

## Features

- **Real-time WebSocket voice chat** — full-duplex audio streaming with a per-message `ack`
- **5-stage async pipeline** — Preprocess → STT → LLM (FAQ / RAG / LLM) → TTS → Send, all running concurrently
- **Two STT backends** — Groq hosted Whisper (`whisper-large-v3-turbo`, default) or `faster-whisper` on-device (`tiny.en`)
- **FAQ vector search** — cosine similarity (≥ 0.8) with `BAAI/bge-small-en-v1.5` embeddings + pgvector HNSW index
- **In-memory FAQ & history indexes** — all embeddings loaded at startup; searches are numpy dot products (< 1 ms, zero DB round trips)
- **Speculative embedding** — embedding starts during STT so it overlaps rather than adds latency
- **RAG history grounding** — on an FAQ miss, the top-K most similar history chunks are retrieved and injected into the LLM prompt, with a context-aware query for follow-up questions
- **LLM replies** — Groq `llama-3.1-8b-instant` generates an in-character, JSON-structured reply (answer + emotion + sources)
- **Verification layer** — synchronous regex gates (profanity + anachronism) before TTS, plus OpenAI Moderation + an LLM judge that races TTS and can correct or replace a flagged answer
- **ElevenLabs TTS with sentence pipelining** — next sentence is prefetched while the current one streams; per-sentence MP3 is decoded, silence-trimmed, and re-encoded to WAV before sending
- **Fallback & verify audio** — per-character WAV clips play on TTS failure or when the verifier rejects a reply, so the user never hears silence
- **Multi-turn conversation memory** — the last few (question, answer) pairs per connection are fed back to the LLM for coherent follow-ups
- **Interaction logging** — every question, answer, emotion, timing breakdown, verifier result, and audio URL saved to `past_questions`
- **Audio archiving** — both the client's question audio and the TTS response audio are uploaded to Supabase Storage
- **Comprehensive latency reports** — per-stage timing printed after each utterance
- **Connection-pool & model warmup** — DB pool, embedding model, and ElevenLabs voices are warmed at startup to eliminate cold-start lag

---

## Architecture

```
Client (WebSocket)
       │
       │ audio_chunk  (base64 WAV / PCM16)
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          5-Stage Pipeline                              │
│                                                                        │
│  ┌────────────┐   ┌─────────┐   ┌──────────────────────────────┐     │
│  │ Preprocess │──▶│   STT   │──▶│  FAQ lookup                   │     │
│  │            │   │  Groq / │   │  1. Exact-match cache         │     │
│  │ High-pass  │   │  Local  │   │  2. In-memory vector index    │     │
│  │ Trim sil.  │   │ Whisper │   │  3. DB pgvector fallback      │     │
│  │ Normalise  │   └─────────┘   └──────────┬───────────────────┘     │
│  └────────────┘         ▲  speculative      │ hit / miss              │
│                         │  embedding         ▼                        │
│                         │            ┌───────────────┐                │
│                         │            │  RAG retrieval │ (on miss)      │
│                         │            │  history chunks│                │
│                         │            └──────┬─────────┘                │
│                         │                   ▼                          │
│                         │            ┌───────────────┐                │
│                         └────────────│  LLM (Groq)    │                │
│                                      │  llama-3.1-    │                │
│                                      │  8b-instant    │                │
│                                      └──────┬─────────┘                │
│                                             │                          │
│                          regex gate (profanity + anachronism)          │
│                                             │                          │
│                                      ┌──────▼─────────┐                │
│                                      │  TTS           │   verify       │
│                                      │  ElevenLabs    │◀─ (moderation  │
│                                      │  (sentence     │    + LLM judge │
│                                      │  pipelining)   │    in parallel)│
│                                      └──────┬─────────┘                │
│                                             │                          │
│                              tts_audio_chunk (WAV)                     │
└─────────────────────────────────────────────┼─────────────────────────┘
                                              │
                                       Client receives audio
                                              │
                                  ┌───────────▼───────────┐
                                  │  Background Tasks      │
                                  │  - Upload question +   │
                                  │    response audio to   │
                                  │    Supabase Storage    │
                                  │  - Save to             │
                                  │    past_questions      │
                                  └────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn (async WebSocket) |
| Speech-to-Text | Groq Whisper API (`whisper-large-v3-turbo`) / `faster-whisper` (`tiny.en`, local) |
| Reply LLM | Groq `llama-3.1-8b-instant` (OpenAI `gpt-5-nano` available as an alternate backend) |
| Verification LLM | OpenAI `gpt-4.1-nano` (LLM judge) + OpenAI Moderation API |
| Text-to-Speech | ElevenLabs `eleven_v3` (MP3 internally → re-encoded to WAV per sentence) |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim, local CPU, asymmetric query prefix) |
| Vector Search | pgvector (HNSW index) + in-memory numpy indexes |
| Database | PostgreSQL (Supabase) + SQLAlchemy 2.0 async + asyncpg |
| Migrations | Alembic |
| Audio Storage | Supabase Storage |
| Audio Processing | librosa, soundfile, scipy, noisereduce, numpy |
| HTTP Client | httpx (async) |
| Deployment | GCP VM · Docker Compose · nginx (TLS via Let's Encrypt) |

---

## Characters

The API serves three characters from Al-Mohandeskhana / Cairo University's Faculty of Engineering. Each has a fixed personality, internal conflict, a unique ElevenLabs voice, and an `operation_year` that bounds their knowledge and the anachronism check.

### Morad Ali El-Attar — `s1`  ·  1918
> *Irrigation Engineering student. From a wealthy family, raised partly in Britain. Polite and ambitious with leadership presence, but overconfident and prone to procrastination.*

"Am I truly capable… or am I only here because of my family name?"

### Hassan Kareem Shawky — `s2`  ·  1960
> *Mechanical Engineering student and top of his class. Calm under technical pressure, observant and helpful — but socially awkward and insecure, carrying the weight of his struggling family.*

"If I fail, my family has nothing."

### Amin Saleh El-Shazly — `p1`  ·  1918
> *Mechanical Engineering professor, French-trained (doctorate from France). Deeply knowledgeable, patient, and inspires genuine respect, though he can be rigid and emotionally distant.*

"Should Egypt follow Europe… or define its own engineering path?"

Each character answers strictly within their own era. Modern concepts, events, or language are never used, and any 4-digit year past a character's `operation_year` is flagged by the anachronism check (see [Verification Layer](#verification-layer)).

---

## Getting Started

### Prerequisites

- Python 3.11
- PostgreSQL database with the `pgvector` extension enabled (Supabase recommended)
- API keys for Groq, ElevenLabs, and OpenAI (OpenAI powers the verification layer)
- Supabase project with storage buckets for FAQ audio, response audio, and question audio
- `ffmpeg` and `libsndfile` available (already included in the Docker image)

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
OPENAI_API_KEY=sk-proj-...          # used by the verification layer (moderation + LLM judge)

# --- ElevenLabs Voice IDs (one per character) ---
AHMAD_VOICE_ID=...                  # character s1 (Morad)
ACHRAF_VOICE_ID=...                 # character s2 (Hassan)
ROGER_VOICE_ID=...                  # character p1 (Amin)

# --- Database ---
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/postgres

# --- Supabase Storage ---
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
SUPABASE_AUDIO_BUCKET=faq-audios            # pre-generated FAQ audio (used by helpers/faq_manager.py)
SUPABASE_RESPONSES_BUCKET=response-audios   # archived TTS responses (uploaded by the pipeline)
SUPABASE_QUESTIONS_BUCKET=question-audios   # archived client question audio (uploaded by the pipeline)
```

> **Tip for cloud deployments:** Use the Supabase **Session Pooler** URL (e.g. `aws-0-eu-central-1.pooler.supabase.com`) instead of the direct DB host to avoid DNS resolution failures.

### Database Setup

Make sure `pgvector` is enabled on your PostgreSQL instance, then run all migrations:

```bash
alembic upgrade head
```

This creates:
- `frequently_asked_questions` — FAQ entries with 384-dim embeddings and an HNSW index
- `RAG_data` — college-history chunks for RAG grounding (also embedded, also HNSW)
- `past_questions` — interaction log with full timing + verification breakdown

### Running Locally

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# or:  make server
```

On startup you will see (abridged):

```
🚀 [STARTUP] Loading models...
✅ [EMBEDDING] Model loaded
⏳ [TTS] Warming up ElevenLabs (model=eleven_v3, 3 voices)...
✅ [TTS] ElevenLabs ready (model=eleven_v3)
✅ [DB] Connection pool warmed up (5/5 connections)
✅ [FAQ CACHE] FAQs loaded into memory
✅ [HISTORY CACHE] history chunks loaded into memory
✅ [PIPE] all 5 workers started (preprocess → stt → llm → tts → send)
🎉 [STARTUP] Done
```

---

## WebSocket API

> A full, browseable reference lives in [`docs/index.html`](docs/index.html) and the machine-readable AsyncAPI 2.6 spec in [`docs/asyncapi.yaml`](docs/asyncapi.yaml).

### Connecting

```
ws://localhost:8000/ws/voice-chat
```

Production:

```
wss://immersa-api-voice-chat.run.place/ws/voice-chat
```

All messages in both directions are JSON objects identified by a `type` field.

### Client → Server Events

#### `start_session`
Initialise a session for a specific character. Send once after `connection_established`.

```json
{
  "type": "start_session",
  "character_id": "s1",
  "sample_rate": 16000,
  "audio_format": "wav"
}
```

| Field | Values | Default |
|---|---|---|
| `character_id` | `"s1"`, `"s2"`, `"p1"` | required |
| `sample_rate` | integer (Hz) | `16000` |
| `audio_format` | `"wav"` · `"pcm16_base64_chunks"` | `"wav"` |

#### `audio_chunk`
Send a chunk of recorded audio (base64-encoded). The server runs rolling STT every 5 chunks so transcription progresses before end-of-utterance.

```json
{
  "type": "audio_chunk",
  "chunk_index": 0,
  "audio": "<base64-encoded WAV or PCM16 bytes>"
}
```

For `audio_format: "wav"`, the first chunk should be a complete WAV file (header + PCM); subsequent chunks may be headerless PCM — the server caches the header and reassembles. For `pcm16_base64_chunks`, every chunk is raw PCM16-LE.

#### `end_of_utterance`
Signal that the user has finished speaking. Triggers final STT, the LLM/FAQ pipeline, and TTS streaming.

```json
{ "type": "end_of_utterance" }
```

#### `close_session`
Cleanly close the session and free resources.

```json
{ "type": "close_session" }
```

### Server → Client Events

#### `connection_established`
Sent immediately after the handshake.
```json
{
  "type": "connection_established",
  "session_id": "3f4a1b2c-8d9e-4f5a-b6c7-d8e9f0a1b2c3",
  "message": "WebSocket connected successfully"
}
```

#### `ack`
Generic acknowledgement for **every** client message. The `event` field mirrors the message being acked; extra fields depend on the event.

```json
{ "type": "ack", "event": "start_session", "message": "Session started successfully",
  "session_id": "3f4a1b2c-...", "character_id": "s1", "sample_rate": 16000,
  "audio_format": "wav", "state": "LISTENING" }
```
```json
{ "type": "ack", "event": "audio_chunk", "message": "Audio chunk received",
  "chunk_index": 0, "total_chunks": 1 }
```
`end_of_utterance` and `close_session` acks carry only `type`, `event`, and `message`.

#### `final_transcript`
Complete transcription of the full utterance, sent after `end_of_utterance`.
```json
{ "type": "final_transcript", "text": "What subjects do you study?" }
```

#### `reply_text_done`
The character's full reply text (arrives before any audio). `emotion` is one of `happy`, `sad`, `angry`, `disgust`, `surprise`, `neutral` (or `null`).
```json
{ "type": "reply_text_done", "text": "We study hydraulics and land surveying…",
  "length": 39, "emotion": "neutral" }
```

#### `tts_audio_chunk`
One self-contained **WAV** file (base64), one per sentence. ElevenLabs returns MP3 internally; the server decodes, silence-trims, and re-encodes each sentence as **WAV PCM_16, 44 100 Hz, mono** before sending. Play chunks in order.
```json
{ "type": "tts_audio_chunk", "chunk_index": 0, "audio": "<base64-encoded WAV>" }
```

#### `tts_done`
All audio chunks have been sent. The session returns to `LISTENING`.
```json
{ "type": "tts_done" }
```

#### `error`
```json
{ "type": "error", "message": "Failed to decode base64 audio", "chunk_index": 3 }
```

> There is no `partial_transcript` event — rolling STT happens internally and only the `final_transcript` is sent.

### Session States

```
CONNECTED → LISTENING → FINALIZING_TRANSCRIPT → GENERATING_REPLY → STREAMING_TTS → LISTENING
LISTENING → CLOSED   (via close_session)
```

The current `state` is included in the `start_session` ack.

---

## Pipeline Internals

The pipeline runs as **5 independent async workers** connected by queues. Each stage produces output for the next without blocking the event loop.

```
preprocess_queue → stt_queue → llm_queue → tts_queue → send_queue
```

The `llm_queue` worker owns the decision logic: FAQ lookup → (on miss) RAG retrieval + LLM generation → regex gating → TTS, with moderation/judge verification kicked off in parallel.

### Latency optimisations

| Optimisation | Effect |
|---|---|
| In-memory FAQ & history indexes | Eliminates DB round trips (~500 ms → < 1 ms) |
| Speculative embedding during STT | Embedding runs in parallel with remaining STT batches; reused for both FAQ and RAG |
| Rolling STT every 5 chunks | Transcription progresses before end-of-utterance |
| TTS sentence pipelining | Next sentence prefetched while the current one plays |
| Verification races TTS | Moderation + LLM judge run alongside speech, not before it |
| Noise reduction disabled | Saves 50–200 ms per batch |
| DB pool / embedding / TTS warmup | Eliminates first-request cold starts |

### Latency report (printed after each utterance)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⏱  LATENCY REPORT  —  Apr 25, 2026  03:12:08 PM (Cairo)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Preprocess  (batch 1)  : 0.031s
  STT         (batch 1)  : 0.284s
  FAQ lookup  (miss ❌)  : 0.012s
  RAG (6 chunk(s))       : 0.041s
  LLM                    : 0.612s
  Content filter         : 0.001s
  Anachronism            : PASS ✅
  TTS first chunk        : 0.843s
  TTS total              : 2.114s
  Verifier               : 1.380s
  ─────────────────────────────────────
  Time to first audio    : 1.261s
  Total                  : 3.403s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## FAQ System

FAQs are pre-authored question–answer pairs, each with an emotion and an optional pre-generated voice recording. When a user's question semantically matches a FAQ entry (cosine similarity ≥ 0.8), the pipeline:

1. Returns the stored answer text instantly (with its emotion)
2. Streams the pre-recorded audio if available — skipping the LLM and TTS entirely; otherwise sends the answer to TTS

This is the lowest-latency path.

### FAQ lookup path

```
transcript
    │
    ├─ Exact-match cache? → 0ms (skips embedding + search)
    │
    ├─ In-memory vector index (numpy dot product) → < 1ms
    │
    └─ DB pgvector fallback (if cache not loaded) → 50–500ms
```

Manage FAQs with the interactive CLI (`python -m helpers.faq_manager` or `make faq`) — add (LLM auto-generates an in-character answer or enter manually), list, update, delete, batch-fill missing audio, and batch-fill missing embeddings.

---

## RAG / History Grounding

When no FAQ matches, the LLM is grounded in real history rather than left to invent facts:

1. Source documents live in `data/history/` and are indexed into the `RAG_data` table (chunked, embedded with `BAAI/bge-small-en-v1.5`).
2. On an FAQ miss, the query embedding (reused from STT) searches the in-memory history index for the top `HISTORY_TOP_K` (default 6) chunks above a similarity threshold of 0.4.
3. For **follow-up** questions, a context-aware query prepends the last couple of user turns so an elliptical question like *"when was it?"* still retrieves the right facts.
4. Chunks scoped to a character (filename prefix `s1__…`) plus global chunks (no prefix) are injected into the prompt's `<history>` block. The LLM must answer **only** from that block or admit it doesn't know.

Manage and re-index history with `python -m helpers.history_manager` (or `make history`). Indexing is idempotent per source file.

---

## Verification Layer

Every LLM reply passes through a tiered verification layer (`app/services/verification/`) designed so safety never costs latency on clean answers:

**Tier 1 — synchronous regex (sub-millisecond, gates *before* TTS)**
- Profanity check on the user's transcript
- Profanity + anachronism check on the LLM answer. Anachronism flags any 4-digit year past the character's `operation_year` (plus modern terms, URLs, emails).
- A flagged answer is **never** played: the verifier LLM is asked to regenerate a clean reply; if that fails, a static fallback/verify clip plays instead.

**Tier 2 / 3 — async checks that *race* TTS**
- OpenAI Moderation on the question (kicked off in parallel with FAQ lookup, cancelled on an FAQ hit)
- OpenAI Moderation on the answer + an OpenAI **LLM judge** (`gpt-4.1-nano`) scoring historical accuracy, appropriateness, modern references, and in-character fidelity
- If a check fails mid-stream, TTS is aborted. If the judge supplied a `corrected_answer`, the verify clip plays and the corrected reply is re-synthesised; otherwise the static verify audio plays and the rejected answer is dropped from conversation memory.

Toggles live in `app/core/config.py` (`MODERATION_ENABLED`, `ANACHRONISM_ENABLED`). All verifier outcomes and timings are logged to `past_questions`.

---

## Conversation Memory

Each WebSocket connection keeps a short multi-turn memory — the last `CONVERSATION_HISTORY_MAX_TURNS` (default 6) `(question, answer)` pairs — fed back to the LLM so follow-ups stay coherent. FAQ hits are recorded too, and verifier corrections overwrite the stored turn so memory matches what the user actually heard. Memory lives only for the duration of the connection and is cleared on `close_session`.

---

## REST Endpoints

Alongside the WebSocket, the app exposes a small HTTP surface:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/` | Liveness message |
| `GET` | `/health` | Health check (`{"status": "ok"}`) |
| `GET` | `/faqs` | List FAQs (optional `?character_id=`) |
| `PATCH` | `/faqs/{faq_id}` | Update a FAQ (question, answer, tag, language, emotion) |
| `PATCH` | `/faqs/{faq_id}/emotion` | Set a FAQ's emotion |
| `DELETE` | `/faqs/{faq_id}` | Delete one FAQ |
| `DELETE` | `/faqs` | Delete all FAQs |

Valid emotions: `happy`, `sad`, `angry`, `disgust`, `surprise`, `neutral`.

---

## Tooling & Scripts

A `Makefile` wraps the common entry points:

```bash
make server    # uvicorn app.main:app --reload
make faq       # python -m helpers.faq_manager      — interactive FAQ manager
make history   # python -m helpers.history_manager   — RAG history manager + indexer
make test      # python -m helpers.test_script       — live mic/text/file test client
make audio     # python -m helpers.generate_audio_from_text — TTS a string to a WAV
```

- **`helpers/test_script.py`** — end-to-end test client. Records from the mic (or sends a WAV from `helpers/test_files/`), streams it over the WebSocket, plays the reply, and measures end-of-utterance → first-audio latency. Defaults to `pcm16_base64_chunks` against the production URL (both configurable at the top of the file).
- **`helpers/faq_manager.py`** — interactive CLI for the FAQ table (see [FAQ System](#faq-system)).
- **`helpers/history_manager.py`** — interactive CLI for the `RAG_data` store: add, index/re-index files from `data/history/`, list, view, update, delete, and test retrieval with similarity scores.
- **`helpers/index_history.py`** — chunking/indexing library used by the history manager (no CLI of its own).
- **`helpers/generate_audio_from_text.py`** — generate a WAV from text using the exact ElevenLabs settings from `config.py`.

---

## Deployment

The API is deployed on a **GCP VM** running **Docker Compose** behind **nginx** (TLS via Let's Encrypt). Deployment is automated by GitHub Actions: a push to `production` SSHes into the VM, pulls, and runs `docker compose up -d --build` (see [`.github/workflows/deploy.yml`](.github/workflows/deploy.yml)).

### Stack

- **`api`** service — the FastAPI app (`uvicorn app.main:app --host 0.0.0.0 --port 8000`), with a `/health` healthcheck and persistent HuggingFace / Whisper model caches.
- **`nginx`** service — terminates TLS for `immersa-api-voice-chat.run.place`, redirects HTTP→HTTPS, and reverse-proxies WebSocket traffic with long read/send timeouts for long-lived voice sessions.

### Run it yourself

```bash
docker compose up -d --build
```

All secrets are provided via the `.env` file (loaded by `env_file` in `docker-compose.yml`).

### STT in production

Keep `stt_provider = "groq"` in `app/core/config.py` (default). Local `faster-whisper` needs more CPU/RAM (or a GPU) than the slim container provides.

### Supabase storage buckets

Create three **public** buckets in your Supabase project:
- `faq-audios` — pre-recorded FAQ audio (uploaded via `helpers/faq_manager.py`)
- `response-audios` — archived TTS response audio (uploaded automatically by the pipeline)
- `question-audios` — archived client question audio (uploaded automatically by the pipeline)

---

## Project Structure

```
IMMERSA-Voice-Chat-API/
├── app/
│   ├── main.py                        # FastAPI app, lifespan startup/shutdown
│   ├── api/
│   │   ├── websocket_routes.py        # WebSocket /ws/voice-chat endpoint
│   │   └── faq_routes.py              # REST /faqs CRUD endpoints
│   ├── characters/
│   │   ├── build_prompt.py            # Prompt assembly engine
│   │   ├── characters_info.py         # Character metadata, voices, operation_year
│   │   └── prompts.py                 # Narrator / verifier system & user prompts
│   ├── core/
│   │   ├── clients.py                 # AI client init (Groq, ElevenLabs, OpenAI)
│   │   └── config.py                  # All configuration constants & env vars
│   ├── db/
│   │   ├── database.py                # Async engine & session factory
│   │   ├── models.py                  # ORM models: FAQ, HistoryChunk, PastQuestion
│   │   └── repositories/
│   │       ├── faq_repository.py      # FAQ CRUD + pgvector similarity search
│   │       ├── history_repository.py  # RAG_data CRUD + similarity search
│   │       └── past_questions_repository.py
│   ├── services/
│   │   ├── audio/preprocessor.py      # High-pass filter, silence trim, normalise
│   │   ├── embedding_service.py       # BAAI/bge-small-en-v1.5 embeddings (local CPU)
│   │   ├── faq_memory_cache.py        # In-memory FAQ vector index (numpy)
│   │   ├── history_memory_cache.py    # In-memory RAG vector index (numpy)
│   │   ├── llm/
│   │   │   ├── groq_service.py        # Groq LLM (active reply backend)
│   │   │   └── openai_service.py      # OpenAI LLM (alternate backend)
│   │   ├── pipeline/pipeline.py       # 5-stage async queue pipeline
│   │   ├── streaming/
│   │   │   ├── audio_buffer.py        # Per-session chunk accumulator
│   │   │   ├── connection_manager.py  # WebSocket session registry
│   │   │   ├── event_protocol.py      # JSON event builders
│   │   │   └── stream_session.py      # Per-connection state + conversation memory
│   │   ├── stt/
│   │   │   ├── groq_whisper.py        # Groq Whisper API
│   │   │   └── local_whisper.py       # faster-whisper on-device
│   │   ├── tts/elevenlabs_service.py  # ElevenLabs streaming TTS
│   │   └── verification/              # Tiered verification (regex + models)
│   │       ├── base.py                # CheckResult / AggregateResult
│   │       ├── regex_checks.py        # Profanity + anachronism
│   │       ├── model_checks.py        # OpenAI moderation + LLM judge
│   │       └── orchestrator.py        # Composes checks into a pipeline API
│   └── utils/
│       ├── log.py                     # Structured console logger
│       └── utils.py
├── alembic/versions/                  # Database migration history
├── helpers/
│   ├── faq_manager.py                 # Interactive FAQ management CLI
│   ├── history_manager.py             # Interactive RAG/history CLI
│   ├── index_history.py              # History chunking/indexing library
│   ├── generate_audio_from_text.py    # Text → WAV via ElevenLabs
│   ├── test_script.py                 # Live mic/text/file WebSocket test client
│   └── test_files/                    # Sample WAVs for testing
├── data/
│   ├── history/                       # Source docs indexed into RAG_data
│   ├── fallback_audios/               # Per-character TTS-failure clips (s1, p1)
│   ├── verification_audios/           # Per-character verifier-reject clips (s1, p1)
│   └── thinking_audios/               # Per-character "thinking" clips
├── docs/
│   ├── index.html                     # Browseable WebSocket API reference
│   └── asyncapi.yaml                  # AsyncAPI 2.6 spec
├── Dockerfile
├── docker-compose.yml                 # api + nginx services
├── nginx.conf                         # TLS + WebSocket reverse proxy
├── .github/workflows/deploy.yml       # GCP VM deploy on push to production
├── Makefile
├── alembic.ini
├── .env                               # Local environment variables (not committed)
└── requirements.txt
```

---

## Contact

**Mariam Elsoufyx** — mariamelsoufyx@gmail.com

---

<p align="center">Built for the IMMERSA immersive experience &mdash; Al-Mohandeskhana, Cairo</p>
