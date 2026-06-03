# IMMERSA Voice Chat API

A real-time, low-latency voice chat backend powering historical-character roleplay for the **IMMERSA** immersive experience. Users speak with AI characters drawn from the history of the **Faculty of Engineering, Cairo University** — students and a professor, each anchored to their own era of Egyptian engineering education — and receive in-character voice responses in roughly a second from end-of-utterance to first audio.

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
- [REST API](#rest-api)
- [Pipeline Internals](#pipeline-internals)
- [FAQ System](#faq-system)
- [History RAG](#history-rag)
- [Verification & Safety](#verification--safety)
- [Helper Scripts](#helper-scripts)
- [Deployment](#deployment)
- [Project Structure](#project-structure)
- [Contact](#contact)

---

## Overview

IMMERSA Voice Chat API is the backend engine behind interactive, voice-driven conversations with AI characters set at the Faculty of Engineering, Cairo University. The system accepts real-time audio from a client (game, web app, or test script), transcribes it, retrieves a contextually accurate response (from a pre-built FAQ or an LLM grounded in a curated history corpus), synthesises speech, and streams it back — all while staying strictly in historical character.

Each character is bound to a specific year, and the LLM is grounded in a retrieval corpus of Egyptian engineering history so replies stay period-accurate. Every generated answer also passes through a tiered verification layer (profanity, anachronism, moderation, and an LLM judge) that can correct or replace a reply mid-stream before the user finishes hearing it.

The pipeline is optimised at every stage for low latency: rolling (speculative) STT during recording, speculative embedding that overlaps with transcription, in-memory vector indexes for both FAQ and history search, sentence-level TTS pipelining, and async queue-based concurrency throughout.

---

## Features

- **Real-time WebSocket voice chat** — full-duplex audio streaming with per-message ACKs
- **5-stage async pipeline** — Preprocess → STT → LLM → TTS → Send, all running concurrently as independent workers
- **Two STT backends** — Groq hosted Whisper (`whisper-large-v3-turbo`, default) or `faster-whisper` on-device
- **Rolling STT** — partial transcription runs server-side every 5 chunks so transcription progresses before end-of-utterance
- **Speculative embedding** — the query embedding starts on the first partial transcript, overlapping with the rest of STT
- **FAQ vector search** — cosine similarity over `BAAI/bge-small-en-v1.5` embeddings, served from an in-memory numpy index (DB pgvector HNSW as fallback)
- **History RAG** — on FAQ miss, relevant history chunks are retrieved from an in-memory index and injected into the prompt to ground the LLM
- **Conversation memory** — per-connection multi-turn context, with a context-aware retrieval query so follow-ups ("when was it?") resolve correctly
- **LLM replies with emotion** — Groq `llama-3.1-8b-instant` returns `{ answer, emotion }`; the emotion drives TTS expressiveness
- **Tiered verification** — regex profanity/anachronism gates, OpenAI Moderation on question and answer, and an LLM judge scoring historical accuracy, appropriateness, modern references, and in-character fidelity — running **in parallel with TTS** and able to abort and replay a corrected answer
- **ElevenLabs TTS with sentence pipelining** — the next sentence is prefetched while the current one streams
- **Per-character fallback & verify audio** — pre-recorded WAVs play on TTS failure or verifier rejection so the user never hears silence
- **Interaction logging** — every question, answer, emotion, timing breakdown, and verification result saved to `past_questions`
- **Audio archiving** — both the user's question audio and the TTS response audio are uploaded to Supabase Storage
- **Comprehensive latency reports** — per-stage timing printed after each utterance
- **Warm starts** — embedding model, DB connection pool, FAQ cache, history cache, and ElevenLabs voices are all warmed at startup

---

## Architecture

```
Client (WebSocket)
       │
       │ audio_chunk  (base64 WAV / PCM16)
       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          5-Stage Pipeline                             │
│                                                                       │
│  ┌────────────┐   ┌─────────┐   ┌────────────────────────┐           │
│  │ Preprocess │──▶│   STT   │──▶│  FAQ Lookup            │           │
│  │            │   │  Groq / │   │  1. Exact-match cache  │           │
│  │ WAV rebuild│   │  Local  │   │  2. Memory vector index│           │
│  │ High-pass  │   │ Whisper │   │  3. DB pgvector        │           │
│  │ Normalise  │   └─────────┘   └──────────┬─────────────┘           │
│  └────────────┘    ▲ speculative           │ hit / miss              │
│                    │ embedding             ▼                         │
│                    │              ┌──────────────────┐                │
│                    │              │  History RAG     │  (on miss)     │
│                    │              │  in-memory index │                │
│                    │              └────────┬─────────┘                │
│                    │                       ▼                         │
│                    │              ┌──────────────────┐                │
│                    └──────────────│  LLM (Groq)      │                │
│                                   │  + conversation  │                │
│                                   │    memory        │                │
│                                   └────────┬─────────┘                │
│                                            │ answer + emotion         │
│                ┌───────────────────────────┼──────────────┐          │
│                │                            ▼              │          │
│        ┌───────▼────────┐          ┌────────────────┐     │          │
│        │  Verification  │◀────────▶│  TTS           │     │          │
│        │  (parallel)    │  abort / │  ElevenLabs    │     │          │
│        │  regex · mod · │  replay  │  (sentence     │     │          │
│        │  LLM judge     │          │   pipelining)  │     │          │
│        └────────────────┘          └───────┬────────┘     │          │
│                                            │ tts_audio_chunk (WAV)    │
└────────────────────────────────────────────┼─────────────────────────┘
                                             │
                                      Client receives audio
                                             │
                                  ┌──────────▼──────────┐
                                  │  Background Tasks    │
                                  │  - Upload question + │
                                  │    response audio    │
                                  │  - Save to           │
                                  │    past_questions    │
                                  └─────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn (async WebSocket) |
| Speech-to-Text | Groq Whisper API (`whisper-large-v3-turbo`) / `faster-whisper` (local) |
| Language Model | Groq `llama-3.1-8b-instant` |
| Verifier / Judge | OpenAI `gpt-4.1-nano` + OpenAI Moderation API |
| Text-to-Speech | ElevenLabs (`eleven_v3`) |
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim, local CPU) |
| Vector Search | In-memory numpy index + pgvector (HNSW) fallback |
| Database | PostgreSQL (Supabase) + SQLAlchemy 2.0 async + asyncpg |
| Migrations | Alembic |
| Audio Storage | Supabase Storage |
| Audio Processing | librosa, soundfile, scipy, numpy |
| HTTP Client | httpx (async) |
| Deployment | Docker Compose + nginx (TLS) on a VM, deployed via GitHub Actions |

---

## Characters

The API serves three characters from the history of the Faculty of Engineering, Cairo University. Each has a fixed personality, an internal conflict, a unique ElevenLabs voice, and a specific year that bounds their knowledge — anything beyond that year is treated as an anachronism and blocked.

### Morad Ali El-Attar — `s1`
> *Irrigation Engineering student, ~2000. From a wealthy landowning family, raised in Britain. Polite, proud, and ambitious — but overconfident, prone to procrastination, and avoids asking for help.*

"Am I truly capable… or am I only here because of my family name?"

### Hassan Kareem Shawky — `s2`
> *Mechanical Engineering student, ~1960. Top of his class, from a struggling family supported by his father's employer. Observant, responsible, and calm under pressure — but socially awkward and insecure about his finances.*

"If I fail, my family has nothing."

### Amin Saleh El-Shazly — `p1`
> *Mechanical Engineering professor, ~2000. French-trained doctorate, deeply knowledgeable and patient, inspires genuine respect — but expects a lot and can be emotionally distant.*

"Should Egypt follow Europe… or define its own engineering path?"

Each character speaks with knowledge strictly bounded to their own year. Modern concepts, events, or language outside that horizon are never used.

---

## Getting Started

### Prerequisites

- Python 3.11+
- PostgreSQL database with the `pgvector` extension enabled (Supabase recommended)
- API keys for Groq, ElevenLabs, and OpenAI
- A Supabase project with storage buckets for response and question audio

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
OPENAI_API_KEY=sk-proj-...          # used for the verifier LLM + moderation

# --- ElevenLabs Voice IDs (one per character) ---
AHMAD_VOICE_ID=...                  # character s1 (Morad)
KARIM_VOICE_ID=...                  # character s2 (Hassan)
HANAFI_VOICE_ID=...                 # character p1 (Amin)

# --- Database ---
DATABASE_URL=postgresql+asyncpg://user:password@host:5432/postgres

# --- Supabase Storage ---
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=eyJ...
SUPABASE_RESPONSES_BUCKET=response-audios   # archived TTS responses
SUPABASE_QUESTIONS_BUCKET=question-audios   # archived user question audio
```

> **Tip for cloud/VM deployments:** Use the Supabase **Session Pooler** URL (e.g. `aws-0-eu-central-1.pooler.supabase.com`) instead of the direct DB host to avoid DNS resolution failures.

### Database Setup

Make sure `pgvector` is enabled on your PostgreSQL instance, then run all migrations:

```bash
alembic upgrade head
```

This creates:
- `frequently_asked_questions` — FAQ entries with 384-dim embeddings and an HNSW index
- `college_history_chunks` — the history RAG corpus with embeddings
- `past_questions` — interaction log with full timing and verification breakdown

### Running Locally

```bash
make server
# or:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

On startup you will see the embedding model, DB pool, FAQ cache, history cache, and ElevenLabs voices warm up before the pipeline workers start.

---

## WebSocket API

### Connecting

```
ws://localhost:8000/ws/voice-chat
```

Production:

```
wss://immersa-api-voice-chat.run.place/ws/voice-chat
```

All messages are JSON objects with a `type` field.

### Client → Server Events

#### `start_session`
Initialise a session for a specific character.

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
| `sample_rate` | e.g. `16000` | `16000` |
| `audio_format` | `"wav"` · `"pcm16_base64_chunks"` | `"wav"` |

For `"wav"`, the first chunk must carry a valid WAV header (it is cached and re-applied to later headerless batches). For `"pcm16_base64_chunks"`, every chunk is raw signed-16-bit PCM.

#### `audio_chunk`
Send a chunk of recorded audio (base64-encoded). The server runs rolling STT every 5 chunks so transcription starts before end-of-utterance.

```json
{
  "type": "audio_chunk",
  "chunk_index": 0,
  "audio": "<base64-encoded WAV or PCM16 bytes>"
}
```

#### `end_of_utterance`
Signal that the user has finished speaking. Triggers final STT and the full LLM → verification → TTS pipeline.

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
```json
{ "type": "connection_established", "session_id": "abc123", "message": "WebSocket connected successfully" }
```

#### `ack`
Sent after every client message. The `event` field identifies which message is being acknowledged; extra fields depend on the event (e.g. `start_session` returns `state`, `audio_chunk` returns `total_chunks`).
```json
{ "type": "ack", "event": "audio_chunk", "message": "Audio chunk received", "chunk_index": 0, "total_chunks": 1 }
```

#### `final_transcript`
Complete transcription of the user's full utterance (sent before the reply).
```json
{ "type": "final_transcript", "text": "What subjects do you study?" }
```

#### `reply_text_done`
Full reply text from FAQ or LLM, with the chosen emotion. Arrives before any audio so a text bubble can render immediately.
```json
{ "type": "reply_text_done", "text": "We study hydraulics and surveying…", "length": 32, "emotion": "neutral" }
```

#### `tts_audio_chunk`
A base64-encoded WAV (16-bit PCM) audio chunk. Chunks arrive sequentially and should be played in order.
```json
{ "type": "tts_audio_chunk", "chunk_index": 0, "audio": "<base64-encoded WAV>" }
```

#### `tts_done`
All audio chunks for this utterance have been sent. The session returns to `LISTENING`.
```json
{ "type": "tts_done" }
```

#### `error`
```json
{ "type": "error", "message": "STT failed: ..." }
```

---

## REST API

In addition to the WebSocket, the app exposes a small REST surface (FastAPI, auto-docs at `/docs`):

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Liveness message |
| `GET` | `/health` | Health check |
| `GET` | `/faqs` | List FAQs (optional `?character_id=` filter) |
| `PATCH` | `/faqs/{id}` | Update a FAQ (`question`, `answer`, `tag`, `language`, `emotion`) |
| `PATCH` | `/faqs/{id}/emotion` | Set a FAQ's emotion |
| `DELETE` | `/faqs/{id}` | Delete a FAQ |
| `DELETE` | `/faqs` | Delete all FAQs |

Valid emotions: `happy`, `sad`, `angry`, `disgust`, `surprise`, `neutral`.

---

## Pipeline Internals

The pipeline runs as **5 independent async workers** connected by queues. Each stage produces output for the next without blocking the event loop.

```
preprocess_queue → stt_queue → llm_queue → tts_queue → send_queue
```

### Latency optimisations

| Optimisation | Benefit |
|---|---|
| In-memory FAQ & history vector indexes | Numpy dot products (~1 ms), no DB round trip |
| Speculative embedding during STT | Embedding overlaps the remaining STT batches |
| Rolling STT every 5 chunks | Transcription progresses before end-of-utterance |
| TTS sentence pipelining | Next sentence prefetched while current plays |
| Verification in parallel with TTS | Safety checks never block the first audio |
| Noise reduction disabled | Saves 50–200 ms per batch |
| Connection-pool, model, cache & voice warmup | Eliminates first-request cold starts |

### Latency report (printed after each utterance)

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⏱  LATENCY REPORT  —  Apr 25, 2026  03:12:08 PM (Cairo)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Preprocess  (batch 1)  : 0.031s
  STT         (batch 1)  : 0.284s
  FAQ lookup  (miss ❌)  : 0.102s
  RAG (6 chunk(s))       : 0.044s
  LLM                    : 0.521s
  Anachronism            : PASS ✅
  TTS first chunk        : 0.843s
  TTS total              : 2.114s
  ─────────────────────────────────────
  Time to first audio    : 1.261s
  Total                  : 3.403s
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## FAQ System

FAQs are pre-authored question–answer pairs, optionally with a pre-generated voice recording and an emotion tag. When a user's question semantically matches a FAQ entry (cosine similarity ≥ `FAQ_SIMILARITY_THRESHOLD`, default `0.78`), the pipeline:

1. Returns the stored answer text instantly (skipping the LLM)
2. Streams the pre-recorded audio if available — skipping TTS entirely

This is the lowest-latency path.

### FAQ lookup path

```
transcript
    │
    ├─ Exact-match cache? → 0ms (skips embedding too)
    │
    ├─ In-memory vector index (numpy dot product) → ~1ms
    │
    └─ DB pgvector fallback (if cache not loaded) → 50–500ms
```

Manage FAQs with the interactive CLI (`make faq`) or the REST endpoints above.

---

## History RAG

On a FAQ miss, the pipeline grounds the LLM in a curated corpus of Egyptian engineering history (`data/history/*.txt`, indexed into `college_history_chunks`). The top `HISTORY_TOP_K` (default 6) chunks above `HISTORY_SIMILARITY_THRESHOLD` (default 0.4) are retrieved from an in-memory index and injected into the system prompt.

For follow-up questions, a context-aware query is built from the recent conversation turns so a bare "when was it?" still retrieves the right facts. Manage and re-index the corpus with `make history` and `helpers/index_history.py`.

---

## Verification & Safety

Every LLM answer is checked before and during playback, so the user never hears an unsafe or anachronistic reply:

- **Tier 1 — regex (synchronous):** profanity gate on the question and answer, plus an anachronism check that flags any year/term beyond the character's own `operation_year`.
- **Tier 3 — moderation:** OpenAI Moderation API on the question (started in parallel with FAQ lookup) and on the answer.
- **Tier 2/3 — LLM judge:** OpenAI `gpt-4.1-nano` scores the answer for historical accuracy, appropriateness, modern references, and in-character fidelity. It runs **in parallel with TTS** and, on failure, aborts the in-flight audio and either replays a corrected answer or plays the character's verify audio.

Profanity/anachronism hits caught before TTS trigger an immediate regeneration via the verifier LLM; the flagged answer is never played. All verification outcomes and timings are logged to `past_questions`.

---

## Helper Scripts

Helper tooling lives in `helpers/` and is exposed via the `Makefile`:

| Command | Script | Purpose |
|---|---|---|
| `make faq` | `helpers/faq_manager.py` | Interactive FAQ management (add, list, update, fill audio/embeddings) |
| `make history` | `helpers/history_manager.py` | Manage the history RAG corpus |
| `make audio` | `helpers/generate_audio_from_text.py` | Generate ElevenLabs audio from text |
| `make test` | `helpers/test_script.py` | End-to-end WebSocket test client |
| `make server` | — | Run the API locally with `--reload` |

`helpers/index_history.py` (re)builds the history embeddings, and `helpers/test_files/` holds sample WAVs of varying lengths for testing.

---

## Deployment

The API is containerised and deployed with **Docker Compose** behind **nginx** (TLS via Let's Encrypt) on a VM. Pushing to the `production` branch triggers a GitHub Actions workflow (`.github/workflows/deploy.yml`) that SSHes into the VM and runs `docker compose up -d --build`.

### Services (`docker-compose.yml`)
- `api` — the FastAPI/Uvicorn app on port 8000, with a `/health` healthcheck and cached HuggingFace/Whisper model volumes
- `nginx` — terminates TLS on 80/443 and proxies WebSocket traffic to the API (`nginx.conf`)

### Required environment variables

```
GROQ_API_KEY
ELEVENLABS_API_KEY
OPENAI_API_KEY
DATABASE_URL            (use Supabase Session Pooler URL)
SUPABASE_URL
SUPABASE_SERVICE_KEY
SUPABASE_RESPONSES_BUCKET
SUPABASE_QUESTIONS_BUCKET
AHMAD_VOICE_ID          (s1)
KARIM_VOICE_ID          (s2)
HANAFI_VOICE_ID         (p1)
```

### STT in production

Set `stt_provider = "groq"` in `app/core/config.py` (default). Local Whisper requires a GPU or a significantly larger container.

### Supabase storage buckets

Create two **public** buckets in your Supabase project:
- `response-audios` — archived TTS audio from every interaction
- `question-audios` — archived user question audio

---

## Project Structure

```
IMMERSA-Voice-Chat-API/
├── app/
│   ├── main.py                        # FastAPI app, lifespan startup/shutdown
│   ├── api/
│   │   ├── websocket_routes.py        # WebSocket /ws/voice-chat endpoint
│   │   └── faq_routes.py              # REST /faqs endpoints
│   ├── characters/
│   │   ├── build_prompt.py            # Prompt assembly engine
│   │   ├── characters_info.py         # Character metadata & voice IDs
│   │   └── prompts.py                 # System & user prompt templates
│   ├── core/
│   │   ├── clients.py                 # AI client initialisation (Groq, ElevenLabs, OpenAI)
│   │   └── config.py                  # All configuration constants & env vars
│   ├── db/
│   │   ├── database.py                # Async engine & session factory
│   │   ├── models.py                  # ORM models: FAQ, PastQuestion, HistoryChunk
│   │   └── repositories/              # FAQ, history & past-questions data access
│   ├── services/
│   │   ├── audio/preprocessor.py      # WAV rebuild, high-pass filter, normalise
│   │   ├── embedding_service.py       # BAAI/bge-small-en-v1.5 embeddings (local CPU)
│   │   ├── faq_memory_cache.py        # In-memory FAQ vector index
│   │   ├── history_memory_cache.py    # In-memory history (RAG) vector index
│   │   ├── llm/                       # Groq + OpenAI LLM services
│   │   ├── pipeline/pipeline.py       # 5-stage async queue pipeline
│   │   ├── streaming/                 # WS session, connection manager, event protocol
│   │   ├── stt/                       # Groq Whisper + faster-whisper
│   │   ├── tts/elevenlabs_service.py  # ElevenLabs streaming TTS
│   │   └── verification/              # Tiered regex / moderation / LLM-judge checks
│   └── utils/
├── alembic/versions/                  # Database migration history
├── helpers/                           # FAQ/history managers, audio gen, test client
│   └── test_files/                    # Sample WAV files
├── data/
│   ├── history/                       # History RAG corpus (.txt)
│   ├── fallback_audios/               # Per-character fallback WAVs
│   ├── verification_audios/           # Per-character verifier-reject WAVs
│   └── thinking_audios/               # Per-character "thinking" WAVs
├── docs/index.html                    # Standalone WebSocket API documentation
├── Dockerfile
├── docker-compose.yml
├── nginx.conf
├── Makefile
├── alembic.ini
└── requirements.txt
```

---

## Contact

**Mariam Elsoufyx** — mariamelsoufyx@gmail.com

---

<p align="center">Built for the IMMERSA immersive experience &mdash; Faculty of Engineering, Cairo University</p>
