# Chapter 4: System Design and Architecture

This chapter presents the design and implementation of the IMMERSA conversational backend — the cloud service that lets a user hold a spoken, real-time conversation with a historical character. It begins with the overall architecture and the reasoning behind it, then devotes a separate section to each of the three modules that make up the backend. For every module the discussion covers not only what the module does and how it is built, but also the scientific and engineering principles it relies on, so that the design choices can be understood rather than merely listed.

## 4.1 Scope and Design Goals

The backend has a single, demanding purpose: to receive a user's spoken question, understand it, produce a historically accurate and in-character answer, and speak that answer back — all fast enough that the exchange feels like a natural conversation rather than a series of queries. In practice the target is to begin playing the reply within roughly one second of the user finishing speaking, because delays beyond about a second are perceived as unnatural pauses in human dialogue.

Two decisions distinguish the implemented system from the original proposal. First, the standalone mobile application was removed entirely; the experience is now delivered hands-free through a Meta Quest 3s headset application, keeping the user immersed in the augmented scene. Second, all of the heavy work — speech recognition, semantic search, language generation, content verification, and speech synthesis — was consolidated into one cloud-deployed backend. The headset client is deliberately thin: it captures microphone audio and plays returned audio, nothing more. This separation means the system's intelligence can be improved, corrected, and scaled centrally without ever shipping a software update to the wearable device.

**Primary design goals.** The architecture is shaped by four goals that recur throughout this chapter: (1) low end-to-end latency; (2) historical accuracy and safety of every spoken reply; (3) believable, consistent characters; and (4) robustness, so that a failure in any single component never leaves the user in silence.

## 4.2 Backend System Architecture

### 4.2.1 Block Diagrams

The backend architecture is presented at two levels of detail. Figure 4.1 gives the high-level structure of the system and its main building blocks, while Figure 4.2 expands that structure into the full processing flow of a single voice interaction.

> *[ Insert Figure 4.1 — high-level backend architecture diagram ]*

*Figure 4.1: High-level backend architecture, showing the wearable client, the cloud backend pipeline, and the external services it depends on.*

As shown in Figure 4.1, the system is organised into three tiers. The first is the **Meta Quest 3s client**, a deliberately thin application that only captures the user's microphone audio and plays back the character's speech; it communicates with the server through a single **WebSocket** connection, over which audio chunks stream up to the backend and reply audio streams back down. The second is the **cloud backend**, built with FastAPI, which holds the five-stage processing pipeline (*audio preprocessing → speech-to-text → retrieve-and-generate → text-to-speech → stream*). The third is the layer of **external services and data stores** the backend depends on: Groq for hosted Whisper speech-to-text, a large language model (Groq, with OpenAI as a verifier) for reply generation, ElevenLabs for speech synthesis, and a PostgreSQL database extended with pgvector for the knowledge base and storage.

> *[ Insert Figure 4.2 — detailed backend flowchart ]*

*Figure 4.2: Detailed block diagram of the cloud backend, showing the flow of a single voice interaction from received audio to streamed reply.*

Figure 4.2 details how a single utterance moves through the backend. The user's audio enters at *received audio preprocessing*, is cleaned, and is passed to *speech-to-text processing*, which produces the *detected text*. This text reaches the *FAQ Match?* decision, the central branch of the pipeline. On a **match (yes)**, the stored answer is sent directly to *generated audio streaming*, bypassing both language models and speech synthesis. On a **miss (no)**, the relevant historical passages (*related RAG chunks*) are retrieved from the *RAG cache* and, together with the detected text, passed to the *first LLM for reply generation*; the *generated reply* is then synthesised by *text-to-speech processing* and streamed to the user. While the reply streams, a *second LLM for verification* checks it in parallel and may let it play, *block the streaming of the original reply*, or *send a new generated reply to TTS* in its place. Every interaction is consolidated into a *fully generated response* — the detected text, the reply, the synthesised audio, and the verification report — and written to the *database (DB)*, which is also the store from which the *FAQ cache* and *RAG cache* are loaded at start-up.

### 4.2.2 Architectural Overview

The backend is an asynchronous Python service built on the FastAPI framework and served by the Uvicorn ASGI server. The work of turning speech into a spoken reply is organised as a five-stage pipeline whose stages — Preprocessing, Speech-to-Text, Retrieval and Generation, Text-to-Speech, and Streaming — run as independent asynchronous workers connected by in-memory queues. Because the stages operate concurrently, the pipeline never blocks: a slow external call in one stage does not stall the others, and multiple phases of work overlap in time (for example, one utterance can be synthesised into speech while the next is still being transcribed). The three modules in Sections 4.3–4.5 own these stages together with their supporting services.

Concretely, the service runs on a single thread driven by an asynchronous event loop. Rather than blocking the whole program while waiting for a network reply, an asynchronous function yields control back to the event loop at each waiting point; the loop then runs other ready work and resumes the function when its data arrives. This cooperative multitasking lets one thread serve many concurrent connections, provided no single operation monopolises the CPU. The five stages are joined in the classic producer–consumer pattern — each takes an item from its input queue, processes it, and places the result on the next queue:

```
preprocess_queue → stt_queue → llm_queue → tts_queue → send_queue
```

Operations that are unavoidably CPU-bound or that use blocking libraries (audio filtering, the local embedding model, audio decoding) are pushed onto a separate thread pool, so the event loop stays responsive while the heavy computation proceeds elsewhere. End to end, a single utterance therefore traverses the system as follows:

1. The client streams chunks of recorded microphone audio to the backend over the WebSocket connection and, when the user stops talking, sends an end-of-utterance signal.
2. Stage 1 (Preprocessing) cleans each audio chunk; Stage 2 (Speech-to-Text) transcribes it into text.
3. Stage 3 (Retrieval and Generation) decides what to say: it first checks the FAQ knowledge base for a ready-made answer and, on a miss, retrieves relevant historical context and asks the language model to generate a reply.
4. Stage 4 (Text-to-Speech) synthesises the reply into speech, sentence by sentence.
5. Stage 5 (Streaming) sends the audio back to the client in order; in parallel, the verification layer checks the reply and can correct or replace it mid-stream, and a background task archives the interaction.

All communication between the client and the backend uses a single WebSocket connection at the endpoint `/ws/voice-chat`. A WebSocket differs fundamentally from an ordinary web (HTTP) request: instead of a one-shot request-and-response that closes immediately, it establishes a single, long-lived, full-duplex channel over which either side may send messages at any time. This is essential here for two reasons. First, the user's speech is an open-ended stream of audio chunks rather than one fixed payload, and the reply is likewise streamed back as it is produced. Second, a persistent connection avoids the repeated handshake cost of opening a new connection per message, which would add latency to every exchange. The protocol is event-based: small JSON messages name an event type and carry its data. Table 4.1 lists the principal events.

| Event | Direction | Purpose |
|---|---|---|
| `start_session` | Client → Server | Opens a session, selecting the character (s1, s2, p1) and audio format. |
| `audio_chunk` | Client → Server | A base64-encoded slice of microphone audio, sent continuously while the user speaks. |
| `end_of_utterance` | Client → Server | Signals the user has stopped; triggers full pipeline execution. |
| `final_transcript` | Server → Client | The complete recognized text of the question. |
| `reply_text_done` | Server → Client | The reply text and an emotional-tone label, sent before any audio. |
| `tts_audio_chunk` | Server → Client | An ordered chunk of synthesized reply audio for playback. |
| `tts_done` | Server → Client | Marks the end of the reply; the session returns to listening. |

*Table 4.1: Principal WebSocket events of the voice-chat protocol.*

Table 4.2 summarizes the technologies used across the backend and the role of each.

| Layer | Technology |
|---|---|
| API framework / server | FastAPI application run by the Uvicorn ASGI server (asynchronous WebSockets) |
| Speech-to-text | Groq-hosted Whisper (`whisper-large-v3-turbo`); faster-whisper available on-device |
| Language model | Groq `llama-3.1-8b-instant` (an OpenAI-based implementation is also available) |
| Embeddings | `BAAI/bge-small-en-v1.5`, 384-dimensional, run locally on CPU |
| Vector search | pgvector (HNSW index) in the database; an in-memory NumPy index in the hot path |
| Verification | OpenAI `omni-moderation-latest` classifier and a `gpt-4.1-nano` LLM judge |
| Text-to-speech | ElevenLabs neural voice synthesis (`eleven_v3`) |
| Database / ORM | PostgreSQL on Supabase via SQLAlchemy (async) and the asyncpg driver |
| Object storage | Supabase Storage for archived question and response audio |
| Deployment | Single cloud-hosted service configured entirely through environment variables |

*Table 4.2: Backend technology stack.*

## 4.3 Module 1: Audio Processing and Generation

This module is the system's ear and voice. It owns the first two pipeline stages — preparing and transcribing the user's speech — and the fourth stage, synthesizing the character's reply. Because everything the user hears and everything the system understands passes through it, its design directly determines both recognition accuracy and the naturalness of the delivered voice.

### 4.3.1 Functional Description

This module is responsible for everything related to sound: it turns the user's spoken question into clean text the rest of the system can process, and it turns the character's textual reply back into expressive, natural speech. Its overriding goal is to deliver high recognition accuracy and convincing voice output — with accurate reply content — while keeping the delay relatively short. Concretely, it owns the first two stages of the pipeline (audio preprocessing and speech-to-text) and the fourth stage (text-to-speech).

### 4.3.2 Modular Decomposition

The module is composed of three components. Each is described below together with the science that underpins it.

#### 4.3.2.1 Audio Preprocessing

Reconstructs each incoming chunk into a valid 16 kHz waveform, applies a high-pass filter to remove low-frequency rumble, trims leading and trailing silence, and normalises the signal amplitude before recognition.

- **Digital audio and the 16 kHz choice.** Sound is a continuous pressure wave, but a computer can store only numbers, so the wave is *sampled* — its amplitude is measured at regular intervals. The Nyquist–Shannon sampling theorem states that to represent frequencies up to *F* Hz, a signal must be sampled at least at 2*F*. Intelligible speech lies almost entirely below 8 kHz, so a 16 kHz sample rate captures it faithfully while using a fraction of the data of music-grade rates such as 44.1 kHz; it is also the rate the recogniser expects. All audio is therefore standardised to 16 kHz, mono, 16-bit PCM.
- **High-pass filtering.** A fourth-order Butterworth high-pass filter with an 80 Hz cut-off removes energy below the human voice. The Butterworth design is chosen for its maximally flat passband, so it does not colour the voice; the 80 Hz cut-off sits beneath the lowest vocal tones but above handling noise, footsteps, and microphone rumble. Zero-phase (forward–backward) filtering avoids introducing any time shift.
- **Silence trimming.** Leading and trailing silence is removed by comparing the signal's loudness, in decibels relative to its peak, against a threshold (20 dB below peak). This shortens the clip sent to the recogniser and removes ambiguous near-silence that can cause spurious transcriptions.
- **Amplitude normalisation.** The waveform is peak-normalised so its loudest point reaches full scale, compensating for users who speak softly or far from the microphone and presenting the recogniser with a consistent signal level.

#### 4.3.2.2 Speech-to-Text (STT)

Transcribes the cleaned audio. It runs on a rolling basis as audio arrives — rather than only after the user stops — so transcription is well advanced by the end of the utterance.

- **How modern recognition works.** The system uses OpenAI's Whisper, a transformer encoder–decoder trained on hundreds of thousands of hours of audio. The audio is first turned into a *log-mel spectrogram* — a time–frequency image scaled to mimic human pitch and loudness perception. The encoder converts this into a rich numerical representation, and the decoder then emits the transcript one token at a time, attending at each step to the encoded audio and the words produced so far. Training on vast, diverse, noisy data is exactly why Whisper is so robust to accents and background noise.
- **Why Groq, and rolling recognition.** The chosen model, `whisper-large-v3-turbo`, is a distilled, speed-optimised variant of Whisper's largest model, served on Groq's low-latency inference hardware and typically returning in a few hundred milliseconds. Recognition is performed in rolling batches as audio arrives, so it is nearly complete by the time the end-of-utterance signal is received.

#### 4.3.2.3 Text-to-Speech (TTS)

Synthesises the reply in the character's voice. The reply is split into sentences and synthesised sentence by sentence, so playback of the first sentence begins while later sentences are still being generated.

- **Neural synthesis.** ElevenLabs generates the speech waveform directly from text using deep models, predicting natural prosody — rhythm, stress, and intonation — and reproducing a specific voice identity. This is far more natural than older concatenative methods that stitched together pre-recorded fragments.
- **Sentence-level pipelining.** Synthesising a whole paragraph before playing any of it would force the user to wait for the longest part of the reply. Splitting the reply into sentences and pre-fetching the next while the current one plays means playback starts as soon as the first sentence is ready, and the rest follow seamlessly.

### 4.3.3 Design Constraints

- **Real-time operation:** every transformation is chosen for its latency-to-quality ratio, and all blocking work runs in background threads so the server's event loop is never stalled.
- **Noise reduction deliberately disabled:** aggressive spectral noise reduction is switched off — it was measured to add 50–200 ms per chunk for a negligible accuracy gain on the modern hosted recogniser, and the Meta Quest 3s microphone is clean enough that recognition already performs well without it.
- **Strict first-chunk timeout:** if a slow synthesis call exceeds the allotted time, a pre-recorded fallback clip is played instead, so the user never hears silence.
- **Streamed, not file-based:** audio is handled as a continuous stream rather than a complete file, so recognition and playback can both begin before the corresponding audio is complete.

### 4.3.4 Other Description of Module 1

By default the module uses Groq's hosted Whisper model (`whisper-large-v3-turbo`) for recognition, chosen for its accuracy and very low response time; an on-device alternative based on faster-whisper is also supported for offline or cost-sensitive deployments. Speech synthesis is provided by ElevenLabs, with a distinct voice assigned to each character and voice settings tuned for expressive delivery. A particularly important optimisation lives here: as soon as the first partial transcript is available, the module speculatively begins computing the semantic embedding of the question (used later for knowledge retrieval in Module 2) in parallel with the remaining recognition, so that vector is usually ready the instant it is needed. The expressiveness of the delivered voice is governed by each character's chosen voice together with a fixed set of voice settings; the emotional-tone label produced during response generation (Module 2) does not alter synthesis — it is forwarded to the client to choose the suitable facial expression and recorded in the interaction log.

## 4.4 Module 2: Knowledge Retrieval, Response Generation, and Verification

This module is the conversational core — the "brain" of the system. Taking the transcribed question from Module 1, it decides what the character should say, generates the reply in character, and verifies that reply for accuracy and safety, all before and during playback. It brings together what were originally three concerns — knowledge retrieval, response generation, and content verification — because they form one continuous decision: *where the answer comes from, how it is phrased, and whether it is allowed to be spoken.*

### 4.4.1 Functional Description

The module produces every spoken answer and guarantees that answer is fast, historically accurate, in character, and safe. It performs three tightly coupled jobs:

- **Deciding the source of truth.** It prefers curated knowledge over free generation: first it looks for a ready-made answer to a frequently asked question, and, failing that, it retrieves the most relevant verified history passages to ground the language model.
- **Generating the reply.** When no ready-made answer exists, it assembles a detailed prompt from the character's identity and the retrieved history, asks a large language model for a reply, and keeps that reply consistent with the character's personality and the recent flow of conversation.
- **Verifying the reply.** It checks every generated reply for historical accuracy and appropriateness in parallel with speech synthesis, correcting or replacing it mid-stream if necessary.

All three rest on a shared foundation of semantic vector search, so a user's question can be matched to stored knowledge even when the wording differs.

### 4.4.2 Modular Decomposition

The module is organised into three sub-systems — Knowledge Retrieval, Response Generation, and Content Verification — described in turn.

#### 4.4.2.1 Knowledge Retrieval (FAQ and RAG)

Built on a single foundation, semantic vector search, whose mathematics is described under the Embedding Service.

- **Embedding Service.** Represents text as 384-dimensional semantic vectors using the `BAAI/bge-small-en-v1.5` model, run locally on the server CPU. *What an embedding is:* comparing questions by their words fails immediately — "What do you study?" and "Which subjects are you taking?" share almost no words yet mean the same thing. An embedding model is a neural network that maps text to a fixed-length vector positioned so that texts with similar meaning land close together; it is trained by *contrastive learning* (shown many sentence pairs, it learns to place paraphrases near each other and unrelated sentences far apart). The model runs in roughly 60–80 ms and is *asymmetric* — queries get a short instruction prefix that stored documents do not, improving match quality. *Cosine similarity:* closeness is the cosine of the angle between two vectors (1 = same meaning, 0 = unrelated, −1 = opposite); because every vector is normalised to unit length, this reduces to a simple dot product, so searching all stored vectors is one matrix–vector multiplication — well under a millisecond.
- **FAQ Store and In-Memory Index.** Holds pre-authored question–answer pairs, each with an embedding and, where available, a pre-recorded audio clip. A question is resolved cheapest-first: (1) an *exact-match cache* returns the stored result instantly for a verbatim repeat, skipping embedding and search; (2) otherwise the question is embedded and compared against the in-memory index, accepting the best match only above a high similarity threshold of **0.8**; (3) if the in-memory index is unavailable, the same search runs against the database via pgvector as a correct fallback.
- **RAG History Retrieval.** On an FAQ miss, retrieves the most relevant verified history passages for the active character to ground the model. *Why grounding matters:* a model asked about specifics it was never reliably taught may *hallucinate* — produce a fluent but false answer; Retrieval-Augmented Generation (RAG) prevents this by inserting factual passages into the prompt so the model phrases an answer drawn from supplied facts. The verified history is split into short *chunks*; a miss retrieves up to the top **six** that clear a looser threshold of **0.4**, combining character-scoped chunks with globally shared history. The threshold is looser than the FAQ's because retrieval only needs to surface useful context, whereas an FAQ match must be near-certain before it replaces generation.

#### 4.4.2.2 Response Generation and Character Dialogue

Produces the reply when no FAQ answer exists, in the voice of a specific, believable person rather than a generic assistant.

- **Prompt Builder.** Assembles a *system prompt* from the character's biography, traits, courses, possessions, and internal conflicts, combined with the retrieved history and the user's question. A model's behaviour is governed almost entirely by its prompt, so this is where the character lives: the system prompt fixes the persona, the operating year, the injected history as the authoritative source of facts, and the rule to stay in character and within the period, while a *user prompt* carries the transcribed question. Personas are templates with placeholders filled at request time, so all characters share one tested structure while differing in content. The model must reply as a **JSON object**, so one response carries both the answer text and an emotional-tone label; a tolerant parser recovers if the model ever wraps the JSON in extra text.
- **Language-Model Service.** Sends the prompt to a large language model (Groq `llama-3.1-8b-instant`) and returns the reply with its emotional-tone label. *How an LLM works:* a transformer network trained to predict the next *token* given those before it, generating *autoregressively*; its *self-attention* mechanism weighs the relevance of every earlier token, producing coherent, on-topic text. The eight-billion-parameter model is served on Groq's low-latency hardware, with the reply consumed as a stream of tokens.
- **Conversation Memory.** Retains the connection's most recent exchanges (by default the last six question–answer pairs) so the character can resolve pronouns and follow-ups. The window is kept small because a model can consider only a limited context, longer prompts cost more and run slower, and old turns rarely matter. This same memory feeds the context-aware retrieval query (see Other Description) and is kept in step with what the user hears — if verification replaces a reply, the stored turn is updated to match.

#### 4.4.2.3 Content Verification and Safety

Protects both historical authenticity and appropriateness. Its defining property is that it runs *in parallel* with speech synthesis rather than before it: the reply begins playing immediately, and if verification later rejects it, the audio is cut and corrected — preserving responsiveness without sacrificing safety. The checks are layered in three tiers of increasing cost, plus a handler.

- **Tier 1 — Regex Pattern Checks.** Local regular-expression scans under a millisecond, fast enough to gate the pipeline before any expensive work. A *profanity* filter screens question and answer against a curated word-list (English and transliterated Arabic), and an *anachronism* check flags anything that breaks the historical illusion: any year later than the character's own year of operation, modern terms and brand names, web addresses, and e-mail addresses. Because the cut-off is per-character, the same check enforces a 2000 cut-off for Morad and Amin and a 1960 cut-off for Hassan.
- **Tier 2 — Neural Content Moderation.** A dedicated moderation model (OpenAI's `omni-moderation-latest`), a multilingual classifier returning category signals (hate, sexual, self-harm, violence), catches unsafe content phrased without blacklisted words. The question's moderation is started early — in parallel with the FAQ lookup — so its latency is hidden, and the answer's moderation runs alongside synthesis.
- **Tier 3 — LLM-as-Judge.** A second, separate model (`gpt-4.1-nano`) scores the reply on four dimensions — historical accuracy, appropriateness, presence of modern references, and staying in character — and may supply a *corrected answer*. The overall verdict is recomputed deterministically from these dimensions rather than trusting the judge's own summary. Evaluating a candidate against explicit criteria is easier and more reliable for a model than generating the answer, and using an independent model reduces the chance that a single model's blind spot slips through.
- **Fallback and Correction Handler.** Decides the graceful response when a check fails: replaying a corrected answer in the character's voice, or streaming a pre-recorded, character-specific fallback clip. The same mechanism also covers any technical failure or timeout elsewhere in the pipeline.

### 4.4.3 Design Constraints

- **Sub-millisecond common path:** retrieval lookups must complete in well under a millisecond, which is why both the FAQ and history indexes are held in memory rather than queried from the database on every question.
- **Purpose-tuned thresholds:** a high threshold (0.8) governs FAQ matches to avoid answering the wrong stored question, while a looser threshold (0.4) governs history retrieval so useful context is still found on partial matches.
- **Period-bound knowledge:** every character is strictly bound to the knowledge of its year of operation; modern concepts, events, and language must never appear, enforced by the per-character anachronism cut-off.
- **Structured reply with emotion:** replies carry both the answer text and an emotional-tone label; the label is delivered to the client (to select a facial expression) and logged, while speech synthesis itself uses each character's fixed voice configuration.
- **Bounded memory:** conversation memory is limited to a few recent turns to preserve coherence without inflating prompt cost or latency.
- **Verification overlaps, never gates:** verification must run in parallel with playback; it may abort an in-progress stream but must never delay its start, and checks run cheapest-first so the costly judge runs only when needed.
- **Always a coherent reply:** every failure path must end in a correction or a fallback clip — never silence or a raw error — and a rejected reply is removed from conversation memory so it cannot influence later turns.
- **Fail-open model checks:** if a moderation or judge service errors or times out, the check passes by default, so an external outage cannot block a legitimate reply; the always-local Tier 1 checks provide the guaranteed floor of protection.
- **Database as fallback:** the database (with its pgvector index) must remain a correct fallback for both indexes should the in-memory copies be unavailable.

### 4.4.4 Other Description of Module 2

**The characters.** The system currently models three figures from Cairo University's Faculty of Engineering, set across 1960 and 2000, summarised in Table 4.3. Each is defined not only by biographical facts but by a central internal tension that gives the conversation emotional depth. Because the characters belong to different eras, each carries its own *year of operation*, which also sets the cut-off used by the anachronism check.

| ID | Character | Year of operation | Role and defining tension |
|---|---|---|---|
| s1 | Morad Ali El-Attar | 2000 | Wealthy irrigation-engineering student, raised in Britain; polite and ambitious but overconfident and prone to procrastination — doubts whether he earned his place or owes it to his family name. |
| s2 | Hassan Kareem Shawky | 1960 | Top mechanical-engineering student from a wealthy family; observant, responsible, and calm under pressure but socially awkward — driven to excel and make his father proud. |
| p1 | Amin Saleh El-Shazly | 2000 | Mechanical-engineering professor, French-trained; deeply knowledgeable, patient, and respected — weighing whether Egypt should follow Europe or define its own engineering path. |

*Table 4.3: The characters served by the backend.*

**Follow-up questions.** A bare query such as "when was it built?" carries no subject on its own and would retrieve nothing useful. The module solves this by constructing a context-aware retrieval query that prepends the user's recent questions, restoring the missing subject so retrieval returns the passages the follow-up actually depends on.

**Lowest-latency path.** When a FAQ entry has an associated pre-recorded audio clip, a hit streams the clip directly and skips both the language model and speech synthesis entirely, making it the fastest possible response in the whole system.

**Storage and indexing (pgvector + HNSW).** In the database, embeddings are stored using the pgvector extension, which adds a native vector type and similarity operators to PostgreSQL. For large collections, comparing a query against every stored vector becomes slow, so pgvector builds an HNSW (Hierarchical Navigable Small World) index — an approximate-nearest-neighbour graph that reaches a query's neighbours in roughly logarithmic time, trading a tiny amount of accuracy for a large speed gain. In normal operation, however, the database is only the source of truth and the fallback path; live lookups are served from the in-memory index.

**Correction over rejection.** When the judge supplies a corrected answer, the system plays a brief verification clip and then streams the corrected reply in the character's voice, updating the conversation memory so that what the user heard and what the character "remembers" stay consistent. The combination of fast local gates that always run, neural checks that run in parallel and fail open, and a single fallback path for every failure lets the system stay both safe and responsive at once.

## 4.5 Module 3: Real-Time Communication and Pipeline Orchestration

This module is the backbone that ties the other two together. It manages the client connection, coordinates the flow of work through the five pipeline stages, streams the reply back in order, and records every interaction. It is what makes the multi-step processing of the other modules feel to the user like a single, instant conversation.

### 4.5.1 Functional Description

The module owns the live connection to the headset and the coordination of all the work that happens between receiving audio and returning speech. It keeps each conversation's state, moves work between the pipeline stages in the right order and at the right time, streams the synthesised reply back chunk by chunk, and — once the user has been served — records the interaction for later analysis.

### 4.5.2 Modular Decomposition

The module comprises three components.

#### 4.5.2.1 WebSocket Communication Layer

Maintains the persistent, full-duplex connection, tracks per-session state, and serialises the event protocol of Table 4.1. Each session holds the short-lived state for one connection: the chosen character, the audio format, a buffer accumulating the incoming audio chunks, the in-progress transcript, and the conversation memory.

#### 4.5.2.2 Pipeline Orchestrator

Runs the five stage-workers as independent coroutines joined by in-memory queues and coordinates the subtle timing the system depends on: launching the speculative embedding, starting question moderation in parallel with the FAQ lookup, releasing the reply to synthesis while spawning verification as a concurrent task, and signalling completion only once both synthesis and verification have settled. This careful overlapping of independent work is the practical mechanism behind the system's low latency.

#### 4.5.2.3 Persistence and Logging

After each reply is delivered, a background task (off the response path) archives the user's question audio and the response audio to object storage and writes a detailed row to the interaction log.

### 4.5.3 Design Constraints

- **Full-duplex, low-latency:** communication carries audio in both directions over a single persistent connection.
- **Decoupled stages:** pipeline stages must be non-blocking so external-service latency in one stage cannot stall the rest.
- **Off-path persistence:** archival and logging must run outside the response path so they add nothing to perceived latency.
- **Clean per-utterance reset:** per-connection state must reset cleanly between utterances while preserving multi-turn memory.

### 4.5.4 Other Description of Module 3

Each logged interaction captures the question and answer text, whether the answer came from the FAQ store or the model, the emotional-tone label, the result of every verification check (including any corrected answer), and a full per-stage timing breakdown. These timings are the evidence base for the latency analysis in Section 4.6, and the archived audio supports later review and dataset building.

## 4.6 Cross-Cutting Concern: Latency Engineering

Sub-second responsiveness is not the product of one trick but of a discipline applied at every stage, all aimed at the same idea: do work ahead of time, avoid repeating work, and overlap independent work so that waiting is shared rather than summed. The total response time is bounded by the longest single chain of dependent steps, not by the sum of all work, so the engineering goal is to move as much as possible off that critical chain. Table 4.4 collects the principal techniques used across the modules and the scientific idea behind each.

| Technique | Principle and benefit |
|---|---|
| In-memory vector indexes | Caching: keep the working set in RAM so searches are sub-millisecond NumPy operations instead of network database queries. |
| Exact-match question cache | Memoization: store the result of an exact question so repeats skip embedding and search entirely. |
| Speculative embedding | Speculative execution: start the question's embedding during recognition, before it is known to be needed, removing it from the critical path. |
| Rolling speech recognition | Pipelining: transcribe audio as it arrives so recognition is nearly done when the user stops speaking. |
| Sentence-level TTS pipelining | Pipelining: begin playback on the first sentence while later sentences are still being synthesised. |
| Parallel verification | Concurrency: overlap safety checks with playback instead of gating playback on them. |
| Parallel question moderation | Concurrency: run moderation during the FAQ lookup so its latency is hidden on the common path. |
| Disabled spectral noise reduction | Cost–benefit pruning: remove a 50–200 ms step that yielded negligible accuracy gain. |
| Start-up warm-up | Cold-start elimination: pre-load models, open database connections, and fill caches before the first request. |

*Table 4.4: Latency-engineering techniques and the principle behind each.*

## 4.7 Deployment and Infrastructure

The backend is deployed as a single cloud service. All credentials and tunable parameters are supplied through environment variables, so no secret is embedded in the source and the same code runs unchanged across local, staging, and production environments. The PostgreSQL database and the object storage both run on Supabase; the database schema is managed through versioned migrations, and two storage buckets hold the archived question and response audio. At start-up the service performs an explicit warm-up sequence — it loads the embedding model and runs a first inference, opens and primes a pool of database connections, warms the speech and synthesis services, and loads the FAQ and history indexes into memory — so that the very first user request is served at full speed rather than paying cold-start costs. Paired with the thin headset client, this cloud-only deployment allows the entire experience to be improved and scaled centrally with no update to the wearable device.

## 4.8 Summary

The IMMERSA backend realises the voice and character-interaction capabilities of the wider system as a single, cloud-deployed, real-time speech service. A concurrent five-stage pipeline turns speech into a spoken reply; a knowledge-and-reasoning layer built on semantic vector search keeps that reply fast and historically grounded; a parallel, tiered verification layer keeps it accurate and safe without sacrificing responsiveness; and an orchestration layer binds these together behind a single streaming connection to a deliberately thin wearable client. Every major design choice described in this chapter — the asynchronous pipeline, in-memory retrieval, retrieval-augmented generation, parallel verification, and pervasive latency engineering — serves the same end: immersive, in-character, historically faithful conversation delivered at the speed of natural human dialogue.
