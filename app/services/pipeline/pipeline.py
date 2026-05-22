import asyncio
import base64
import io
import json
import os
import queue as thread_queue
import re
import threading
import time

import numpy as np
import soundfile as sf


import httpx

import app.core.config as config
from app.characters.build_prompt import build_narrator_prompts
from app.db.repositories.past_questions_repository import create_past_question
from app.services.streaming.event_protocol import (
    build_error_event,
    build_final_transcript_event,
    build_reply_text_done_event,
    build_tts_audio_chunk_event,
    build_tts_done_event,
)
from app.services.verification import Verifier
from app.services.embedding_service import generate_query_embedding
from app.db.repositories.faq_repository import search_similar_faq
from app.db.repositories.history_repository import search_history_chunks
from app.utils.log import log


class Pipeline:
    def __init__(self, connection_manager, audio_preprocessor, stt_service, llm_service, elevenlabs_service, db_session_factory=None, faq_memory_cache=None, openai_client=None):
        self.connection_manager = connection_manager
        self.audio_preprocessor = audio_preprocessor
        self.stt_service = stt_service
        self.llm_service = llm_service
        self.elevenlabs_service = elevenlabs_service
        self.db_session_factory = db_session_factory  # async session factory for past_questions writes
        self.faq_memory_cache = faq_memory_cache       # in-memory FAQ index (no DB hit for lookups)
        self.openai_client = openai_client             # OpenAI client for parallel response verification
        self.verifier = Verifier(openai_client=openai_client)  # tiered regex / models orchestrator

        self.preprocess_queue: asyncio.Queue = asyncio.Queue()
        self.stt_queue: asyncio.Queue = asyncio.Queue()
        self.llm_queue: asyncio.Queue = asyncio.Queue()
        self.tts_queue: asyncio.Queue = asyncio.Queue()
        self.send_queue: asyncio.Queue = asyncio.Queue()

        # session_id → collected timings for this utterance
        self._timings: dict[str, dict] = {}

        # Sessions whose TTS stream should be cut — verifier flagged them mid-stream
        self._verify_abort: set[str] = set()

        # Per-session event fired by _tts_worker when streaming ends (normally or aborted)
        # _run_verification awaits this before sending the final done/verify signal
        self._tts_done_events: dict[str, asyncio.Event] = {}

        # Sessions where TTS failed with a real error — _run_verification skips them
        self._tts_error_sessions: set[str] = set()

        # In-memory FAQ cache: (normalized_transcript, character_id) → FAQ | None
        # Skips embedding + DB entirely for repeated/identical questions
        self._faq_cache: dict[tuple[str, str], object] = {}

    def start(self):
        asyncio.create_task(self._preprocess_worker())
        asyncio.create_task(self._stt_worker())
        asyncio.create_task(self._llm_worker())
        asyncio.create_task(self._tts_worker())
        asyncio.create_task(self._send_worker())
        log.ok("PIPE", "all 5 workers started (preprocess → stt → llm → tts → send)")

    def _t(self, session_id: str) -> dict:
        """Return (or create) the timings dict for a session."""
        if session_id not in self._timings:
            self._timings[session_id] = {
                "preprocess": [],
                "stt": [],
                "faq_lookup": None,
                "faq_hit": False,
                "faq_audio_url": None,
                "rag": None,
                "rag_chunks": None,
                "llm": None,
                "content_filter": None,
                "content_filter_pass": None,
                "content_filter_flagged": None,
                "verifier": None,
                "verifier_pass": None,
                "verifier_historical_accuracy": None,
                "verifier_appropriateness": None,
                "verifier_modern_references": None,
                "verifier_in_character": None,
                "verifier_corrected_answer": None,
                "verifier_corrected_emotion": None,
                "anachronism_pass": None,
                "anachronism_reasons": None,
                "moderation_q_pass": None,
                "moderation_q_categories": None,
                "moderation_a_pass": None,
                "moderation_a_categories": None,
                "tts_first_chunk": None,
                "tts_total": None,
                "time_to_first_audio": None,
                "total": None,
                "emotion": None,
            }
        return self._timings[session_id]

    def _print_report(self, session_id: str):
        t = self._timings.pop(session_id, None)
        if not t:
            return
        from datetime import datetime, timezone, timedelta
        cairo = timezone(timedelta(hours=3))  # Africa/Cairo — UTC+3
        now_cairo = datetime.now(tz=cairo)
        lines = [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  ⏱  LATENCY REPORT  —  {now_cairo.strftime('%b %d, %Y  %I:%M:%S %p')} (Cairo)",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━══",
        ]
        for i, d in enumerate(t["preprocess"]):
            lines.append(f"  Preprocess  (batch {i+1})  : {d:.3f}s")
        for i, d in enumerate(t["stt"]):
            lines.append(f"  STT         (batch {i+1})  : {d:.3f}s")
        if t["faq_lookup"] is not None:
            hit_label = "HIT ✅" if t["faq_hit"] else "miss ❌"
            lines.append(f"  FAQ lookup  ({hit_label})  : {t['faq_lookup']:.3f}s")
        if t["rag"] is not None:
            lines.append(f"  RAG ({t['rag_chunks']} chunk(s))       : {t['rag']:.3f}s")
        if t["llm"] is not None:
            lines.append(f"  LLM                    : {t['llm']:.3f}s")
        if t["content_filter"] is not None:
            lines.append(f"  Content filter         : {t['content_filter']:.3f}s")
        if t["anachronism_pass"] is not None:
            label = "PASS ✅" if t["anachronism_pass"] else f"FAIL ❌ ({t['anachronism_reasons']})"
            lines.append(f"  Anachronism            : {label}")
        if t["moderation_q_pass"] is not None:
            label = "PASS ✅" if t["moderation_q_pass"] else f"FAIL ❌ ({t['moderation_q_categories']})"
            lines.append(f"  Moderation (question)  : {label}")
        if t["moderation_a_pass"] is not None:
            label = "PASS ✅" if t["moderation_a_pass"] else f"FAIL ❌ ({t['moderation_a_categories']})"
            lines.append(f"  Moderation (answer)    : {label}")
        if t["verifier"] is not None:
            lines.append(f"  Verifier               : {t['verifier']:.3f}s")
        if t["tts_first_chunk"] is not None:
            lines.append(f"  TTS first chunk        : {t['tts_first_chunk']:.3f}s")
        if t["tts_total"] is not None:
            lines.append(f"  TTS total              : {t['tts_total']:.3f}s")
        lines.append("  ─────────────────────────────────────")
        if t["time_to_first_audio"] is not None:
            lines.append(f"  Time to first audio    : {t['time_to_first_audio']:.3f}s")
        if t["total"] is not None:
            lines.append(f"  Total                  : {t['total']:.3f}s")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        report = "\n".join(lines)
        print(report)

    @staticmethod
    def _log_history(session_id: str, history: list | None):
        """Print the conversation memory being fed to the LLM this turn."""
        turns = len(history) // 2 if history else 0
        lines = [
            "┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄",
            f"  🧠 CONVERSATION HISTORY  —  sid={session_id[:8]}  ({turns} turn(s))",
            "┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄",
        ]
        if not history:
            lines.append("  (empty — first turn of this connection)")
        else:
            for msg in history:
                role = msg.get("role", "?")
                tag = "👤 user " if role == "user" else "🎭 reply"
                content = (msg.get("content") or "").replace("\n", " ")
                lines.append(f"  {tag} : {content}")
        lines.append("┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄┄")
        print("\n".join(lines))

    async def enqueue(self, session_id: str, audio_bytes: bytes, is_final: bool = False):
        await self.preprocess_queue.put((session_id, audio_bytes, is_final))

    async def enqueue_finalize(self, session_id: str):
        """Called when end_of_utterance lands exactly on a 5-chunk boundary — no remaining audio.
        Routes through preprocess_queue (not stt_queue directly) so it always arrives
        AFTER all pending batches finish preprocessing — prevents the finalize signal
        from racing ahead of still-running preprocess threads."""
        await self.preprocess_queue.put((session_id, None, True))

    # --- Workers ---

    async def _preprocess_worker(self):
        while True:
            session_id, audio_bytes, is_final = await self.preprocess_queue.get()

            # Finalize-only signal (no audio) — pass straight to stt_queue so it
            # arrives after all previously queued preprocess jobs complete.
            if audio_bytes is None:
                await self.stt_queue.put((session_id, None, True))
                continue

            t0 = time.perf_counter()
            try:
                preprocessed = await asyncio.to_thread(self._run_preprocess, audio_bytes)
                self._t(session_id)["preprocess"].append(time.perf_counter() - t0)
                await self.stt_queue.put((session_id, preprocessed, is_final))
            except Exception as e:
                await self._send_error(session_id, f"Preprocessing failed: {e!r}")

    async def _stt_worker(self):
        # session_id → background embedding task (speculative, started on first partial)
        _speculative_embed: dict[str, asyncio.Task] = {}

        while True:
            session_id, preprocessed_audio, is_final = await self.stt_queue.get()
            t0 = time.perf_counter()
            try:
                session = self.connection_manager.get_session(session_id)

                if preprocessed_audio is not None:
                    transcript = await asyncio.to_thread(self.stt_service.transcribe, preprocessed_audio)
                    self._t(session_id)["stt"].append(time.perf_counter() - t0)
                    if session and transcript.strip():
                        session.append_partial_transcript(transcript)

                        # Speculatively start embedding on the first partial transcript
                        # so it runs in parallel with remaining STT batches.
                        if self.db_session_factory and session_id not in _speculative_embed:
                            partial = session.get_combined_transcript()
                            _speculative_embed[session_id] = asyncio.create_task(
                                asyncio.to_thread(generate_query_embedding, partial)
                            )
                            log.detail("speculative embedding started on partial transcript")

                if not is_final:
                    continue

                full_transcript = session.get_combined_transcript() if session else ""
                if not full_transcript:
                    _speculative_embed.pop(session_id, None)
                    await self._send_error(session_id, "Empty transcript")
                    continue

                if session:
                    session.set_final_transcript(full_transcript)
                await self.connection_manager.send_json(session_id, build_final_transcript_event(full_transcript))

                # Pass along the speculative embedding task (may already be done)
                spec_task = _speculative_embed.pop(session_id, None)
                await self.llm_queue.put((session_id, full_transcript, spec_task))
            except Exception as e:
                _speculative_embed.pop(session_id, None)
                await self._send_error(session_id, f"STT failed: {e!r}")

    async def _llm_worker(self):
        while True:
            session_id, transcript, spec_embed_task = await self.llm_queue.get()
            t0 = time.perf_counter()
            try:
                session = self.connection_manager.get_session(session_id)
                character_id = session.character_id if session else None

                # Tier 3 — kick off question moderation in parallel with FAQ lookup so we
                # don't pay its 200-300ms latency on FAQ hits or legitimate questions.
                question_moderation_task = self.verifier.start_question_moderation(transcript)

                # --- FAQ lookup (skip LLM if we have a matching answer) ---
                if self.faq_memory_cache or self.db_session_factory:
                    log.step("FAQ", f"lookup (character={character_id}) — {transcript[:60]!r}")
                    t_faq = time.perf_counter()
                    try:
                        faq = await asyncio.wait_for(
                            self._lookup_faq(transcript, character_id, spec_embed_task),
                            timeout=config.FAQ_LOOKUP_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        log.warn("FAQ", f"lookup timed out after {config.FAQ_LOOKUP_TIMEOUT}s — falling through to LLM")
                        faq = None
                    self._t(session_id)["faq_lookup"] = time.perf_counter() - t_faq
                    if faq:
                        # FAQ wins — discard the in-flight question moderation, no LLM call needed.
                        if question_moderation_task:
                            question_moderation_task.cancel()
                        self._t(session_id)["faq_hit"] = True
                        self._t(session_id)["faq_audio_url"] = faq.audio_url
                        faq_emotion = getattr(faq, "emotion", None)
                        self._t(session_id)["emotion"] = faq_emotion
                        log.ok("FAQ", f"hit ({self._t(session_id)['faq_lookup']*1000:.0f}ms) — {faq.question[:60]!r}")
                        log.detail(f"answer    : {faq.answer}")
                        log.detail(f"emotion   : {faq_emotion}")
                        log.detail(f"audio_url : {faq.audio_url or '(none — will use TTS)'}")
                        parsed = {"answer": faq.answer, "sources": [], "emotion": faq_emotion}
                        if session:
                            session.set_reply_text(parsed)
                            # FAQ hits skip the LLM but must still be recorded so the
                            # conversation stays coherent across turns.
                            session.append_turn(transcript, faq.answer)
                        await self.connection_manager.send_json(session_id, build_reply_text_done_event(faq.answer, emotion=faq_emotion))

                        if faq.audio_url:
                            log.step("FAQ", "streaming cached audio (skipping LLM + TTS)")
                            await self._stream_cached_audio(session_id, faq.audio_url)
                        else:
                            log.step("FAQ", "no cached audio — sending to TTS")
                            await self.tts_queue.put((session_id, faq.answer))
                        continue  # skip LLM
                    else:
                        log.info("FAQ", f"miss ({self._t(session_id)['faq_lookup']*1000:.0f}ms) — falling through to LLM")

                # --- No FAQ hit — proceed with LLM ---
                # Tier 1 — synchronous regex profanity gate on the question.
                q_regex = self.verifier.regex_check_question(transcript)
                self._t(session_id)["content_filter"] = q_regex.latency_s
                if not q_regex.passed:
                    self._t(session_id)["content_filter_pass"] = False
                    self._t(session_id)["content_filter_flagged"] = ", ".join(q_regex.details.get("flagged", []))
                    log.fail("PIPE", "question blocked by REGEX — playing fallback audio")
                    if question_moderation_task:
                        question_moderation_task.cancel()
                    await self._send_fallback_audio(session_id, character_id)
                    continue

                # Tier 3 — question moderation result (already running since FAQ kickoff).
                if question_moderation_task:
                    q_mod = await question_moderation_task
                    cats = q_mod.details.get("categories", [])
                    self._t(session_id)["moderation_q_pass"] = q_mod.passed
                    self._t(session_id)["moderation_q_categories"] = ", ".join(cats) if cats else None
                    if not q_mod.passed:
                        log.fail("PIPE", f"question blocked by MODELS moderation: {cats} — playing fallback audio")
                        await self._send_fallback_audio(session_id, character_id)
                        continue

                # --- RAG: retrieve verified history to ground the LLM (FAQ miss path) ---
                # Reuse the speculative embedding already computed during STT — same
                # query embedding works for both FAQ and history search.
                retrieved_chunks = []
                if self.db_session_factory:
                    t_rag = time.perf_counter()
                    try:
                        embedding = await self._resolve_embedding(transcript, spec_embed_task)
                        retrieved_chunks = await asyncio.wait_for(
                            self._retrieve_history(embedding, character_id),
                            timeout=config.HISTORY_LOOKUP_TIMEOUT,
                        )
                    except asyncio.TimeoutError:
                        log.warn("RAG", f"history retrieval timed out after {config.HISTORY_LOOKUP_TIMEOUT}s — answering ungrounded")
                    except Exception as e:
                        log.warn("RAG", f"history retrieval failed (non-fatal): {e}")
                    self._t(session_id)["rag"] = time.perf_counter() - t_rag
                    self._t(session_id)["rag_chunks"] = len(retrieved_chunks)
                    log.ok("RAG", f"retrieved {len(retrieved_chunks)} history chunk(s) ({self._t(session_id)['rag']*1000:.0f}ms)")

                prompt_key = config.get_prompt_key_by_character_id(character_id) if character_id else "mohandeskhana-student"
                user_prompt, system_prompt = build_narrator_prompts(
                    character_id=character_id,
                    question=transcript,
                    prompt_key=prompt_key,
                    retrieved_chunks=retrieved_chunks,
                )
                # Conversation memory for this connection (excludes the current turn,
                # which we append only after the reply is finalized below).
                history = session.get_history() if session else None
                self._log_history(session_id, history)
                log.step("LLM", f"generating reply (history_turns={len(history) // 2 if history else 0})")
                reply_raw = await asyncio.to_thread(self.llm_service.generate_reply, user_prompt, system_prompt, history)
                self._t(session_id)["llm"] = time.perf_counter() - t0
                parsed = self._parse_llm_reply(reply_raw)
                emotion = parsed.get("emotion")
                self._t(session_id)["emotion"] = emotion

                log.ok("LLM", f"reply generated ({self._t(session_id)['llm']:.2f}s)")
                log.detail(f"answer  : {parsed['answer']}")
                log.detail(f"emotion : {emotion}")
                log.detail(f"raw     : {reply_raw}")

                # Tier 1 — synchronous regex on the answer (profanity + anachronism).
                # Profanity hits → generic fallback (inappropriate content).
                # Anachronism hits → verifier-style verify audio (historical-accuracy failure).
                a_regex = self.verifier.regex_check_answer(parsed["answer"], character_id)
                self._t(session_id)["content_filter"] = (
                    (self._t(session_id).get("content_filter") or 0) + (a_regex.total_latency_s or 0)
                )

                profanity_result = a_regex.by_name("regex.profanity_answer")
                anachronism_result = a_regex.by_name("regex.anachronism")

                if profanity_result and not profanity_result.passed:
                    self._t(session_id)["content_filter_pass"] = False
                    self._t(session_id)["content_filter_flagged"] = ", ".join(profanity_result.details.get("flagged", []))
                    log.fail("PIPE", "answer blocked by REGEX profanity — playing fallback audio")
                    await self._send_fallback_audio(session_id, character_id)
                    continue
                self._t(session_id)["content_filter_pass"] = True

                if anachronism_result is not None:
                    self._t(session_id)["anachronism_pass"] = anachronism_result.passed
                    if not anachronism_result.passed:
                        self._t(session_id)["anachronism_reasons"] = "; ".join(anachronism_result.reasons)
                        log.fail("PIPE", "answer blocked by REGEX anachronism — playing verify audio")
                        await self._send_verifier_fallback_audio(session_id, character_id)
                        continue

                if session:
                    session.set_reply_text(parsed)
                    # Commit this exchange to conversation memory. If the verifier later
                    # produces a corrected answer, _run_verification updates this entry.
                    session.append_turn(transcript, parsed["answer"])

                await self.connection_manager.send_json(session_id, build_reply_text_done_event(parsed["answer"], emotion=emotion))
                # TTS starts immediately — verification races alongside it
                self._tts_done_events[session_id] = asyncio.Event()
                await self.tts_queue.put((session_id, parsed["answer"]))
                asyncio.create_task(self._run_verification(session_id, transcript, parsed["answer"], character_id, emotion))
            except Exception as e:
                await self._send_error(session_id, f"LLM failed: {e}")

    async def _run_verification(self, session_id: str, transcript: str, answer: str, character_id: str | None, emotion: str | None = None):
        """Background task: runs Tier 2/3 verification in parallel with TTS.

        Owns the final done signal or verify audio — _tts_worker just sets the event.
        On failure, if the LLM judge provided a ``corrected_answer``, replace the
        original text/audio with the corrected reply. Otherwise plays the static
        verify audio.
        """
        try:
            agg = await self.verifier.run_answer_async_checks(
                transcript=transcript,
                answer=answer,
                character_id=character_id,
                fallback_emotion=emotion,
            )
            self._t(session_id)["verifier"] = agg.total_latency_s

            # Persist per-check timings + flagged details
            for r in agg.results:
                if r.name == "models.moderation_answer":
                    self._t(session_id)["moderation_a_pass"] = r.passed
                    cats = r.details.get("categories", [])
                    self._t(session_id)["moderation_a_categories"] = ", ".join(cats) if cats else None
                elif r.name == "models.llm_judge":
                    self._t(session_id)["verifier_pass"] = r.passed
                    self._t(session_id)["verifier_historical_accuracy"] = json.dumps(r.details.get("historical_accuracy"))
                    self._t(session_id)["verifier_appropriateness"] = json.dumps(r.details.get("appropriateness"))
                    self._t(session_id)["verifier_modern_references"] = json.dumps(r.details.get("modern_references"))
                    self._t(session_id)["verifier_in_character"] = json.dumps(r.details.get("in_character"))

            if not agg.passed:
                # Abort if TTS still streaming — no effect if already done
                self._verify_abort.add(session_id)

            # Wait for TTS to finish streaming (aborted or natural end) before finalizing
            event = self._tts_done_events.get(session_id)
            if event:
                try:
                    await asyncio.wait_for(event.wait(), timeout=15.0)
                except asyncio.TimeoutError:
                    log.warn("VERIFY", f"timed out waiting for TTS to finish (sid={session_id[:8]})")

            # Always clear the abort flag once TTS has stopped — _stream_tts_live only
            # consumes it when it sees it mid-stream, so a flag added after TTS already
            # finished naturally would otherwise leak and cut the next utterance.
            self._verify_abort.discard(session_id)

            if session_id in self._tts_error_sessions:
                self._tts_error_sessions.discard(session_id)
                return

            if agg.passed:
                await self.send_queue.put((session_id, None, -1))
                return

            corrected = (agg.corrected_answer or "").strip()
            if corrected:
                corrected_emotion = agg.corrected_emotion or emotion
                log.step("VERIFY", "replaying with corrected answer (verify audio first, then corrected TTS)")
                log.detail(f"corrected         : {corrected}")
                log.detail(f"corrected_emotion : {corrected_emotion}")
                self._t(session_id)["verifier_corrected_answer"] = corrected
                self._t(session_id)["verifier_corrected_emotion"] = corrected_emotion
                # Keep conversation memory in sync with what the user actually hears:
                # overwrite the last assistant turn (set in _llm_worker) with the correction.
                session = self.connection_manager.get_session(session_id)
                if session and session.conversation_history and session.conversation_history[-1]["role"] == "assistant":
                    session.conversation_history[-1]["content"] = corrected
                await self.connection_manager.send_json(session_id, build_reply_text_done_event(corrected, emotion=corrected_emotion))
                # Play the static verify audio first; suppress its `done` signal so the
                # corrected TTS that follows can stream into the same utterance.
                await self._send_verifier_fallback_audio(session_id, character_id, send_done=False)
                # No event registered → _tts_worker will send `done` itself when the corrected TTS finishes.
                await self.tts_queue.put((session_id, corrected))
            else:
                log.fail("VERIFY", "no corrected_answer — playing verify audio")
                # The rejected answer never reached the user (verify audio plays instead),
                # so drop it from conversation memory to avoid poisoning later turns.
                session = self.connection_manager.get_session(session_id)
                if session and session.conversation_history and session.conversation_history[-1]["role"] == "assistant":
                    session.conversation_history = session.conversation_history[:-2]
                await self._send_verifier_fallback_audio(session_id, character_id)

        except Exception as e:
            log.warn("VERIFY", f"unexpected error in _run_verification: {e} — sending done")
            self._tts_done_events.pop(session_id, None)
            self._tts_error_sessions.discard(session_id)
            self._verify_abort.discard(session_id)
            await self.send_queue.put((session_id, None, -1))

    async def _tts_worker(self):
        while True:
            session_id, reply_text = await self.tts_queue.get()
            t0 = time.perf_counter()
            try:
                #raise Exception("forced test error") # TESTING: force an error to verify fallback audio works
                session = self.connection_manager.get_session(session_id)
                character_id = session.character_id if session else None
                if session:
                    session.set_state("STREAMING_TTS")
                    session.dead_time_end = time.time()

                completed = await self._stream_tts_live(session_id, reply_text, t0, character_id)
                self._t(session_id)["tts_total"] = time.perf_counter() - t0
                # Signal _run_verification that streaming has ended (pass or abort)
                # _run_verification owns the final done/verify signal on the LLM path
                event = self._tts_done_events.pop(session_id, None)
                if event:
                    event.set()
                else:
                    # FAQ path — no verification running, send done directly
                    if completed:
                        await self.send_queue.put((session_id, None, -1))

            except asyncio.TimeoutError:
                elapsed = time.perf_counter() - t0
                log.fail("TTS", f"first-chunk timeout ({elapsed:.2f}s > {config.TTS_FIRST_CHUNK_TIMEOUT}s) — streaming fallback")
                self._tts_error_sessions.add(session_id)
                event = self._tts_done_events.pop(session_id, None)
                if event:
                    event.set()
                await self._send_fallback_audio(session_id, character_id)

            except Exception as e:
                log.fail("TTS", f"streaming failed (sid={session_id[:8]}): {e}")
                self._tts_error_sessions.add(session_id)
                event = self._tts_done_events.pop(session_id, None)
                if event:
                    event.set()
                await self._send_fallback_audio(session_id, character_id)

    # --- TTS sentence helpers ---

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text on sentence boundaries, keeping punctuation with the sentence."""
        parts = re.split(r'(?<=[.!?؟،])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _pcm_chunk_to_wav(pcm_bytes: bytes, sample_rate: int = 44100, channels: int = 1) -> bytes:
        """Wrap a raw PCM16 chunk in a minimal WAV header so the client can decode it."""
        import struct
        bits = 16
        data_len = len(pcm_bytes)
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + data_len, b"WAVE",
            b"fmt ", 16, 1, channels, sample_rate,
            sample_rate * channels * bits // 8,
            channels * bits // 8, bits,
            b"data", data_len,
        )
        return header + pcm_bytes

    def _start_sentence_tts(self, sentence: str, character_id: str) -> thread_queue.Queue:
        """Kick off ElevenLabs for one sentence in a background thread.
        Collects the full sentence MP3, decodes it, then puts a single WAV chunk
        in the queue followed by a None sentinel."""
        q: thread_queue.Queue = thread_queue.Queue()

        def run():
            try:
                import librosa
                mp3_chunks = []
                for chunk in self.elevenlabs_service.stream_audio(sentence, character_id):
                    if chunk:
                        mp3_chunks.append(chunk)

                if not mp3_chunks:
                    return

                mp3_bytes = b"".join(mp3_chunks)
                audio, sr = librosa.load(io.BytesIO(mp3_bytes), sr=None, mono=True)
                trimmed, _ = librosa.effects.trim(audio, top_db=35, frame_length=512, hop_length=128)
                if len(trimmed) == 0:
                    trimmed = audio

                wav_buf = io.BytesIO()
                sf.write(wav_buf, trimmed, sr, format="WAV", subtype="PCM_16")
                wav_buf.seek(0)
                q.put(wav_buf.read())

            except Exception as e:
                q.put(e)
            finally:
                q.put(None)

        threading.Thread(target=run, daemon=True).start()
        return q

    async def _stream_tts_live(self, session_id: str, text: str, t0: float, character_id: str = None):
        bridge: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        sentences = self._split_sentences(text) or [text]

        def run_all():
            try:
                # Start sentence 0 immediately; prefetch sentence N+1 while N plays.
                queues = [self._start_sentence_tts(sentences[0], character_id)]
                next_idx = 1

                for q in queues:
                    # Fire off the next sentence's TTS request before blocking on current.
                    if next_idx < len(sentences):
                        queues.append(self._start_sentence_tts(sentences[next_idx], character_id))
                        next_idx += 1

                    while True:
                        item = q.get()
                        if item is None:
                            break
                        if isinstance(item, Exception):
                            raise item
                        loop.call_soon_threadsafe(bridge.put_nowait, item)

            except Exception as e:
                loop.call_soon_threadsafe(bridge.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(bridge.put_nowait, None)

        threading.Thread(target=run_all, daemon=True).start()

        chunk_index = 0
        first_chunk = True
        while True:
            if first_chunk:
                # Enforce the first-chunk deadline; TimeoutError propagates to _tts_worker.
                item = await asyncio.wait_for(bridge.get(), timeout=config.TTS_FIRST_CHUNK_TIMEOUT)
            else:
                item = await bridge.get()

            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            if first_chunk:
                self._t(session_id)["tts_first_chunk"] = time.perf_counter() - t0
                first_chunk = False

            if session_id in self._verify_abort:
                self._verify_abort.discard(session_id)
                log.fail("TTS", f"verifier abort — cutting stream (sid={session_id[:8]})")
                return False

            await self.send_queue.put((session_id, item, chunk_index))
            chunk_index += 1

        return True

    async def _send_worker(self):
        first_chunk_sent: dict[str, bool] = {}
        collected_audio: dict[str, list[bytes]] = {}   # session_id → WAV chunks to upload
        while True:
            session_id, audio_chunk, chunk_index = await self.send_queue.get()
            try:
                if audio_chunk is None:
                    first_chunk_sent.pop(session_id, None)
                    session = self.connection_manager.get_session(session_id)

                    # Snapshot the client-recorded question audio BEFORE the buffer
                    # gets cleared — _save_past_question runs as a background task and
                    # would otherwise see an empty buffer.
                    question_wav: bytes | None = None
                    if session and self.db_session_factory:
                        question_wav = self._assemble_question_wav(session)

                    if session and session.dead_time_start:
                        self._t(session_id)["total"] = time.time() - session.dead_time_start

                    # Snapshot timings + audio before _print_report pops them, then save in background
                    t = self._timings.get(session_id, {}).copy()
                    chunks = collected_audio.pop(session_id, [])
                    self._print_report(session_id)
                    if self.db_session_factory and session:
                        # Snapshot Q/A now — reset_for_next_utterance() below clears these,
                        # and _save_past_question reads them after its own awaits.
                        question_text = session.final_transcript
                        answer_text = session.reply_text
                        asyncio.create_task(self._save_past_question(session, t, chunks, question_wav, question_text, answer_text))

                    # Reset per-utterance state so the NEXT turn on this same connection
                    # starts clean. Preserves conversation_history (multi-turn memory).
                    if session:
                        session.reset_for_next_utterance()

                    await self.connection_manager.send_json(session_id, build_tts_done_event())
                else:
                    if not first_chunk_sent.get(session_id):
                        session = self.connection_manager.get_session(session_id)
                        if session and session.dead_time_start:
                            self._t(session_id)["time_to_first_audio"] = time.time() - session.dead_time_start
                        first_chunk_sent[session_id] = True

                    # Collect chunk for background upload
                    collected_audio.setdefault(session_id, []).append(audio_chunk)

                    encoded = base64.b64encode(audio_chunk).decode("utf-8")
                    await self.connection_manager.send_json(
                        session_id,
                        build_tts_audio_chunk_event(chunk_index=chunk_index, audio=encoded),
                    )
            except Exception as e:
                log.fail("PIPE", f"send worker error (sid={session_id[:8]}): {e}")

    # --- Helpers ---

    @staticmethod
    def _parse_llm_reply(reply_raw: str) -> dict:
        """Robustly extract {answer, emotion, sources} from whatever the LLM returned.

        Handles three cases the model sometimes produces:
          1. Clean JSON object  → parse directly
          2. Plain text + JSON appended at the end → extract the last {...} block
          3. Anything else      → treat the whole string as the answer
        """
        # 1. Try direct parse first
        try:
            parsed = json.loads(reply_raw)
            if isinstance(parsed, dict) and "answer" in parsed:
                return parsed
            # Parsed but not the right shape (e.g. a bare string in quotes)
            return {"answer": str(parsed), "sources": [], "emotion": None}
        except Exception:
            pass

        # 2. Try to pull out the last {...} block (model prepended plain text before JSON)
        match = re.search(r'\{[\s\S]*\}(?=[^}]*$)', reply_raw)
        if match:
            try:
                parsed = json.loads(match.group())
                if isinstance(parsed, dict) and "answer" in parsed:
                    return parsed
            except Exception:
                pass

        # 3. Fall back: use the raw string but strip any trailing JSON blob
        # so TTS doesn't read out the JSON syntax
        clean = re.sub(r'\s*\{[\s\S]*\}\s*$', '', reply_raw).strip()
        return {"answer": clean or reply_raw, "sources": [], "emotion": None}

    def _run_preprocess(self, audio_bytes: bytes):
        # Rebuild a proper WAV in memory (correct header sizes) and process.
        # No temp file → no Windows file-locking issues, no librosa/audioread fallback.
        audio = self.audio_preprocessor.load_audio_from_wav_bytes(audio_bytes)
        return self.audio_preprocessor.process_audio(audio)

    async def _resolve_embedding(self, transcript: str, spec_embed_task: asyncio.Task = None) -> list[float]:
        """Return the query embedding, reusing the speculative task if it succeeded.
        Awaiting an already-finished task just returns its stored result, so this is
        safe even though _lookup_faq awaited the same task earlier."""
        if spec_embed_task is not None:
            try:
                return await spec_embed_task
            except Exception:
                pass
        return await asyncio.to_thread(generate_query_embedding, transcript)

    async def _retrieve_history(self, embedding: list[float], character_id: str) -> list:
        """Vector-search the college-history KB for grounding chunks (scoped to this
        character + global). Returns [] on any failure — RAG is best-effort."""
        async with self.db_session_factory() as db:
            return await search_history_chunks(db, embedding, character_id)

    async def _lookup_faq(self, transcript: str, character_id: str, spec_embed_task: asyncio.Task = None):
        """Embed the transcript and find the best-matching FAQ.

        Fast-path priority:
        1. Exact-match transcript cache → 0ms (skips embed + search entirely)
        2. In-memory vector index (faq_memory_cache) → embed only, then numpy dot product (<1ms)
        3. DB pgvector search (fallback if memory cache not loaded) → embed + network round trip
        """
        cache_key = (transcript.strip().lower(), (character_id or "").lower())

        if cache_key in self._faq_cache:
            cached = self._faq_cache[cache_key]
            log.detail(f"exact-match cache {'HIT' if cached else 'miss (no match)'} — skipped embed + search")
            return cached

        try:
            # Generate (or reuse speculative) embedding
            t_embed = time.perf_counter()
            if spec_embed_task is not None:
                try:
                    embedding = await spec_embed_task
                    log.detail(f"speculative embedding reused ({(time.perf_counter() - t_embed)*1000:.0f}ms)")
                except Exception:
                    embedding = await asyncio.to_thread(generate_query_embedding, transcript)
                    log.detail(f"embedding (fallback) generated ({(time.perf_counter() - t_embed)*1000:.0f}ms)")
            else:
                embedding = await asyncio.to_thread(generate_query_embedding, transcript)
                log.detail(f"embedding generated ({(time.perf_counter() - t_embed)*1000:.0f}ms)")

            # Search: memory index first, DB fallback
            if self.faq_memory_cache and self.faq_memory_cache.is_loaded:
                result = await asyncio.to_thread(
                    self.faq_memory_cache.search, embedding, character_id
                )
            elif self.db_session_factory:
                t_db = time.perf_counter()
                async with self.db_session_factory() as db:
                    result = await search_similar_faq(db, embedding, character_id.lower() if character_id else character_id)
                log.detail(f"DB query completed ({(time.perf_counter() - t_db)*1000:.0f}ms)")
            else:
                result = None

            self._faq_cache[cache_key] = result  # cache both hits and misses
            return result
        except Exception as e:
            log.warn("FAQ", f"lookup failed (non-fatal): {e}")
            return None

    async def _stream_cached_audio(self, session_id: str, audio_url: str):
        """Fetch pre-generated audio from storage URL and stream it to the client."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(audio_url)
                response.raise_for_status()
                audio_bytes = response.content

            await self.send_queue.put((session_id, audio_bytes, 0))
            log.ok("AUDIO", f"sent cached audio (sid={session_id[:8]})")
        except Exception as e:
            log.warn("AUDIO", f"failed to fetch cached audio: {e} — falling back to TTS")
            # audio_url fetch failed — signal done so session resets cleanly
        finally:
            await self.send_queue.put((session_id, None, -1))  # TTS done signal

    async def _send_verifier_fallback_audio(self, session_id: str, character_id: str = None, send_done: bool = True):
        """Stream <character_id>_verify.wav from data/verification_audios/ when the verifier rejects a response.

        Pass send_done=False to chain more audio after this clip (e.g. a corrected answer's TTS).
        """
        sent = False
        try:
            filename = f"{str(character_id).lower()}_verify.wav" if character_id else "verify.wav"
            path = os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "verification_audios", filename
            ))
            if not os.path.exists(path):
                log.warn("AUDIO", f"verify audio not found: {path} — falling back to default")
                await self._send_fallback_audio(session_id, character_id, send_done=send_done)
                return

            audio, sr = sf.read(path, dtype="float32", always_2d=False)
            wav_buf = io.BytesIO()
            sf.write(wav_buf, audio, sr, format="WAV", subtype="PCM_16")
            await self.send_queue.put((session_id, wav_buf.getvalue(), 0))
            sent = True
            log.ok("AUDIO", f"sent verify audio ({filename}) → sid={session_id[:8]}")
        except Exception as e:
            log.warn("AUDIO", f"could not load verify audio: {e} — falling back to default")
            await self._send_fallback_audio(session_id, character_id, send_done=send_done)
            return
        finally:
            if sent and send_done:
                await self.send_queue.put((session_id, None, -1))

    async def _send_fallback_audio(self, session_id: str, character_id: str = None, send_done: bool = True):
        """Stream <character_id>_fallback.wav to the client as a single TTS chunk, then signal done.

        Called on pipeline errors and TTS first-chunk timeouts so the user always
        hears something instead of silence. Pass send_done=False to chain more audio after.
        """
        try:
            filename = f"{str(character_id).lower()}_fallback.wav" if character_id else "fallback.wav"
            fallback_path = os.path.normpath(os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data", "fallback_audios", filename
            ))
            if not os.path.exists(fallback_path):
                log.warn("AUDIO", f"fallback audio not found: {fallback_path}")
                return

            audio, sr = sf.read(fallback_path, dtype="float32", always_2d=False)
            wav_buf = io.BytesIO()
            sf.write(wav_buf, audio, sr, format="WAV", subtype="PCM_16")
            await self.send_queue.put((session_id, wav_buf.getvalue(), 0))
            log.ok("AUDIO", f"sent fallback audio ({filename}) → sid={session_id[:8]}")
        except Exception as e:
            log.warn("AUDIO", f"could not load fallback audio: {e}")
        finally:
            if send_done:
                # Always signal TTS done so the client and session state reset cleanly.
                await self.send_queue.put((session_id, None, -1))

    def _assemble_question_wav(self, session) -> bytes | None:
        """Reassemble all client-recorded audio chunks for the current utterance into a single valid WAV file.

        Two formats are supported (set on the session via `start_session`):
        - "pcm16_base64_chunks": every chunk is raw PCM16 LE → concat all, wrap in fresh WAV header.
        - "wav": chunk 1 is a complete WAV file (header + PCM); chunks 2+ are headerless PCM →
          strip chunk 1's header, concat its PCM with the rest, wrap in a fresh WAV header
          so the data-size field reflects the full length.
        """
        try:
            chunks_b64 = session.audio_buffer.get_all_chunks()
            if not chunks_b64:
                log.info("AUDIO", "question buffer empty — nothing to assemble")
                return None
            decoded = [base64.b64decode(c) for c in chunks_b64]

            if session.audio_format == "pcm16_base64_chunks":
                raw_pcm = b"".join(decoded)
            else:
                first = decoded[0]
                idx = first.find(b"data")
                if idx == -1 or idx + 8 > len(first):
                    log.fail("AUDIO", f"question chunk 1 missing 'data' marker — cannot assemble (first 32 bytes: {first[:32]!r})")
                    return None
                raw_pcm = first[idx + 8:] + b"".join(decoded[1:])

            if not raw_pcm:
                log.fail("AUDIO", "raw_pcm empty after assembly")
                return None
            wav = self._pcm_chunk_to_wav(raw_pcm, sample_rate=session.sample_rate, channels=1)
            log.ok("AUDIO", f"question wav assembled ({len(decoded)} chunks, {len(wav)} bytes)")
            return wav
        except Exception as e:
            log.fail("AUDIO", f"question assembly failed: {e}")
            return None

    async def _upload_question_audio(self, audio_bytes: bytes, character_id: str) -> str | None:
        """Upload the assembled question audio (already a complete WAV) to the questions bucket."""
        if not audio_bytes:
            log.detail("question audio upload skipped: empty bytes")
            return None
        if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_KEY:
            log.detail("question audio upload skipped: SUPABASE_URL or SUPABASE_SERVICE_KEY not configured")
            return None
        try:
            import uuid as _uuid
            filename = f"{character_id.lower()}_{_uuid.uuid4().hex[:8]}_q.wav"
            url = f"{config.SUPABASE_URL}/storage/v1/object/{config.QUESTIONS_AUDIO_BUCKET}/{filename}"
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    content=audio_bytes,
                    headers={
                        "Authorization": f"Bearer {config.SUPABASE_SERVICE_KEY}",
                        "Content-Type": "audio/wav",
                    },
                )
            if response.status_code in (200, 201):
                public_url = f"{config.SUPABASE_URL}/storage/v1/object/public/{config.QUESTIONS_AUDIO_BUCKET}/{filename}"
                return public_url
            log.detail(f"question audio upload failed: HTTP {response.status_code} (bucket={config.QUESTIONS_AUDIO_BUCKET}) — {response.text}")
            return None
        except Exception as e:
            log.detail(f"question audio upload error: {e!r}")
            return None

    async def _combine_and_upload_audio(self, wav_chunks: list[bytes], character_id: str) -> str | None:
        """Combine WAV chunks into one file and upload to Supabase Storage."""
        if not wav_chunks or not config.SUPABASE_URL or not config.SUPABASE_SERVICE_KEY:
            return None
        try:
            # Combine all WAV chunks into one PCM array
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
            audio_bytes = buf.getvalue()

            # Upload to Supabase Storage
            import uuid as _uuid
            filename = f"{character_id.lower()}_{_uuid.uuid4().hex[:8]}.wav"
            url = f"{config.SUPABASE_URL}/storage/v1/object/{config.RESPONSES_AUDIO_BUCKET}/{filename}"
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    url,
                    content=audio_bytes,
                    headers={
                        "Authorization": f"Bearer {config.SUPABASE_SERVICE_KEY}",
                        "Content-Type": "audio/wav",
                    },
                )
            if response.status_code in (200, 201):
                public_url = f"{config.SUPABASE_URL}/storage/v1/object/public/{config.RESPONSES_AUDIO_BUCKET}/{filename}"
                return public_url
            else:
                log.detail(f"response audio upload failed: HTTP {response.status_code} — {response.text}")
                return None
        except Exception as e:
            log.detail(f"response audio combine/upload error: {e}")
            return None

    async def _save_past_question(self, session, timings: dict, wav_chunks: list[bytes], question_wav: bytes | None = None, question_text: str | None = None, answer_text: str | None = None):
        """Fire-and-forget: combine audio, upload to storage, save interaction to past_questions.
        Runs as a background task so it never blocks the pipeline.

        question_text/answer_text are snapshotted by the caller before the session is
        reset for the next utterance — prefer them over live session fields, which may
        already be cleared by the time this task's awaits resolve."""
        try:
            character_id = (session.character_id or "").lower()

            # Use FAQ cached audio URL if already available, otherwise upload the streamed audio
            audio_url = timings.get("faq_audio_url")
            if not audio_url and wav_chunks:
                audio_url = await self._combine_and_upload_audio(wav_chunks, character_id)

            question_audio_url: str | None = None
            if question_wav:
                question_audio_url = await self._upload_question_audio(question_wav, character_id)
            else:
                log.info("DB", "no question audio to upload")

            from datetime import datetime, timezone, timedelta
            cairo = timezone(timedelta(hours=3))  # Africa/Cairo — UTC+3

            preprocess = sum(timings.get("preprocess", []) or [])
            stt = sum(timings.get("stt", []) or [])
            async with self.db_session_factory() as db:
                await create_past_question(db, {
                    "character_id": character_id,
                    "question": (question_text if question_text is not None else session.final_transcript) or "",
                    "answer": (answer_text if answer_text is not None else session.reply_text) or "",
                    "audio_url": audio_url,
                    "question_audio_url": question_audio_url,
                    "source": "faq" if timings.get("faq_hit") else "llm",
                    "faq_hit": bool(timings.get("faq_hit")),
                    "emotion": timings.get("emotion"),
                    "preprocess_s": round(preprocess, 4) if preprocess else None,
                    "stt_s": round(stt, 4) if stt else None,
                    "faq_lookup_s": round(timings["faq_lookup"], 4) if timings.get("faq_lookup") else None,
                    "llm_s": round(timings["llm"], 4) if timings.get("llm") else None,
                    "content_filter_s": round(timings["content_filter"], 4) if timings.get("content_filter") else None,
                    "content_filter_pass": timings.get("content_filter_pass"),
                    "content_filter_flagged": timings.get("content_filter_flagged"),
                    "verifier_s": round(timings["verifier"], 4) if timings.get("verifier") else None,
                    "verifier_pass": timings.get("verifier_pass"),
                    "verifier_historical_accuracy": timings.get("verifier_historical_accuracy"),
                    "verifier_appropriateness": timings.get("verifier_appropriateness"),
                    "verifier_modern_references": timings.get("verifier_modern_references"),
                    "verifier_in_character": timings.get("verifier_in_character"),
                    "verifier_corrected_answer": timings.get("verifier_corrected_answer"),
                    "verifier_corrected_emotion": timings.get("verifier_corrected_emotion"),
                    "anachronism_pass": timings.get("anachronism_pass"),
                    "anachronism_reasons": timings.get("anachronism_reasons"),
                    "moderation_q_pass": timings.get("moderation_q_pass"),
                    "moderation_q_categories": timings.get("moderation_q_categories"),
                    "moderation_a_pass": timings.get("moderation_a_pass"),
                    "moderation_a_categories": timings.get("moderation_a_categories"),
                    "tts_first_chunk_s": round(timings["tts_first_chunk"], 4) if timings.get("tts_first_chunk") else None,
                    "tts_total_s": round(timings["tts_total"], 4) if timings.get("tts_total") else None,
                    "time_to_first_audio_s": round(timings["time_to_first_audio"], 4) if timings.get("time_to_first_audio") else None,
                    "total_s": round(timings["total"], 4) if timings.get("total") else None,
                    "created_at": datetime.now(tz=cairo),
                })
            log.ok("DB", f"past question saved (source={'faq' if timings.get('faq_hit') else 'llm'}, audio={'yes' if audio_url else 'no'})")
        except Exception as e:
            import traceback
            log.fail("DB", f"failed to save past question: {e}")
            traceback.print_exc()

    async def _send_error(self, session_id: str, message: str):
        log.fail("PIPE", f"error (sid={session_id[:8]}): {message}")
        session = self.connection_manager.get_session(session_id)
        character_id = session.character_id if session else None
        await self._send_fallback_audio(session_id, character_id)
        if session:
            session.set_state("LISTENING")
