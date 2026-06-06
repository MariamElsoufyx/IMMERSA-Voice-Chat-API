
import asyncio
import time

import app.core.config as config
from app.services.verification.base import CheckResult, AggregateResult
from app.services.verification.regex_checks import check_profanity, check_anachronism
from app.services.verification.model_checks import check_moderation, check_llm_judge
from app.utils.log import log


class Verifier:
    def __init__(self, openai_client=None):
        self.openai_client = openai_client

    # -----------------------------------------------------------------
    # Tier 1 — synchronous regex (sub-millisecond, gates BEFORE TTS)
    # -----------------------------------------------------------------

    def regex_check_question(self, transcript: str) -> CheckResult:
        return check_profanity(transcript, name="regex.profanity_question")

    def regex_check_answer(self, answer: str, character_id: str | None) -> AggregateResult:
        """Profanity + anachronism in one bundle."""
        t0 = time.perf_counter()
        results = [check_profanity(answer, name="regex.profanity_answer")]
        if config.ANACHRONISM_ENABLED:
            results.append(check_anachronism(answer, character_id))
        return AggregateResult(results=results, total_latency_s=time.perf_counter() - t0)

    # -----------------------------------------------------------------
    # Tier 2 helper — start question moderation in parallel with FAQ
    # -----------------------------------------------------------------

    def start_question_moderation(self, transcript: str) -> "asyncio.Task[CheckResult] | None":
        """Returns a Task you can either ``await`` or ``cancel()`` if FAQ wins."""
        if not (config.MODERATION_ENABLED and self.openai_client):
            return None
        return asyncio.create_task(
            check_moderation(self.openai_client, transcript, name="models.moderation_question")
        )

    # -----------------------------------------------------------------
    # Tier 2 — async checks gathered IN PARALLEL with TTS
    # -----------------------------------------------------------------

    async def run_answer_async_checks(
        self,
        *,
        transcript: str,
        answer: str,
        character_id: str | None,
        fallback_emotion: str | None = None,
    ) -> AggregateResult:
        """OpenAI moderation + LLM judge — gathered concurrently.

        The pipeline kicks this off as a background task at the same moment it
        queues TTS, so the checks race speech synthesis. ``corrected_answer``
        and ``corrected_emotion`` (only the LLM judge supplies them) are
        surfaced on the returned :class:`AggregateResult`.
        """
        t0 = time.perf_counter()
        coros = []
        order: list[str] = []

        if config.MODERATION_ENABLED and self.openai_client:
            coros.append(check_moderation(self.openai_client, answer, name="models.moderation_answer"))
            order.append("moderation")

        if self.openai_client:
            coros.append(check_llm_judge(
                self.openai_client,
                transcript=transcript,
                answer=answer,
                character_id=character_id,
                fallback_emotion=fallback_emotion,
            ))
            order.append("llm_judge")

        if not coros:
            return AggregateResult(results=[], total_latency_s=0.0)

        log.step("VERIFY", f"running {len(coros)} async check(s) in parallel: {', '.join(order)}")
        gathered = await asyncio.gather(*coros)

        agg = AggregateResult(results=list(gathered), total_latency_s=time.perf_counter() - t0)
        if agg.passed:
            log.ok("VERIFY", f"all checks passed ({agg.total_latency_s:.2f}s)")
        else:
            log.fail("VERIFY", f"FAILED ({agg.total_latency_s:.2f}s) — reasons: {agg.all_reasons()}")
        for r in gathered:
            if r.name == "models.llm_judge":
                agg.corrected_answer = r.details.get("corrected_answer")
                agg.corrected_emotion = r.details.get("corrected_emotion")
                break
        return agg
