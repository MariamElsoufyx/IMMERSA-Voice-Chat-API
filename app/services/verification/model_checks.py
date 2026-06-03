"""Tier 3 — OpenAI-model–backed checks. 200ms (moderation) – multi-second (LLM judge).

Two distinct models live here:

* ``omni-moderation-latest`` — multilingual safety classifier; categorical
  hate / sexual / self-harm / violence / etc. signals. Cheap, ~200ms.
* ``gpt-4.1-nano`` (configurable) — LLM-as-judge that scores the answer on
  ``historical_accuracy``, ``appropriateness``, ``modern_references``,
  ``in_character`` and may return a ``corrected_answer``. Slower, multi-second.

Both are fail-open on error so an OpenAI outage cannot block legitimate replies.
"""
import asyncio
import json
import time

import app.core.config as config
from app.characters.build_prompt import build_verifier_prompts
from app.services.verification.base import CheckResult
from app.utils.log import log


MODERATION_MODEL = "omni-moderation-latest"


# ---------------------------------------------------------------------------
# OpenAI Moderation API — multilingual safety classifier
# ---------------------------------------------------------------------------

def _moderate_sync(openai_client, text: str) -> tuple[bool, list[str]]:
    if not openai_client or not text or not text.strip():
        return True, []
    resp = openai_client.moderations.create(model=MODERATION_MODEL, input=text)
    result = resp.results[0]
    if not result.flagged:
        return True, []
    cats = result.categories
    flagged: list[str] = []
    for name in dir(cats):
        if name.startswith("_"):
            continue
        try:
            if getattr(cats, name) is True:
                flagged.append(name)
        except Exception:
            continue
    return False, flagged


async def check_moderation(openai_client, text: str, *, name: str) -> CheckResult:
    """Run OpenAI Moderation on `text`. `name` differentiates question vs answer."""
    label = name.split(".")[-1]  # "moderation_question" / "moderation_answer"
    t0 = time.perf_counter()
    try:
        passed, flagged = await asyncio.to_thread(_moderate_sync, openai_client, text)
    except Exception as e:
        log.warn("MODELS", f"{label} error: {e} — fail-open")
        return CheckResult(
            name=name, passed=True, reasons=[], details={"error": str(e)},
            latency_s=time.perf_counter() - t0,
        )
    latency = time.perf_counter() - t0
    if passed:
        log.ok("MODELS", f"{label} clean ({latency*1000:.0f}ms)")
    else:
        log.fail("MODELS", f"{label} flagged: {flagged} ({latency*1000:.0f}ms)")
    return CheckResult(
        name=name,
        passed=passed,
        reasons=[f"moderation: {c}" for c in flagged],
        details={"categories": flagged, "model": MODERATION_MODEL},
        latency_s=latency,
    )


# ---------------------------------------------------------------------------
# LLM-as-judge — character-aware multi-criterion verifier
# ---------------------------------------------------------------------------

_ALLOWED_EMOTIONS = {"happy", "sad", "angry", "disgust", "surprise", "neutral"}


def _llm_judge_sync(openai_client, system_prompt: str, user_prompt: str) -> str:
    completion = openai_client.chat.completions.create(
        model=config.openai_verifier_model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=config.openai_verifier_max_tokens,
        response_format={"type": "json_object"},
        stream=False,
    )
    return (completion.choices[0].message.content or "").strip()


async def check_llm_judge(
    openai_client,
    *,
    transcript: str,
    answer: str,
    character_id: str | None,
    fallback_emotion: str | None = None,
) -> CheckResult:
    """Multi-criterion LLM judge. Stuffs `corrected_answer` / `corrected_emotion` into details."""
    t0 = time.perf_counter()
    if not openai_client:
        log.warn("MODELS", "llm_judge skipped — no openai_client")
        return CheckResult(name="models.llm_judge", passed=True, latency_s=0.0)
    try:
        user_prompt, system_prompt = build_verifier_prompts(
            character_id=character_id, question=transcript, answer=answer,
        )
        if not user_prompt or not system_prompt:
            log.warn("MODELS", f"llm_judge prompts missing for character={character_id} — fail-open")
            return CheckResult(
                name="models.llm_judge", passed=True,
                details={"reason": "prompts_missing"},
                latency_s=time.perf_counter() - t0,
            )

        raw = await asyncio.to_thread(_llm_judge_sync, openai_client, system_prompt, user_prompt)
        if not raw:
            log.warn("MODELS", "llm_judge empty response — fail-open")
            return CheckResult(
                name="models.llm_judge", passed=True,
                details={"reason": "empty_response"},
                latency_s=time.perf_counter() - t0,
            )

        parsed = json.loads(raw)

        # Recompute `overall` deterministically from the dimensions instead of
        # trusting the judge's overall_pass field. `historical_accuracy` still
        # gates — but the prompt narrows it to only fail on clearly invented /
        # fabricated facts, NOT on the judge's unreliable "anachronistic for the
        # operation year" guesses (which produced false failures on correct
        # period facts).
        def _dim_pass(d) -> bool:
            return bool(d.get("pass", True)) if isinstance(d, dict) else True

        modern_refs = parsed.get("modern_references")
        modern_found = bool(modern_refs.get("found", False)) if isinstance(modern_refs, dict) else False
        overall = (
            _dim_pass(parsed.get("historical_accuracy"))
            and _dim_pass(parsed.get("appropriateness"))
            and not modern_found
            and _dim_pass(parsed.get("in_character"))
        )

        raw_corrected_emotion = (parsed.get("corrected_emotion") or "").strip().lower()
        corrected_emotion = (
            raw_corrected_emotion if raw_corrected_emotion in _ALLOWED_EMOTIONS else fallback_emotion
        )

        latency = time.perf_counter() - t0
        if overall:
            log.ok("MODELS", f"llm_judge overall_pass=true ({latency:.2f}s)")
        else:
            log.fail("MODELS", f"llm_judge overall_pass=false ({latency:.2f}s)")
        log.detail(f"historical_accuracy : {parsed.get('historical_accuracy')}")
        log.detail(f"appropriateness     : {parsed.get('appropriateness')}")
        log.detail(f"modern_references   : {parsed.get('modern_references')}")
        log.detail(f"in_character        : {parsed.get('in_character')}")

        return CheckResult(
            name="models.llm_judge",
            passed=overall,
            reasons=[] if overall else ["llm_judge: overall_pass=false"],
            details={
                "raw": parsed,
                "historical_accuracy": parsed.get("historical_accuracy"),
                "appropriateness": parsed.get("appropriateness"),
                "modern_references": parsed.get("modern_references"),
                "in_character": parsed.get("in_character"),
                "corrected_answer": (parsed.get("corrected_answer") or "").strip() or None,
                "corrected_emotion": corrected_emotion,
                "model": config.openai_verifier_model_name,
            },
            latency_s=latency,
        )

    except Exception as e:
        log.warn("MODELS", f"llm_judge error: {e} — fail-open")
        return CheckResult(
            name="models.llm_judge", passed=True,
            details={"error": str(e)},
            latency_s=time.perf_counter() - t0,
        )
