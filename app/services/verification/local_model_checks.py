"""Local equivalents of the OpenAI-backed Tier 3 checks in model_checks.py.

Both functions return the same CheckResult shape so the Verifier orchestrator
can be wired to either one with no other code changes.

* check_moderation_local() — HuggingFace `unitary/toxic-bert` classifier, ~30ms
  on CPU. Replaces OpenAI's omni-moderation-latest. Categorical: toxic, severe,
  obscene, threat, insult, identity_hate.
* check_llm_judge_local()  — Reuses LLMLocalService (Ollama / llama-cpp) to
  emit the same JSON judge schema as gpt-4.1-nano.

Both fail open on error: a local model crash must never block a legitimate reply.
"""
import asyncio
import json
import re
import time

import app.core.config as config
from app.characters.build_prompt import build_verifier_prompts
from app.services.verification.base import CheckResult
from app.utils.log import log


# ---------------------------------------------------------------------------
# Local moderation — toxic-bert classifier
# ---------------------------------------------------------------------------

_TOXIC_PIPELINE = None  # lazy-loaded HF pipeline; one process-wide instance


def _get_toxic_pipeline():
    global _TOXIC_PIPELINE
    if _TOXIC_PIPELINE is None:
        from transformers import pipeline as hf_pipeline
        _TOXIC_PIPELINE = hf_pipeline(
            "text-classification",
            model=config.local_moderation_model,
            top_k=None,  # return scores for ALL labels, not just argmax
            device=-1,    # CPU; set to 0 for GPU
        )
    return _TOXIC_PIPELINE


def _moderate_local_sync(text: str) -> tuple[bool, list[str]]:
    if not text or not text.strip():
        return True, []
    clf = _get_toxic_pipeline()
    scored = clf(text)[0]  # list of {label, score}
    threshold = config.local_moderation_threshold
    flagged = [item["label"] for item in scored if item["score"] >= threshold]
    return (len(flagged) == 0), flagged


async def check_moderation_local(text: str, *, name: str) -> CheckResult:
    """Local toxic-bert moderation. `name` differentiates question vs answer."""
    label = name.split(".")[-1]
    t0 = time.perf_counter()
    try:
        passed, flagged = await asyncio.to_thread(_moderate_local_sync, text)
    except Exception as e:
        log.warn("LOCAL", f"{label} error: {e} — fail-open")
        return CheckResult(
            name=name, passed=True, reasons=[], details={"error": str(e)},
            latency_s=time.perf_counter() - t0,
        )
    latency = time.perf_counter() - t0
    if passed:
        log.ok("LOCAL", f"{label} clean ({latency*1000:.0f}ms)")
    else:
        log.fail("LOCAL", f"{label} flagged: {flagged} ({latency*1000:.0f}ms)")
    return CheckResult(
        name=name,
        passed=passed,
        reasons=[f"moderation: {c}" for c in flagged],
        details={"categories": flagged, "model": config.local_moderation_model},
        latency_s=latency,
    )


# ---------------------------------------------------------------------------
# Local LLM judge — reuses LLMLocalService
# ---------------------------------------------------------------------------

_ALLOWED_EMOTIONS = {"happy", "sad", "angry", "disgust", "surprise", "neutral"}
_LOCAL_LLM_JUDGE = None  # lazy-loaded; one shared instance per process


def _get_local_judge():
    """Returns a LLMLocalService instance dedicated to judging.

    We keep it separate from the reply-generation LLM so we can swap the judge
    model (smaller / faster / fine-tuned) without touching the reply path.
    """
    global _LOCAL_LLM_JUDGE
    if _LOCAL_LLM_JUDGE is None:
        from app.services.llm.local_llm_service import LLMLocalService
        # Construct directly so a custom judge model can be configured
        _LOCAL_LLM_JUDGE = LLMLocalService()
        # Allow the judge to use a different model than reply generation.
        if config.local_llm_judge_model_name:
            _LOCAL_LLM_JUDGE.model_name = config.local_llm_judge_model_name
    return _LOCAL_LLM_JUDGE


def _extract_judge_json(raw: str) -> dict | None:
    """Pull a JSON object out of the model output even if it added prose first."""
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r"\{[\s\S]*\}(?=[^}]*$)", raw)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


async def check_llm_judge_local(
    *,
    transcript: str,
    answer: str,
    character_id: str | None,
    fallback_emotion: str | None = None,
) -> CheckResult:
    """Multi-criterion LLM judge backed by the local LLM service."""
    t0 = time.perf_counter()
    try:
        user_prompt, system_prompt = build_verifier_prompts(
            character_id=character_id, question=transcript, answer=answer,
        )
        if not user_prompt or not system_prompt:
            log.warn("LOCAL", f"llm_judge prompts missing for character={character_id} — fail-open")
            return CheckResult(
                name="models.llm_judge_local", passed=True,
                details={"reason": "prompts_missing"},
                latency_s=time.perf_counter() - t0,
            )

        judge = _get_local_judge()
        raw = await asyncio.to_thread(judge.generate_reply, user_prompt, system_prompt, None)
        if not raw:
            log.warn("LOCAL", "llm_judge empty response — fail-open")
            return CheckResult(
                name="models.llm_judge_local", passed=True,
                details={"reason": "empty_response"},
                latency_s=time.perf_counter() - t0,
            )

        parsed = _extract_judge_json(raw)
        if not parsed:
            log.warn("LOCAL", "llm_judge non-JSON response — fail-open")
            return CheckResult(
                name="models.llm_judge_local", passed=True,
                details={"reason": "non_json_response", "raw": raw[:500]},
                latency_s=time.perf_counter() - t0,
            )

        overall = bool(parsed.get("overall_pass", True))
        raw_corrected_emotion = (parsed.get("corrected_emotion") or "").strip().lower()
        corrected_emotion = (
            raw_corrected_emotion if raw_corrected_emotion in _ALLOWED_EMOTIONS else fallback_emotion
        )
        latency = time.perf_counter() - t0
        if overall:
            log.ok("LOCAL", f"llm_judge overall_pass=true ({latency:.2f}s)")
        else:
            log.fail("LOCAL", f"llm_judge overall_pass=false ({latency:.2f}s)")

        return CheckResult(
            name="models.llm_judge_local",
            passed=overall,
            reasons=[] if overall else ["llm_judge_local: overall_pass=false"],
            details={
                "raw": parsed,
                "historical_accuracy": parsed.get("historical_accuracy"),
                "appropriateness": parsed.get("appropriateness"),
                "modern_references": parsed.get("modern_references"),
                "in_character": parsed.get("in_character"),
                "corrected_answer": (parsed.get("corrected_answer") or "").strip() or None,
                "corrected_emotion": corrected_emotion,
                "model": judge.model_name,
            },
            latency_s=latency,
        )
    except Exception as e:
        log.warn("LOCAL", f"llm_judge_local error: {e} — fail-open")
        return CheckResult(
            name="models.llm_judge_local", passed=True,
            details={"error": str(e)},
            latency_s=time.perf_counter() - t0,
        )
