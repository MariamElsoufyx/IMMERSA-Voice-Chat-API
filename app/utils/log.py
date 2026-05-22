"""Shared log helper. One format for every subsystem so the console is scannable.

Each line carries TWO emojis you can scan visually:

    {tag-emoji}  [TAG   ]  {level-emoji}  message
                                ↳ sub-detail

* The TAG emoji identifies the subsystem  (🧠 LLM, 📚 FAQ, 🛡️ VERIFY, ...)
* The LEVEL emoji identifies what happened (🚀 starting, ✅ ok, ❌ fail, ⚠️ warn, ℹ️ info)

Usage:
    from app.utils.log import log

    log.step("LLM",   "generating reply (model=gpt-5-nano)")
    log.ok("REGEX",   "question clean")
    log.fail("MODELS","moderation flagged: hate")
    log.warn("FAQ",   "lookup timed out")
    log.info("FAQ",   "miss (62ms)")
    log.detail("category=hate confidence=0.93")
"""

# ---------------------------------------------------------------------------
# Per-subsystem signature emoji — pick one quickly while reading the log
# ---------------------------------------------------------------------------
TAG_EMOJI = {
    "PIPE":   "🔧",   # pipeline / orchestration
    "STT":    "🎤",   # speech-to-text (input)
    "LLM":    "🧠",   # language model (narrator)
    "FAQ":    "📚",   # FAQ knowledge base
    "RAG":    "🔎",   # history retrieval (grounding the LLM)
    "REGEX":  "🔤",   # regex layer (profanity / anachronism)
    "MODELS": "🤖",   # OpenAI models layer (moderation / llm_judge)
    "VERIFY": "🛡️",   # verification orchestrator
    "TTS":    "🔊",   # text-to-speech (output)
    "DB":     "💾",   # database / persistence
    "AUDIO":  "🎵",   # audio assembly / upload / fallback
}

# ---------------------------------------------------------------------------
# Per-level glyph — colorful so failures stand out at a glance
# ---------------------------------------------------------------------------
LEVEL_EMOJI = {
    "step": "🚀",   # phase starting
    "ok":   "✅",   # passed / completed
    "fail": "❌",   # blocked / errored
    "warn": "⚠️ ",  # non-fatal / fail-open
    "info": "ℹ️ ",  # neutral status
}

# Indent for sub-detail lines — chosen to roughly match the prefix width
# `{emoji} [TAG   ] {emoji}  ` so the `↳` lines up under the message.
_DETAIL_INDENT = " " * 16


def _tag_prefix(tag: str) -> str:
    icon = TAG_EMOJI.get(tag, "🔘")
    return f"{icon} [{tag}]"


class _Logger:
    @staticmethod
    def info(tag: str, msg: str) -> None:
        print(f"{_tag_prefix(tag)} {LEVEL_EMOJI['info']} {msg}")

    @staticmethod
    def step(tag: str, msg: str) -> None:
        print(f"{_tag_prefix(tag)} {LEVEL_EMOJI['step']} {msg}")

    @staticmethod
    def ok(tag: str, msg: str) -> None:
        print(f"{_tag_prefix(tag)} {LEVEL_EMOJI['ok']} {msg}")

    @staticmethod
    def fail(tag: str, msg: str) -> None:
        print(f"{_tag_prefix(tag)} {LEVEL_EMOJI['fail']} {msg}")

    @staticmethod
    def warn(tag: str, msg: str) -> None:
        print(f"{_tag_prefix(tag)} {LEVEL_EMOJI['warn']} {msg}")

    @staticmethod
    def detail(msg: str) -> None:
        print(f"{_DETAIL_INDENT}↳ {msg}")


log = _Logger()
