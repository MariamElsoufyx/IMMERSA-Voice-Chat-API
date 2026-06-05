from dataclasses import dataclass, field
from typing import Any


@dataclass
class CheckResult:

    name: str                                        # e.g. "regex.profanity_question"
    passed: bool
    reasons: list[str] = field(default_factory=list)  # human-readable hits
    details: dict[str, Any] = field(default_factory=dict)
    latency_s: float | None = None


@dataclass
class AggregateResult:
    results: list[CheckResult] = field(default_factory=list)
    corrected_answer: str | None = None      # only the LLM judge fills this
    corrected_emotion: str | None = None
    total_latency_s: float | None = None

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def by_name(self, name: str) -> CheckResult | None:
        for r in self.results:
            if r.name == name:
                return r
        return None

    def all_reasons(self) -> list[str]:
        out: list[str] = []
        for r in self.results:
            if not r.passed:
                out.extend(r.reasons)
        return out
