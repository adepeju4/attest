"""The single result shape every check returns: CheckResult with a list of Findings."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, computed_field

PROMPT_VERSION = "2026.06.25"


class Severity(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


class Finding(BaseModel):
    severity: Severity
    verdict: str
    subject: str = ""
    reason: str = ""
    evidence: str | None = None
    step: int | None = None
    metadata: dict = {}


class CheckResult(BaseModel):
    check: str
    passed: bool
    score: float | None = None
    summary: str = ""
    findings: list[Finding] = []
    prompt_version: str = PROMPT_VERSION

    @property
    def failures(self) -> list[Finding]:
        return [f for f in self.findings if f.severity is Severity.FAIL]


class Report(BaseModel):
    results: list[CheckResult]

    @computed_field
    @property
    def overall_score(self) -> float:
        scored = [r.score for r in self.results if r.score is not None]
        return sum(scored) / len(scored) if scored else 1.0

    @computed_field
    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    def by(self, check: str) -> CheckResult | None:
        return next((r for r in self.results if r.check == check), None)
