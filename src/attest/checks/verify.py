"""Faithfulness: verify each claim in the final answer against the real tool outputs."""

from __future__ import annotations

from enum import Enum
from typing import Callable, Literal

from pydantic import BaseModel, computed_field

from .._llm import call
from ..trajectory import Trajectory


class Verdict(str, Enum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNVERIFIABLE = "unverifiable"


class ClaimsOut(BaseModel):
    """The list of atomic factual claims found in the text."""

    claims: list[str]


class VerdictOut(BaseModel):
    """Whether the evidence supports the claim, with a reason and the deciding span."""

    verdict: Literal["supported", "unsupported", "unverifiable"]
    reason: str = ""
    evidence_quote: str | None = None


class ClaimResult(BaseModel):
    claim: str
    verdict: Verdict
    reason: str = ""
    evidence_quote: str | None = None


class TrajectoryScore(BaseModel):
    final_answer: str = ""
    results: list[ClaimResult]

    @computed_field
    @property
    def supported(self) -> int:
        return sum(r.verdict is Verdict.SUPPORTED for r in self.results)

    @computed_field
    @property
    def unsupported(self) -> int:
        return sum(r.verdict is Verdict.UNSUPPORTED for r in self.results)

    @computed_field
    @property
    def checkable(self) -> int:
        return self.supported + self.unsupported

    @computed_field
    @property
    def grounding_rate(self) -> float:
        return self.supported / self.checkable if self.checkable else 1.0


ExtractFn = Callable[[str], list[str]]
VerifyFn = Callable[[str, str], ClaimResult]


def judge_trajectory(traj: Trajectory, extract: ExtractFn, verify: VerifyFn) -> TrajectoryScore:
    evidence = traj.evidence()
    claims = extract(traj.final_answer)
    results = [verify(claim, evidence) for claim in claims]
    return TrajectoryScore(final_answer=traj.final_answer, results=results)


_EXTRACT_SYSTEM = (
    "You split an AI agent's final answer into atomic factual claims. "
    "Each claim must be a single, self-contained, independently-checkable "
    "statement (no 'and', no compound claims). Drop hedges, opinions, and "
    "filler — keep only verifiable assertions of fact. Preserve the original "
    "meaning; do not add information that isn't in the text."
)


def extract_claims(final_answer: str) -> list[str]:
    """Split a final answer into atomic, individually-checkable factual claims."""
    if not final_answer.strip():
        return []
    return call(
        system=_EXTRACT_SYSTEM,
        user=f"Final answer to decompose:\n\n{final_answer}",
        output=ClaimsOut,
    ).claims


_VERIFY_SYSTEM = {
    "factual": (
        "You are a strict fact-checker. Decide whether the EVIDENCE supports the CLAIM, "
        "using ONLY the evidence — no outside knowledge, no assumptions.\n"
        "- supported: the evidence directly entails the claim.\n"
        "- unsupported: the evidence contradicts the claim.\n"
        "- unverifiable: the evidence is silent or unrelated, so it cannot be judged.\n"
        "Give a clear, complete reason for your verdict, and quote the exact deciding "
        "span from the evidence. Watch direction and magnitude carefully (e.g. 'X is "
        "larger than Y' is unsupported if the evidence shows Y is larger)."
    ),
    "plan": (
        "The answer being checked is a PLAN or recommendation, NOT a report of completed "
        "work. Decide whether the CLAIM faithfully reflects the plan in the EVIDENCE, using "
        "ONLY the evidence.\n"
        "- supported: the plan states or directly implies the claim (treat 'X will intercept "
        "requests' / 'the plan covers X' as supported if the plan describes X — even though "
        "nothing has run yet).\n"
        "- unsupported: the plan contradicts the claim.\n"
        "- unverifiable: the plan is silent on it.\n"
        "Do NOT require runtime proof; judge fidelity to the plan. Give a clear reason and "
        "quote the deciding span from the plan."
    ),
}


def grounded_verifier(claim: str, evidence: str, *, answer_kind: str = "factual") -> ClaimResult:
    """Decide whether `evidence` entails `claim`. Sees no agent reasoning."""
    if not evidence.strip():
        return ClaimResult(
            claim=claim,
            verdict=Verdict.UNVERIFIABLE,
            reason="No tool-output evidence was recorded.",
        )
    out = call(
        system=_VERIFY_SYSTEM.get(answer_kind, _VERIFY_SYSTEM["factual"]),
        user=f"CLAIM:\n{claim}\n\nEVIDENCE (the agent's real tool outputs):\n{evidence}",
        output=VerdictOut,
    )
    return ClaimResult(
        claim=claim,
        verdict=Verdict(out.verdict),
        reason=out.reason,
        evidence_quote=out.evidence_quote or None,
    )
