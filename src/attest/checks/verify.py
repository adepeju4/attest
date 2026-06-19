"""
The heart of attest: judge a trajectory by verifying its claims against evidence.

Pipeline:
    final_answer --extract_claims--> [atomic claims]
    each claim   --verify_claim--->  SUPPORTED | UNSUPPORTED | UNVERIFIABLE
    score        = supported / (supported + unsupported)

`judge_trajectory` takes the extract/verify functions as arguments, so it stays
testable without an API. Results are Pydantic models — a full report serialises
with `.model_dump_json()` (handy for caching, storage, or pushing scores to an
observability platform).
"""

from __future__ import annotations

from enum import Enum
from typing import Callable, Literal

from pydantic import BaseModel, computed_field

from .._llm import call
from ..trajectory import Trajectory


class Verdict(str, Enum):
    SUPPORTED = "supported"        # the evidence entails the claim
    UNSUPPORTED = "unsupported"    # the evidence contradicts the claim
    UNVERIFIABLE = "unverifiable"  # the evidence is silent — can't be judged


# --- the LLM layer's tools for this module (one Pydantic model per structured call) ---
class ClaimsOut(BaseModel):
    """The list of atomic factual claims found in the text."""

    claims: list[str]


class VerdictOut(BaseModel):
    """Whether the evidence supports the claim, with a reason and the deciding span."""

    verdict: Literal["supported", "unsupported", "unverifiable"]
    reason: str = ""
    evidence_quote: str | None = None


class ClaimResult(BaseModel):
    """The verdict on one claim, with a detailed reason and the deciding evidence."""

    claim: str
    verdict: Verdict
    reason: str = ""                   # a clear, complete explanation of the verdict
    evidence_quote: str | None = None  # the exact span of evidence that decided it


class TrajectoryScore(BaseModel):
    final_answer: str = ""   # the agent's original answer — the claim attest judged
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
        """Claims we could actually adjudicate (excludes UNVERIFIABLE)."""
        return self.supported + self.unsupported

    @computed_field
    @property
    def grounding_rate(self) -> float:
        """Fraction of checkable claims supported by evidence. 1.0 if none checkable."""
        return self.supported / self.checkable if self.checkable else 1.0


# Function types you inject (real LLM versions, or fakes in tests).
ExtractFn = Callable[[str], list[str]]
VerifyFn = Callable[[str, str], ClaimResult]  # (claim, evidence) -> result


def judge_trajectory(traj: Trajectory, extract: ExtractFn, verify: VerifyFn) -> TrajectoryScore:
    """
    Verify every claim in the final answer against the trajectory's REAL tool
    outputs. Note we deliberately pass `traj.evidence()` — never the agent's
    `thought` text — so a reworded chain-of-thought can't change the verdict.
    """
    evidence = traj.evidence()
    claims = extract(traj.final_answer)
    results = [verify(claim, evidence) for claim in claims]
    return TrajectoryScore(final_answer=traj.final_answer, results=results)


# --------------------------------------------------------------------------- #
# BUILD STEP 2 — claim extraction (LLM)                                       #
# --------------------------------------------------------------------------- #
_EXTRACT_SYSTEM = (
    "You split an AI agent's final answer into atomic factual claims. "
    "Each claim must be a single, self-contained, independently-checkable "
    "statement (no 'and', no compound claims). Drop hedges, opinions, and "
    "filler — keep only verifiable assertions of fact. Preserve the original "
    "meaning; do not add information that isn't in the text."
)


def extract_claims(final_answer: str) -> list[str]:
    """
    Split a final answer into atomic, individually-checkable factual claims.

    Extraction is the *safe* thing to delegate to an LLM — it's only restating,
    not judging. (Judging is what we keep evidence-grounded in step 3.)
    """
    if not final_answer.strip():
        return []
    return call(
        system=_EXTRACT_SYSTEM,
        user=f"Final answer to decompose:\n\n{final_answer}",
        output=ClaimsOut,
    ).claims


# --------------------------------------------------------------------------- #
# BUILD STEP 3 — the grounded verifier (LLM) — THE CORE                       #
# --------------------------------------------------------------------------- #
# This maps directly onto Natural Language Inference: entailment -> SUPPORTED,
# contradiction -> UNSUPPORTED, neutral -> UNVERIFIABLE. (A v2 could swap this
# LLM call for a fast local NLI model, or add a deterministic number/date
# fast-path before ever calling a model.)
#
# Design note: a claim the evidence is *silent* about lands in UNVERIFIABLE and
# is excluded from the score — we only grade what we can actually adjudicate.
# Penalising "asserted but unverified" claims would be a deliberate v2 choice.
_VERIFY_SYSTEM = (
    "You are a strict fact-checker. Decide whether the EVIDENCE supports the CLAIM, "
    "using ONLY the evidence — no outside knowledge, no assumptions.\n"
    "- supported: the evidence directly entails the claim.\n"
    "- unsupported: the evidence contradicts the claim.\n"
    "- unverifiable: the evidence is silent or unrelated, so it cannot be judged.\n"
    "Give a clear, complete reason for your verdict, and quote the exact deciding "
    "span from the evidence. Watch direction and magnitude carefully (e.g. 'X is "
    "larger than Y' is unsupported if the evidence shows Y is larger)."
)


def grounded_verifier(claim: str, evidence: str) -> ClaimResult:
    """
    Decide whether `evidence` (the real tool outputs) entails `claim`.

    What makes this resistant to chain-of-thought gaming: the model sees ONLY the
    claim and the evidence — never the agent's reasoning — and the task is narrow
    entailment, not "is this a good answer?".
    """
    if not evidence.strip():
        return ClaimResult(
            claim=claim,
            verdict=Verdict.UNVERIFIABLE,
            reason="No tool-output evidence was recorded.",
        )
    out = call(
        system=_VERIFY_SYSTEM,
        user=f"CLAIM:\n{claim}\n\nEVIDENCE (the agent's real tool outputs):\n{evidence}",
        output=VerdictOut,
    )
    return ClaimResult(
        claim=claim,
        verdict=Verdict(out.verdict),
        reason=out.reason,
        evidence_quote=out.evidence_quote or None,
    )
