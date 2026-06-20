"""Faithfulness: verify each claim in the final answer against the real tool outputs."""

from __future__ import annotations

from typing import Callable, Literal

from pydantic import BaseModel

from .._llm import call
from ..results import CheckResult, Finding, Severity
from ..trajectory import Trajectory

_SEVERITY = {
    "supported": Severity.PASS,
    "unsupported": Severity.FAIL,
    "unverifiable": Severity.WARN,
}

MAX_CLAIMS = 40
BATCH_SIZE = 15


class ClaimsOut(BaseModel):
    """The list of atomic factual claims found in the text."""

    claims: list[str]


class VerdictOut(BaseModel):
    """Whether the evidence supports the claim, with a reason and the deciding span."""

    verdict: Literal["supported", "unsupported", "unverifiable"]
    reason: str = ""
    evidence_quote: str | None = None


class _ClaimVerdict(BaseModel):
    """A verdict for one numbered claim."""

    claim: int
    verdict: Literal["supported", "unsupported", "unverifiable"]
    reason: str = ""
    evidence_quote: str | None = None


class _BatchOut(BaseModel):
    """One verdict for each numbered claim."""

    verdicts: list[_ClaimVerdict]


ExtractFn = Callable[[str], list[str]]
VerifyFn = Callable[[list[str], str], list[Finding]]


def judge_trajectory(traj: Trajectory, extract: ExtractFn, verify: VerifyFn,
                     *, max_claims: int = MAX_CLAIMS) -> CheckResult:
    evidence = traj.evidence()
    claims = extract(traj.final_answer)
    capped = len(claims) > max_claims
    if capped:
        claims = claims[:max_claims]
    findings = verify(claims, evidence)
    supported = sum(f.verdict == "supported" for f in findings)
    unsupported = sum(f.verdict == "unsupported" for f in findings)
    checkable = supported + unsupported
    rate = supported / checkable if checkable else 1.0
    summary = f"{supported}/{checkable} checkable claims grounded."
    if capped:
        summary += f" (checked the first {max_claims} claims)"
    return CheckResult(
        check="faithfulness",
        passed=unsupported == 0,
        score=rate,
        summary=summary,
        findings=findings,
    )


_EXTRACT_SYSTEM = (
    "You split an AI agent's final answer into atomic factual claims to fact-check. "
    "Each claim must be a single, self-contained, independently-checkable statement. "
    "Consolidate: do NOT split one fact into several near-duplicate sub-claims, do NOT "
    "emit every sentence or clause, and merge overlapping points into one claim. Drop "
    "hedges, opinions, filler, questions, and generic advice. Keep only distinct, "
    "verifiable assertions of fact, and prefer a short list of meaningful claims over a "
    "long list of fragments. Preserve the original meaning; add nothing."
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


def grounded_verifier(claim: str, evidence: str, *, answer_kind: str = "factual") -> Finding:
    """Decide whether `evidence` entails one `claim`. Sees no agent reasoning."""
    if not evidence.strip():
        return Finding(
            severity=Severity.WARN,
            verdict="unverifiable",
            subject=claim,
            reason="No tool-output evidence was recorded.",
        )
    out = call(
        system=_VERIFY_SYSTEM.get(answer_kind, _VERIFY_SYSTEM["factual"]),
        user=f"CLAIM:\n{claim}\n\nEVIDENCE (the agent's real tool outputs):\n{evidence}",
        output=VerdictOut,
    )
    return Finding(
        severity=_SEVERITY[out.verdict],
        verdict=out.verdict,
        subject=claim,
        reason=out.reason,
        evidence=out.evidence_quote or None,
    )


_BATCH_INSTRUCTION = (
    "You are given several NUMBERED claims and one EVIDENCE block. Apply the standard below "
    "to EACH claim independently, using ONLY the evidence. Return exactly one verdict per "
    "claim: its number, the verdict, a brief reason, and the deciding span from the evidence. "
    "Do not skip or merge claims.\n\nSTANDARD:\n"
)


def _batch_system(answer_kind: str) -> str:
    return _BATCH_INSTRUCTION + _VERIFY_SYSTEM.get(answer_kind, _VERIFY_SYSTEM["factual"])


def _verify_batch(batch: list[str], evidence: str, answer_kind: str) -> list[Finding]:
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(batch, start=1))
    try:
        out = call(
            system=_batch_system(answer_kind),
            user=f"CLAIMS:\n{numbered}\n\nEVIDENCE (the agent's real tool outputs):\n{evidence}",
            output=_BatchOut,
            max_tokens=8192,
        )
        by_num = {v.claim: v for v in out.verdicts}
    except Exception:
        by_num = {}
    findings: list[Finding] = []
    for i, claim in enumerate(batch, start=1):
        v = by_num.get(i)
        if v is None:
            findings.append(Finding(
                severity=Severity.WARN, verdict="unverifiable", subject=claim,
                reason="No verdict was returned for this claim."))
        else:
            findings.append(Finding(
                severity=_SEVERITY[v.verdict], verdict=v.verdict, subject=claim,
                reason=v.reason, evidence=v.evidence_quote or None))
    return findings


def verify_claims(claims: list[str], evidence: str, *, answer_kind: str = "factual",
                  batch_size: int = BATCH_SIZE) -> list[Finding]:
    """Verify many claims against the evidence in batched calls (one call per batch)."""
    if not claims:
        return []
    if not evidence.strip():
        return [Finding(severity=Severity.WARN, verdict="unverifiable", subject=c,
                        reason="No tool-output evidence was recorded.") for c in claims]
    findings: list[Finding] = []
    for start in range(0, len(claims), batch_size):
        findings.extend(_verify_batch(claims[start:start + batch_size], evidence, answer_kind))
    return findings
