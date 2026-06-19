"""Tool-use correctness: allowed-tool + error-handling checks, optional LLM appropriateness."""

from __future__ import annotations

from pydantic import BaseModel

from .._llm import call
from ..results import CheckResult, Finding, Severity
from ..trajectory import ToolCall, Trajectory


def _looks_like_error(output: str) -> bool:
    o = output.strip()
    if not o:
        return True
    low = o.lower()
    return (
        low.startswith(("error", "exception", "traceback"))
        or "traceback (most recent call last)" in low
        or "error:" in low
        or "exception:" in low
    )


def _handled_later(traj: Trajectory, error_step: int) -> bool:
    return any(i > error_step for i, _ in traj.tool_calls())


class _AppropriatenessOut(BaseModel):
    """Whether the tool call was a reasonable choice for the step."""

    appropriate: bool
    reason: str = ""


_APPROPRIATE_SYSTEM = (
    "You judge whether an AI agent chose an APPROPRIATE TOOL for a step — not "
    "whether the final answer is good. Given the task, the tools available to the "
    "agent (with descriptions), and the specific call it made, decide whether "
    "calling that tool with those arguments was a reasonable step toward the task. "
    "Judge tool CHOICE only."
)


def _check_appropriate(traj: Trajectory, tc: ToolCall) -> _AppropriatenessOut:
    user = (
        f"TASK:\n{traj.task}\n\n"
        f"TOOLS AVAILABLE (from the agent's system prompt):\n"
        f"{traj.system_prompt or '(no tool descriptions provided)'}\n\n"
        f"TOOL CALL:\n{tc.name}({tc.arguments})"
    )
    return call(system=_APPROPRIATE_SYSTEM, user=user, output=_AppropriatenessOut)


def _review_call(traj: Trajectory, idx: int, tc: ToolCall,
                 allowed_set: set[str], appropriate: bool) -> Finding:
    allowed = (not allowed_set) or (tc.name in allowed_set)
    is_error = _looks_like_error(tc.output)
    error_handled = _handled_later(traj, idx) if is_error else None

    appropriate_flag: bool | None = None
    appr_reason = ""
    if appropriate:
        out = _check_appropriate(traj, tc)
        appropriate_flag, appr_reason = out.appropriate, out.reason

    meta = {"allowed": allowed, "error_handled": error_handled, "appropriate": appropriate_flag}

    def finding(verdict: str, severity: Severity, reason: str, evidence: str | None = None) -> Finding:
        return Finding(severity=severity, verdict=verdict, subject=tc.name, step=idx,
                       reason=reason, evidence=evidence, metadata=meta)

    if not allowed:
        return finding("incorrect", Severity.FAIL,
                       f"'{tc.name}' is not in the agent's allowed tools {sorted(allowed_set)}.")
    if is_error and not error_handled:
        return finding("incorrect", Severity.FAIL,
                       "The tool returned an error that no later step addressed.",
                       tc.output.strip()[:160] or None)
    if appropriate_flag is False:
        return finding("incorrect", Severity.FAIL,
                       appr_reason or "Not an appropriate tool choice for this step.")
    return finding("correct", Severity.PASS, appr_reason or "Allowed tool; no unhandled error.")


def _summarize(findings: list[Finding]) -> str:
    if not findings:
        return "No tool calls were made."
    total = len(findings)
    correct = sum(f.verdict == "correct" for f in findings)
    seq = " -> ".join(f.subject for f in findings)
    parts = [f"{correct}/{total} tool calls correct.", f"Sequence: {seq}."]
    issues = [f for f in findings if f.verdict != "correct"]
    if issues:
        parts.append("Issues: " + "; ".join(f"step {f.step} ({f.subject}) — {f.reason}" for f in issues))
    else:
        parts.append("All tools allowed; no unhandled errors.")
    return " ".join(parts)


def check_tool_use(traj: Trajectory, *, appropriate: bool = False) -> CheckResult:
    """Review each tool call. `appropriate=True` adds the LLM tool-choice check."""
    allowed_set = set(traj.allowed_tools or [])
    findings = [_review_call(traj, idx, tc, allowed_set, appropriate)
                for idx, tc in traj.tool_calls()]
    total = len(findings)
    correct = sum(f.verdict == "correct" for f in findings)
    rate = correct / total if total else 1.0
    return CheckResult(
        check="tool_use",
        passed=all(f.severity is not Severity.FAIL for f in findings),
        score=rate,
        summary=_summarize(findings),
        findings=findings,
    )
