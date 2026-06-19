"""Tool-use correctness: allowed-tool + error-handling checks, optional LLM appropriateness."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, computed_field

from .._llm import call
from ..trajectory import ToolCall, Trajectory


class ToolUseVerdict(str, Enum):
    CORRECT = "correct"
    QUESTIONABLE = "questionable"
    INCORRECT = "incorrect"


class ToolCallReview(BaseModel):
    step: int
    tool: str
    verdict: ToolUseVerdict
    allowed: bool
    error_handled: bool | None
    appropriate: bool | None = None
    reason: str = ""
    evidence_quote: str | None = None


class ToolUseScore(BaseModel):
    reviews: list[ToolCallReview]
    summary: str = ""

    @computed_field
    @property
    def total(self) -> int:
        return len(self.reviews)

    @computed_field
    @property
    def correct(self) -> int:
        return sum(r.verdict is ToolUseVerdict.CORRECT for r in self.reviews)

    @computed_field
    @property
    def correct_rate(self) -> float:
        return self.correct / self.total if self.total else 1.0


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
                 allowed_set: set[str], appropriate: bool) -> ToolCallReview:
    allowed = (not allowed_set) or (tc.name in allowed_set)
    is_error = _looks_like_error(tc.output)
    error_handled = _handled_later(traj, idx) if is_error else None

    appropriate_flag: bool | None = None
    appr_reason = ""
    if appropriate:
        out = _check_appropriate(traj, tc)
        appropriate_flag, appr_reason = out.appropriate, out.reason

    def review(verdict: ToolUseVerdict, reason: str, quote: str | None = None) -> ToolCallReview:
        return ToolCallReview(step=idx, tool=tc.name, verdict=verdict, allowed=allowed,
                              error_handled=error_handled, appropriate=appropriate_flag,
                              reason=reason, evidence_quote=quote)

    if not allowed:
        return review(ToolUseVerdict.INCORRECT,
                      f"'{tc.name}' is not in the agent's allowed tools "
                      f"{sorted(allowed_set)}.")
    if is_error and not error_handled:
        return review(ToolUseVerdict.INCORRECT,
                      "The tool returned an error that no later step addressed.",
                      tc.output.strip()[:160] or None)
    if appropriate_flag is False:
        return review(ToolUseVerdict.INCORRECT,
                      appr_reason or "Not an appropriate tool choice for this step.")
    return review(ToolUseVerdict.CORRECT,
                  appr_reason or "Allowed tool; no unhandled error.")


def _summarize(reviews: list[ToolCallReview]) -> str:
    if not reviews:
        return "No tool calls were made."
    total, correct = len(reviews), sum(r.verdict is ToolUseVerdict.CORRECT for r in reviews)
    seq = " -> ".join(r.tool for r in reviews)
    parts = [f"{correct}/{total} tool calls correct.", f"Sequence: {seq}."]
    issues = [r for r in reviews if r.verdict is not ToolUseVerdict.CORRECT]
    if issues:
        parts.append("Issues: " + "; ".join(f"step {r.step} ({r.tool}) — {r.reason}"
                                             for r in issues))
    else:
        parts.append("All tools allowed; no unhandled errors.")
    return " ".join(parts)


def check_tool_use(traj: Trajectory, *, appropriate: bool = False) -> ToolUseScore:
    """Review each tool call. `appropriate=True` adds the LLM tool-choice check."""
    allowed_set = set(traj.allowed_tools or [])
    reviews = [
        _review_call(traj, idx, tc, allowed_set, appropriate)
        for idx, tc in traj.tool_calls()
    ]
    return ToolUseScore(reviews=reviews, summary=_summarize(reviews))
