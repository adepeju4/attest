"""
Tool-use correctness — judge HOW the agent used its tools, separate from whether
the final answer is faithful. An agent can answer faithfully while calling the
wrong tool, ignoring a tool error, or passing junk arguments.

Three per-call checks, deterministic-first (the rule that keeps this from being the
vibes-based LLM-judge attest is built to beat):

  1. allowed        - is the tool in the agent's declared `allowed_tools`?   (deterministic)
  2. error_handled  - did the output look like a failure, and if so did a later
                      step address it rather than barrel on?                  (deterministic)
  3. appropriate    - was this a reasonable tool for the step, per its
                      description?  (optional, grounded LLM — the only paid part)

attest NEVER executes a tool. This reads the recorded trajectory only.
"""

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
    error_handled: bool | None        # None = no error occurred
    appropriate: bool | None = None   # None = not checked (fast mode)
    reason: str = ""
    evidence_quote: str | None = None


class ToolUseScore(BaseModel):
    reviews: list[ToolCallReview]
    summary: str = ""                 # single statement validating the tools called

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
        """Fraction of tool calls that were correct. 1.0 if no calls were made."""
        return self.correct / self.total if self.total else 1.0


# --- deterministic helpers (no model) --------------------------------------- #

def _looks_like_error(output: str) -> bool:
    """Conservative: empty output, or an error/exception/traceback shape."""
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
    """An error is 'handled' if the agent made another tool call after it."""
    return any(i > error_step for i, _ in traj.tool_calls())


# --- optional LLM check ------------------------------------------------------ #

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


# --- the feature ------------------------------------------------------------- #

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

    # worst issue wins
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
    """
    Review each tool call in the trajectory.

    `appropriate=False` (default, "fast") runs the two deterministic checks only —
    free and fully objective. `appropriate=True` ("thorough") also runs the grounded
    LLM tool-choice check (one model call per tool call).
    """
    allowed_set = set(traj.allowed_tools or [])
    reviews = [
        _review_call(traj, idx, tc, allowed_set, appropriate)
        for idx, tc in traj.tool_calls()
    ]
    return ToolUseScore(reviews=reviews, summary=_summarize(reviews))
