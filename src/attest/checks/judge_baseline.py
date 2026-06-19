"""
The naive LLM-as-judge — the baseline attest exists to beat.

This is what most eval setups do: hand the model the WHOLE trajectory (including
the agent's reasoning/narrative) and ask "is this a good answer?". Because it
reads the story, a confident-but-wrong answer can talk it into a PASS.

attest refuses to read the story — it only checks claims against real tool
outputs. This judge is here so the demo can show, side by side, what that buys you.
"""

from __future__ import annotations

from pydantic import BaseModel

from .._llm import call
from ..trajectory import Trajectory

_JUDGE_SYSTEM = (
    "You are evaluating an AI agent's work. You are shown the agent's full "
    "trajectory — its reasoning, the actions it took, and its final answer. "
    "Decide whether the final answer is correct and well-justified, and set "
    "passed=true if it is a good answer."
)


class JudgeVerdict(BaseModel):
    """Whether the agent's final answer is good, with a reason."""

    passed: bool
    reason: str = ""


def _render(traj: Trajectory) -> str:
    """Render the FULL trajectory — narrative included. That inclusion is the point."""
    lines = [f"TASK: {traj.task}", ""]
    for step in traj.steps:
        if step.thought:
            lines.append(f"Thought: {step.thought}")
        if step.tool_call:
            tc = step.tool_call
            lines.append(f"Action: {tc.name}({tc.arguments}) -> {tc.output}")
    lines += ["", f"FINAL ANSWER: {traj.final_answer}"]
    return "\n".join(lines)


def naive_judge(traj: Trajectory) -> JudgeVerdict:
    """The 'does this look good?' judge — sees everything, narrative and all."""
    return call(system=_JUDGE_SYSTEM, user=_render(traj), output=JudgeVerdict)
