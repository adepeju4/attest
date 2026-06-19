"""The naive LLM-as-judge baseline: reads the full trajectory and grades the answer."""

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
    return call(system=_JUDGE_SYSTEM, user=_render(traj), output=JudgeVerdict)
