"""
The combined trajectory report — every attest dimension in one object, with a
single overall score. `evaluate()` is the one entry point most callers want.
"""

from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, computed_field

from ..checks.tool_use import ToolUseScore, check_tool_use
from ..trajectory import Trajectory
from ..checks.verify import (
    TrajectoryScore,
    extract_claims,
    grounded_verifier,
    judge_trajectory,
)


class TrajectoryReport(BaseModel):
    faithfulness: TrajectoryScore
    tool_use: ToolUseScore

    @computed_field
    @property
    def overall_score(self) -> float:
        """Mean of the faithfulness grounding rate and the tool-use correct rate."""
        return (self.faithfulness.grounding_rate + self.tool_use.correct_rate) / 2


def evaluate(
    traj: Trajectory,
    *,
    appropriate: bool = False,
    extract: Callable[[str], list[str]] = extract_claims,
    verify=grounded_verifier,
) -> TrajectoryReport:
    """
    Run every attest dimension on a trajectory and return one combined report.

    Faithfulness calls the grounded verifier (needs an API key); the tool-use
    deterministic checks do not. `appropriate=True` also runs the LLM tool-choice
    check. `extract`/`verify` are injectable so the report is testable without an API.
    """
    return TrajectoryReport(
        faithfulness=judge_trajectory(traj, extract, verify),
        tool_use=check_tool_use(traj, appropriate=appropriate),
    )
