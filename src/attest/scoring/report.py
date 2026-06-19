"""The combined report: faithfulness + tool-use + one overall score."""

from __future__ import annotations

from functools import partial
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
        return (self.faithfulness.grounding_rate + self.tool_use.correct_rate) / 2


def evaluate(
    traj: Trajectory,
    *,
    appropriate: bool = False,
    answer_kind: str = "factual",
    extract: Callable[[str], list[str]] = extract_claims,
    verify=None,
) -> TrajectoryReport:
    """
    Run every attest dimension and return one combined report. `answer_kind`
    ("factual" or "plan") tunes the faithfulness verifier; an injected `verify`
    overrides it. Faithfulness needs an API key; tool-use deterministic checks don't.
    """
    if verify is None:
        verify = partial(grounded_verifier, answer_kind=answer_kind)
    return TrajectoryReport(
        faithfulness=judge_trajectory(traj, extract, verify),
        tool_use=check_tool_use(traj, appropriate=appropriate),
    )
