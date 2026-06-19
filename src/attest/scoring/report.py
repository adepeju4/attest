"""The combined report: run the chosen checks, return one Report of CheckResults."""

from __future__ import annotations

from functools import partial
from typing import Callable, Sequence

from ..checks.injection import check_injection
from ..checks.role import check_role_adherence
from ..checks.tool_use import check_tool_use
from ..checks.verify import extract_claims, grounded_verifier, judge_trajectory
from ..results import CheckResult, Report
from ..trajectory import Trajectory

DEFAULT_CHECKS = ("faithfulness", "tool_use")


def evaluate(
    traj: Trajectory,
    *,
    checks: Sequence[str] = DEFAULT_CHECKS,
    appropriate: bool = False,
    answer_kind: str = "factual",
    deep: bool = True,
    extract: Callable[[str], list[str]] = extract_claims,
    verify=None,
) -> Report:
    """
    Run the chosen checks and return one Report. `checks` picks any of
    "faithfulness", "tool_use", "injection", "role" (default: faithfulness + tool_use).
    Faithfulness/role and injection's `deep` pass need an API key; the tool-use and
    shallow-injection paths do not.
    """
    if verify is None:
        verify = partial(grounded_verifier, answer_kind=answer_kind)

    runners = {
        "faithfulness": lambda: judge_trajectory(traj, extract, verify),
        "tool_use": lambda: check_tool_use(traj, appropriate=appropriate),
        "injection": lambda: check_injection(traj, deep=deep),
        "role": lambda: check_role_adherence(traj),
    }
    results: list[CheckResult] = [runners[c]() for c in checks if c in runners]
    return Report(results=results)
