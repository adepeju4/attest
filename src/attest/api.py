"""The class entry point: Attest(provider=..., model=..., key=...).evaluate(traj)."""

from __future__ import annotations

from typing import Callable

import instructor

from ._llm import using
from .checks.injection import InjectionReport, check_injection
from .checks.judge_baseline import JudgeVerdict, naive_judge
from .checks.tool_use import ToolUseScore, check_tool_use
from .checks.verify import extract_claims
from .providers import DEFAULT_PROVIDER, build_client, default_model, list_models, providers
from .scoring.report import TrajectoryReport, evaluate
from .scoring.stats import Proportion, wilson_interval
from .trajectory import Trajectory


class Attest:
    """A configured evaluator. Pick a provider + model once; evaluate many trajectories."""

    def __init__(
        self,
        key: str | None = None,
        *,
        provider: str = DEFAULT_PROVIDER,
        model: str | None = None,
        client: instructor.Instructor | None = None,
    ) -> None:
        self.provider = provider
        self.model = model or default_model(provider)
        self._client = client or build_client(provider, self.model, key)

    def evaluate(
        self,
        traj: Trajectory,
        *,
        appropriate: bool = False,
        answer_kind: str = "factual",
        extract: Callable[[str], list[str]] = extract_claims,
        verify=None,
    ) -> TrajectoryReport:
        with using(self._client):
            return evaluate(
                traj,
                appropriate=appropriate,
                answer_kind=answer_kind,
                extract=extract,
                verify=verify,
            )

    def tool_use(self, traj: Trajectory, *, appropriate: bool = False) -> ToolUseScore:
        with using(self._client):
            return check_tool_use(traj, appropriate=appropriate)

    def injection(self, traj: Trajectory, *, deep: bool = False) -> InjectionReport:
        with using(self._client):
            return check_injection(traj, deep=deep)

    def judge(self, traj: Trajectory) -> JudgeVerdict:
        with using(self._client):
            return naive_judge(traj)

    @staticmethod
    def stats(successes: int, n: int) -> Proportion:
        return wilson_interval(successes, n)

    @staticmethod
    def providers() -> list[str]:
        return providers()

    @staticmethod
    def models(provider: str, key: str | None = None) -> list[str]:
        return list_models(provider, key)
