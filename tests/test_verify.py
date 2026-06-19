"""
Tests for the aggregation core — using FAKE extract/verify functions so the whole
judging pipeline is covered without any API calls. This is the pattern you'll keep
using: the LLM is injected, so the logic around it stays deterministic and testable.
"""

from attest.trajectory import Trajectory, Step, ToolCall
from attest.checks.verify import judge_trajectory, ClaimResult, Verdict, TrajectoryScore


def _example() -> Trajectory:
    return Trajectory(
        task="pop compare",
        steps=[
            Step(tool_call=ToolCall(name="get_population",
                                    arguments={"city": "Paris"},
                                    output="Paris: 2,103,000")),
            Step(tool_call=ToolCall(name="get_population",
                                    arguments={"city": "Berlin"},
                                    output="Berlin: 3,677,000")),
        ],
        final_answer="Paris has ~2.1M people. Paris is larger than Berlin.",
    )


def test_evidence_excludes_thoughts():
    traj = _example()
    ev = traj.evidence()
    assert "2,103,000" in ev and "3,677,000" in ev
    assert "larger than Berlin" not in ev  # final answer is never evidence


def test_judge_aggregates_verdicts():
    traj = _example()

    fake_extract = lambda answer: [
        "Paris has about 2.1 million people",   # supported
        "Paris is larger than Berlin",          # unsupported (Berlin is bigger)
    ]

    def fake_verify(claim: str, evidence: str) -> ClaimResult:
        if "larger than Berlin" in claim:
            return ClaimResult(claim=claim, verdict=Verdict.UNSUPPORTED,
                               reason="Berlin 3.6M > Paris 2.1M")
        return ClaimResult(claim=claim, verdict=Verdict.SUPPORTED, reason="matches evidence")

    score: TrajectoryScore = judge_trajectory(traj, fake_extract, fake_verify)
    assert score.supported == 1
    assert score.unsupported == 1
    assert score.checkable == 2
    assert score.grounding_rate == 0.5


def test_unverifiable_excluded_from_rate():
    traj = _example()
    fake_extract = lambda a: ["c1", "c2", "c3"]

    def fake_verify(claim, evidence):
        return {
            "c1": ClaimResult(claim="c1", verdict=Verdict.SUPPORTED),
            "c2": ClaimResult(claim="c2", verdict=Verdict.UNVERIFIABLE),
            "c3": ClaimResult(claim="c3", verdict=Verdict.SUPPORTED),
        }[claim]

    score = judge_trajectory(traj, fake_extract, fake_verify)
    assert score.checkable == 2          # UNVERIFIABLE doesn't count
    assert score.grounding_rate == 1.0   # 2 supported / 2 checkable
