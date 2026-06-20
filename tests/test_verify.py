"""Faithfulness aggregation — fake extract/verify, no API."""

from attest.trajectory import Trajectory, Step, ToolCall
from attest.checks.verify import judge_trajectory
from attest.results import Finding, Severity


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
    assert "larger than Berlin" not in ev


def test_judge_aggregates_verdicts():
    traj = _example()
    fake_extract = lambda answer: [
        "Paris has about 2.1 million people",
        "Paris is larger than Berlin",
    ]

    def fake_verify(claim: str, evidence: str) -> Finding:
        if "larger than Berlin" in claim:
            return Finding(severity=Severity.FAIL, verdict="unsupported", subject=claim,
                           reason="Berlin 3.6M > Paris 2.1M")
        return Finding(severity=Severity.PASS, verdict="supported", subject=claim)

    result = judge_trajectory(traj, fake_extract, fake_verify)
    assert result.check == "faithfulness"
    assert result.score == 0.5
    assert result.passed is False
    assert len(result.failures) == 1


def test_unverifiable_excluded_from_rate():
    fake_extract = lambda a: ["c1", "c2", "c3"]

    def fake_verify(claim, evidence):
        return {
            "c1": Finding(severity=Severity.PASS, verdict="supported", subject="c1"),
            "c2": Finding(severity=Severity.WARN, verdict="unverifiable", subject="c2"),
            "c3": Finding(severity=Severity.PASS, verdict="supported", subject="c3"),
        }[claim]

    result = judge_trajectory(_example(), fake_extract, fake_verify)
    assert result.score == 1.0
    assert result.passed is True


def test_one_flaky_claim_degrades_to_unverifiable():
    fake_extract = lambda a: ["good claim", "boom claim"]

    def fake_verify(claim, evidence):
        if claim == "boom claim":
            raise RuntimeError("provider returned nothing")
        return Finding(severity=Severity.PASS, verdict="supported", subject=claim)

    result = judge_trajectory(_example(), fake_extract, fake_verify)
    assert result.score == 1.0  # the good claim still counts
    assert result.passed is True
    boom = [f for f in result.findings if f.subject == "boom claim"][0]
    assert boom.verdict == "unverifiable"
    assert boom.severity is Severity.WARN
