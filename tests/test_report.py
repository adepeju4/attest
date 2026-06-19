"""The combined report — offline, with injected fake faithfulness functions."""

from attest.scoring.report import evaluate, TrajectoryReport
from attest.trajectory import Step, ToolCall, Trajectory
from attest.checks.verify import ClaimResult, Verdict


def _fake_extract(_answer):
    return ["claim A", "claim B"]


def _fake_verify(claim, _evidence):
    verdict = Verdict.SUPPORTED if "A" in claim else Verdict.UNSUPPORTED
    return ClaimResult(claim=claim, verdict=verdict)


def _traj():
    return Trajectory(
        task="do it",
        allowed_tools=["search", "summarize"],
        steps=[
            Step(tool_call=ToolCall(name="search", arguments={}, output="found it")),
            Step(tool_call=ToolCall(name="summarize", arguments={}, output="summary")),
        ],
        final_answer="A and B",
    )


def test_evaluate_combines_both_dimensions():
    report = evaluate(_traj(), extract=_fake_extract, verify=_fake_verify)
    assert isinstance(report, TrajectoryReport)
    assert report.faithfulness.grounding_rate == 0.5
    assert report.tool_use.correct_rate == 1.0
    assert report.overall_score == 0.75


def test_report_serializes_to_json():
    report = evaluate(_traj(), extract=_fake_extract, verify=_fake_verify)
    js = report.model_dump_json()
    assert '"overall_score"' in js
    assert '"grounding_rate"' in js
    assert '"correct_rate"' in js
    assert '"summary"' in js
