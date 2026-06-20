"""The combined report — offline, with injected fake faithfulness functions."""

from attest.scoring.report import evaluate
from attest.results import Report, Finding, Severity
from attest.trajectory import Step, ToolCall, Trajectory


def _fake_extract(_answer):
    return ["claim A", "claim B"]


def _fake_verify(claims, _evidence) -> list[Finding]:
    out = []
    for claim in claims:
        if "A" in claim:
            out.append(Finding(severity=Severity.PASS, verdict="supported", subject=claim))
        else:
            out.append(Finding(severity=Severity.FAIL, verdict="unsupported", subject=claim))
    return out


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
    assert isinstance(report, Report)
    assert report.by("faithfulness").score == 0.5
    assert report.by("tool_use").score == 1.0
    assert report.overall_score == 0.75
    assert report.passed is False


def test_report_serializes_to_json():
    report = evaluate(_traj(), extract=_fake_extract, verify=_fake_verify)
    js = report.model_dump_json()
    assert '"overall_score"' in js
    assert '"check"' in js
    assert '"findings"' in js
    assert '"prompt_version"' in js
