"""Faithfulness aggregation and the batched verifier, all offline."""

import attest.checks.verify as v
from attest.trajectory import Trajectory, Step, ToolCall
from attest.checks.verify import judge_trajectory, verify_claims
from attest.results import Finding, Severity


def _example() -> Trajectory:
    return Trajectory(
        task="pop compare",
        steps=[
            Step(tool_call=ToolCall(name="get_population",
                                    arguments={"city": "Paris"}, output="Paris: 2,103,000")),
            Step(tool_call=ToolCall(name="get_population",
                                    arguments={"city": "Berlin"}, output="Berlin: 3,677,000")),
        ],
        final_answer="Paris has ~2.1M people. Paris is larger than Berlin.",
    )


def test_evidence_excludes_thoughts():
    traj = _example()
    ev = traj.evidence()
    assert "2,103,000" in ev and "3,677,000" in ev
    assert "larger than Berlin" not in ev


def test_judge_aggregates_verdicts():
    fake_extract = lambda answer: [
        "Paris has about 2.1 million people",
        "Paris is larger than Berlin",
    ]

    def fake_verify(claims, evidence):
        out = []
        for c in claims:
            if "larger than Berlin" in c:
                out.append(Finding(severity=Severity.FAIL, verdict="unsupported", subject=c,
                                   reason="Berlin 3.6M > Paris 2.1M"))
            else:
                out.append(Finding(severity=Severity.PASS, verdict="supported", subject=c))
        return out

    result = judge_trajectory(_example(), fake_extract, fake_verify)
    assert result.check == "faithfulness"
    assert result.score == 0.5
    assert result.passed is False
    assert len(result.failures) == 1


def test_unverifiable_excluded_from_rate():
    fake_extract = lambda a: ["c1", "c2", "c3"]

    def fake_verify(claims, evidence):
        m = {
            "c1": Finding(severity=Severity.PASS, verdict="supported", subject="c1"),
            "c2": Finding(severity=Severity.WARN, verdict="unverifiable", subject="c2"),
            "c3": Finding(severity=Severity.PASS, verdict="supported", subject="c3"),
        }
        return [m[c] for c in claims]

    result = judge_trajectory(_example(), fake_extract, fake_verify)
    assert result.score == 1.0
    assert result.passed is True


def test_max_claims_caps_the_work():
    fake_extract = lambda a: [f"claim {i}" for i in range(100)]
    seen = {}

    def fake_verify(claims, evidence):
        seen["n"] = len(claims)
        return [Finding(severity=Severity.PASS, verdict="supported", subject=c) for c in claims]

    result = judge_trajectory(_example(), fake_extract, fake_verify, max_claims=10)
    assert seen["n"] == 10
    assert len(result.findings) == 10
    assert "first 10 claims" in result.summary


def test_verify_claims_batches_and_maps(monkeypatch):
    monkeypatch.setattr(v, "call", lambda **k: v._BatchOut(verdicts=[
        v._ClaimVerdict(claim=1, verdict="supported", reason="ok"),
        v._ClaimVerdict(claim=2, verdict="unsupported", reason="no", evidence_quote="x"),
    ]))
    findings = verify_claims(["a", "b"], "some evidence")
    assert [f.verdict for f in findings] == ["supported", "unsupported"]
    assert findings[0].subject == "a" and findings[1].subject == "b"
    assert findings[1].evidence == "x"


def test_verify_claims_missing_verdict_degrades(monkeypatch):
    monkeypatch.setattr(v, "call", lambda **k: v._BatchOut(verdicts=[
        v._ClaimVerdict(claim=1, verdict="supported"),
    ]))
    findings = verify_claims(["a", "b"], "evidence")
    assert findings[0].verdict == "supported"
    assert findings[1].verdict == "unverifiable"


def test_verify_claims_error_degrades_whole_batch(monkeypatch):
    def boom(**k):
        raise RuntimeError("provider died")
    monkeypatch.setattr(v, "call", boom)
    findings = verify_claims(["a", "b"], "evidence")
    assert [f.verdict for f in findings] == ["unverifiable", "unverifiable"]


def test_verify_claims_no_evidence_skips_llm(monkeypatch):
    def boom(**k):
        raise AssertionError("must not call the LLM without evidence")
    monkeypatch.setattr(v, "call", boom)
    findings = verify_claims(["a"], "   ")
    assert findings[0].verdict == "unverifiable"
