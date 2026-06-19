"""
Tests for the LLM-backed functions WITHOUT hitting the API. We monkeypatch the
`call()` helper in each module so we test our wiring deterministically and for free.
`call()` now returns a validated Pydantic output model, so the fakes return models too.
"""

import attest.checks.judge_baseline as jb
import attest.checks.verify as v
from attest.trajectory import Trajectory
from attest.checks.verify import Verdict


def test_extract_claims_parses(monkeypatch):
    monkeypatch.setattr(v, "call", lambda **k: v.ClaimsOut(claims=["a", "b"]))
    assert v.extract_claims("an answer") == ["a", "b"]


def test_extract_claims_empty_skips_llm(monkeypatch):
    def boom(**k):
        raise AssertionError("must not call the LLM on empty input")
    monkeypatch.setattr(v, "call", boom)
    assert v.extract_claims("   ") == []


def test_grounded_verifier_parses_contradiction(monkeypatch):
    monkeypatch.setattr(
        v, "call",
        lambda **k: v.VerdictOut(verdict="unsupported", reason="Berlin 3.6M > Paris 2.1M",
                                 evidence_quote="Berlin 3,677,000"),
    )
    r = v.grounded_verifier("Paris is larger than Berlin", "Berlin 3,677,000; Paris 2,103,000")
    assert r.verdict is Verdict.UNSUPPORTED
    assert "Berlin" in r.reason
    assert r.evidence_quote == "Berlin 3,677,000"


def test_grounded_verifier_no_evidence_is_unverifiable(monkeypatch):
    def boom(**k):
        raise AssertionError("must not call the LLM when there's no evidence")
    monkeypatch.setattr(v, "call", boom)
    r = v.grounded_verifier("some claim", "   ")
    assert r.verdict is Verdict.UNVERIFIABLE


def test_naive_judge_parses(monkeypatch):
    monkeypatch.setattr(jb, "call", lambda **k: jb.JudgeVerdict(passed=True, reason="looks correct"))
    traj = Trajectory(task="t", steps=[], final_answer="x")
    verdict = jb.naive_judge(traj)
    assert verdict.passed is True
    assert verdict.reason == "looks correct"


def test_naive_judge_renders_full_trajectory():
    # The whole point of the baseline: it DOES include the narrative.
    from attest.trajectory import Step, ToolCall
    traj = Trajectory(
        task="compare",
        steps=[Step(thought="secret reasoning",
                    tool_call=ToolCall(name="get", arguments={}, output="42"))],
        final_answer="the answer is 42",
    )
    rendered = jb._render(traj)
    assert "secret reasoning" in rendered      # narrative is included (unlike attest)
    assert "42" in rendered
    assert "the answer is 42" in rendered
