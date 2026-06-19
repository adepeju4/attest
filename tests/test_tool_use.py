"""
Tool-use correctness — the deterministic checks are tested with no API at all.
The optional `appropriate` LLM check is tested by mocking `call`.
"""

import attest.checks.tool_use as tu
from attest.checks.tool_use import ToolUseVerdict, check_tool_use
from attest.trajectory import Step, ToolCall, Trajectory


def _traj(*calls, allowed=None, system_prompt=None, task="do a thing"):
    """calls = (name, args, output) tuples."""
    steps = [Step(tool_call=ToolCall(name=n, arguments=a, output=o)) for n, a, o in calls]
    return Trajectory(task=task, system_prompt=system_prompt, allowed_tools=allowed,
                      steps=steps, final_answer="done")


def test_all_allowed_no_errors_is_correct():
    traj = _traj(
        ("search", {"q": "x"}, "result A"),
        ("summarize", {}, "result B"),
        allowed=["search", "summarize"],
    )
    score = check_tool_use(traj)
    assert score.total == 2
    assert score.correct == 2
    assert score.correct_rate == 1.0
    assert all(r.verdict is ToolUseVerdict.CORRECT for r in score.reviews)
    assert "Sequence: search -> summarize" in score.summary


def test_disallowed_tool_is_incorrect():
    traj = _traj(
        ("search", {}, "ok"),
        ("delete_everything", {}, "done"),  # not in allowed set
        allowed=["search", "summarize"],
    )
    score = check_tool_use(traj)
    bad = [r for r in score.reviews if r.tool == "delete_everything"][0]
    assert bad.verdict is ToolUseVerdict.INCORRECT
    assert bad.allowed is False
    assert "not in the agent's allowed tools" in bad.reason


def test_unhandled_error_is_incorrect():
    # error in the LAST tool call, then straight to the answer -> not handled
    traj = _traj(
        ("search", {}, "ok"),
        ("fetch", {}, "Error: connection refused"),
        allowed=["search", "fetch"],
    )
    score = check_tool_use(traj)
    err = [r for r in score.reviews if r.tool == "fetch"][0]
    assert err.verdict is ToolUseVerdict.INCORRECT
    assert err.error_handled is False
    assert "error" in err.reason.lower()


def test_handled_error_is_correct():
    # error, then a later tool call -> the agent did something about it
    traj = _traj(
        ("fetch", {}, "Error: timeout"),
        ("fetch", {}, "ok this time"),
        allowed=["fetch"],
    )
    score = check_tool_use(traj)
    first = score.reviews[0]
    assert first.error_handled is True
    assert first.verdict is ToolUseVerdict.CORRECT


def test_no_allowed_list_means_cannot_disallow():
    traj = _traj(("anything", {}, "ok"))  # allowed_tools is None
    score = check_tool_use(traj)
    assert score.reviews[0].allowed is True
    assert score.reviews[0].verdict is ToolUseVerdict.CORRECT


def test_no_tool_calls():
    traj = Trajectory(task="t", final_answer="answered from memory")
    score = check_tool_use(traj)
    assert score.total == 0
    assert score.correct_rate == 1.0
    assert score.summary == "No tool calls were made."


def test_appropriate_check_can_fail(monkeypatch):
    # thorough mode: mock the LLM tool-choice check to say "inappropriate"
    monkeypatch.setattr(
        tu, "call",
        lambda **k: tu._AppropriatenessOut(appropriate=False, reason="wrong tool for this"),
    )
    traj = _traj(("search", {}, "ok"), allowed=["search"], system_prompt="tools: search")
    score = check_tool_use(traj, appropriate=True)
    r = score.reviews[0]
    assert r.appropriate is False
    assert r.verdict is ToolUseVerdict.INCORRECT
    assert "wrong tool" in r.reason


def test_fast_mode_skips_llm(monkeypatch):
    def boom(**k):
        raise AssertionError("fast mode must not call the LLM")
    monkeypatch.setattr(tu, "call", boom)
    score = check_tool_use(_traj(("search", {}, "ok"), allowed=["search"]))
    assert score.reviews[0].appropriate is None  # not checked
