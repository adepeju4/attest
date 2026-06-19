"""call() retries transient empty/blocked provider responses, then fails clearly."""

import pytest

import attest._llm as llm
from attest.checks.verify import VerdictOut

_EMPTY = AttributeError("'NoneType' object has no attribute 'parts'")


class _FlakyClient:
    def __init__(self, fails, exc):
        self.fails = fails
        self.exc = exc
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fails:
            raise self.exc
        return VerdictOut(verdict="supported")


def test_retries_transient_empty_then_succeeds():
    client = _FlakyClient(fails=2, exc=_EMPTY)
    out = llm.call(system="s", user="u", output=VerdictOut, client=client)
    assert out.verdict == "supported"
    assert client.calls == 3


def test_raises_clear_error_when_always_empty():
    client = _FlakyClient(fails=99, exc=_EMPTY)
    with pytest.raises(RuntimeError, match="no parseable content"):
        llm.call(system="s", user="u", output=VerdictOut, client=client, attempts=2)


def test_propagates_non_empty_error_immediately():
    client = _FlakyClient(fails=99, exc=ValueError("bad schema"))
    with pytest.raises(ValueError):
        llm.call(system="s", user="u", output=VerdictOut, client=client)
    assert client.calls == 1
