"""Self-consistency voting: majority over repeated samples, to dampen judge non-determinism."""

import attest._llm as llm
from pydantic import BaseModel


class _V(BaseModel):
    verdict: str


def test_vote_returns_the_majority(monkeypatch):
    seq = iter([_V(verdict="a"), _V(verdict="b"), _V(verdict="a")])
    monkeypatch.setattr(llm, "call", lambda **k: next(seq))
    out = llm.vote(system="s", user="u", output=_V, key=lambda o: o.verdict, samples=3)
    assert out.verdict == "a"


def test_vote_one_sample_is_a_single_call(monkeypatch):
    calls = {"n": 0}

    def once(**k):
        calls["n"] += 1
        return _V(verdict="x")

    monkeypatch.setattr(llm, "call", once)
    out = llm.vote(system="s", user="u", output=_V, key=lambda o: o.verdict, samples=1)
    assert out.verdict == "x"
    assert calls["n"] == 1
