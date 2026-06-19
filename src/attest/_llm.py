"""Provider-agnostic structured output via instructor. The single LLM chokepoint."""

from __future__ import annotations

import contextlib
from contextvars import ContextVar
from functools import lru_cache
from typing import Iterator, TypeVar

import instructor
from pydantic import BaseModel

from .providers import DEFAULT_PROVIDER, build_client

T = TypeVar("T", bound=BaseModel)

_active_client: ContextVar[instructor.Instructor | None] = ContextVar(
    "attest_client", default=None
)


@lru_cache(maxsize=1)
def _default_client() -> instructor.Instructor:
    return build_client(DEFAULT_PROVIDER)


def _resolve_client() -> instructor.Instructor:
    return _active_client.get() or _default_client()


@contextlib.contextmanager
def using(client: instructor.Instructor) -> Iterator[None]:
    """Run the enclosed calls against a specific provider-bound client."""
    token = _active_client.set(client)
    try:
        yield
    finally:
        _active_client.reset(token)


def call(
    *,
    system: str,
    user: str,
    output: type[T],
    max_tokens: int = 4096,
    client: instructor.Instructor | None = None,
    attempts: int = 3,
) -> T:
    """
    Return a validated instance of the Pydantic `output` model from the LLM.

    Thinking models (e.g. Gemini 2.5) can spend their whole output budget reasoning
    and return an empty candidate, which surfaces as a `'NoneType' ... 'parts'` crash.
    Those are transient, so retry a few times, then fail with a clear message.
    """
    c = client or _resolve_client()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_err: Exception | None = None
    for _ in range(max(1, attempts)):
        try:
            return c.create(response_model=output, max_tokens=max_tokens, messages=messages)
        except Exception as e:
            if not _is_empty_response(e):
                raise
            last_err = e
    raise RuntimeError(
        "The provider returned no parseable content after several attempts — often a "
        "thinking model exhausting its output-token budget, or a safety/recitation block. "
        "Try a larger max_tokens or a different model."
    ) from last_err


def _is_empty_response(err: Exception) -> bool:
    """An empty/blocked provider candidate (vs a real, non-retryable error)."""
    text = f"{type(err).__name__}: {err}"
    return "InstructorRetry" in text or "has no attribute 'parts'" in text
