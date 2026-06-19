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
    max_tokens: int = 1024,
    client: instructor.Instructor | None = None,
) -> T:
    """Return a validated instance of the Pydantic `output` model from the LLM."""
    return (client or _resolve_client()).create(
        response_model=output,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
