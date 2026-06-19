"""
Thin Anthropic wrapper. One job: get *structured* output out of the model reliably.

We force the model to call a single tool whose input_schema is the shape we want.
With `tool_choice` pinned to that tool, the model MUST return data matching the
schema — far more reliable than asking for JSON in prose and parsing it.

The client is created lazily so that `import attest` never requires an API key
(the no-API unit tests depend on this).
"""

from __future__ import annotations

from functools import lru_cache
from typing import TypeVar

from anthropic import Anthropic
from pydantic import BaseModel

DEFAULT_MODEL = "claude-haiku-4-5"

T = TypeVar("T", bound=BaseModel)


@lru_cache(maxsize=1)
def _client() -> Anthropic:
    return Anthropic()


def structured(
    *,
    system: str,
    user: str,
    tool_name: str,
    tool_description: str,
    input_schema: dict,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> dict:
    """Force the model to call `tool_name` and return its validated input dict."""
    msg = _client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        tools=[
            {
                "name": tool_name,
                "description": tool_description,
                "input_schema": input_schema,
            }
        ],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": user}],
    )
    for block in msg.content:
        if block.type == "tool_use" and block.name == tool_name:
            return block.input
    raise RuntimeError(f"Model did not call the forced tool {tool_name!r}.")


def call(
    *,
    system: str,
    user: str,
    output: type[T],
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> T:
    """
    Model-based structured call: the LLM layer's tool *is* the Pydantic `output`
    model. Its JSON schema becomes the forced tool, and we return a validated
    instance. This is how the verifiers get structured output — they hand `call`
    an output model, not a raw schema. (Keep output models simple — use `Literal`
    over enums so the schema stays inline, no `$defs`.)
    """
    result = structured(
        system=system,
        user=user,
        tool_name="respond",
        tool_description=(output.__doc__ or "Return the structured response.").strip(),
        input_schema=output.model_json_schema(),
        model=model,
        max_tokens=max_tokens,
    )
    return output.model_validate(result)
