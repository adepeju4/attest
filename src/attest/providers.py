"""Provider registry: Anthropic / OpenAI / Gemini — keys, default models, clients."""

from __future__ import annotations

import os
from dataclasses import dataclass

import instructor

DEFAULT_PROVIDER = "anthropic"


@dataclass(frozen=True)
class _Provider:
    name: str
    prefix: str
    env_keys: tuple[str, ...]
    default_model: str
    models: tuple[str, ...]


_REGISTRY: dict[str, _Provider] = {
    "anthropic": _Provider(
        name="anthropic",
        prefix="anthropic",
        env_keys=("ANTHROPIC_API_KEY",),
        default_model="claude-haiku-4-5",
        models=("claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-8"),
    ),
    "openai": _Provider(
        name="openai",
        prefix="openai",
        env_keys=("OPENAI_API_KEY",),
        default_model="gpt-4o-mini",
        models=("gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"),
    ),
    "gemini": _Provider(
        name="gemini",
        prefix="google",
        env_keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        default_model="gemini-2.0-flash",
        models=("gemini-2.0-flash", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"),
    ),
}


def providers() -> list[str]:
    return list(_REGISTRY)


def _spec(provider: str) -> _Provider:
    try:
        return _REGISTRY[provider]
    except KeyError:
        raise ValueError(
            f"Unknown provider {provider!r}. Choose from: {', '.join(_REGISTRY)}."
        ) from None


def default_model(provider: str) -> str:
    return _spec(provider).default_model


def curated_models(provider: str) -> list[str]:
    return list(_spec(provider).models)


def resolve_key(provider: str, key: str | None = None) -> str | None:
    if key:
        return key
    for env in _spec(provider).env_keys:
        value = os.getenv(env)
        if value:
            return value
    return None


def build_client(
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    key: str | None = None,
) -> instructor.Instructor:
    """Build an instructor client for `provider`/`model`, key from arg or env."""
    spec = _spec(provider)
    model = model or spec.default_model
    resolved = resolve_key(provider, key)
    kwargs = {"api_key": resolved} if resolved else {}
    return instructor.from_provider(f"{spec.prefix}/{model}", **kwargs)


def _gemini_chat_models(models) -> list[str]:
    names = []
    for m in models:
        actions = getattr(m, "supported_actions", None)
        if actions is None or "generateContent" in actions:
            names.append(m.name.removeprefix("models/"))
    return names


def fetch_models(provider: str, key: str | None = None) -> list[str]:
    """Fetch the live model list from the provider's SDK (needs a key)."""
    spec = _spec(provider)
    resolved = resolve_key(provider, key)
    if not resolved:
        raise ValueError(
            f"A {provider} API key is required to list models live "
            f"(set {' or '.join(spec.env_keys)}, or pass key=...)."
        )

    if provider == "anthropic":
        from anthropic import Anthropic
        return [m.id for m in Anthropic(api_key=resolved).models.list()]

    if provider == "openai":
        from openai import OpenAI
        return sorted(m.id for m in OpenAI(api_key=resolved).models.list())

    if provider == "gemini":
        from google import genai
        return _gemini_chat_models(genai.Client(api_key=resolved).models.list())

    raise ValueError(f"Live model listing is not implemented for {provider!r}.")


def list_models(provider: str, key: str | None = None) -> list[str]:
    """Live list if a key resolves (arg or env), else the curated shortlist."""
    if resolve_key(provider, key):
        return fetch_models(provider, key)
    return curated_models(provider)
