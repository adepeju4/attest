"""Provider registry — all offline (live fetch tested with faked SDK clients)."""

import sys
import types

import pytest

from attest.providers import (
    _gemini_chat_models,
    build_client,
    curated_models,
    default_model,
    fetch_models,
    list_models,
    providers,
    resolve_key,
)


def test_registry_basics():
    assert set(providers()) == {"anthropic", "openai", "gemini"}
    assert default_model("anthropic") == "claude-haiku-4-5"
    for prov in providers():
        assert curated_models(prov)


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        default_model("bogus")


def test_resolve_key_prefers_argument_then_env(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert resolve_key("anthropic", "explicit") == "explicit"
    assert resolve_key("anthropic") is None
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert resolve_key("anthropic") == "from-env"


def test_gemini_falls_back_to_google_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    assert resolve_key("gemini") == "g-key"


def test_list_models_without_key_is_curated(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert list_models("openai") == curated_models("openai")


def test_list_models_with_key_goes_live(monkeypatch):
    _install_fake_sdk(monkeypatch, "openai", "OpenAI",
                      [_Model(id="gpt-live")])
    assert list_models("openai", key="k") == ["gpt-live"]


def test_build_client_anthropic_no_network():
    client = build_client("anthropic", "claude-haiku-4-5", key="sk-ant-fake")
    assert client is not None


class _Model:
    def __init__(self, id=None, name=None, supported_actions=None):
        self.id = id
        self.name = name
        self.supported_actions = supported_actions


class _FakeSDK:
    instances = []

    def __init__(self, *, api_key, items):
        self.api_key = api_key
        self.models = types.SimpleNamespace(list=lambda: items)
        _FakeSDK.instances.append(self)


def _install_fake_sdk(monkeypatch, module_path, cls_name, items):
    captured = {}

    def factory(*, api_key):
        captured["api_key"] = api_key
        return _FakeSDK(api_key=api_key, items=items)

    mod = types.ModuleType(module_path)
    setattr(mod, cls_name, factory)
    monkeypatch.setitem(sys.modules, module_path, mod)
    return captured


def test_fetch_anthropic_returns_ids(monkeypatch):
    captured = _install_fake_sdk(monkeypatch, "anthropic", "Anthropic",
                                 [_Model(id="claude-x"), _Model(id="claude-y")])
    assert fetch_models("anthropic", key="k") == ["claude-x", "claude-y"]
    assert captured["api_key"] == "k"


def test_fetch_openai_sorts_ids(monkeypatch):
    _install_fake_sdk(monkeypatch, "openai", "OpenAI",
                      [_Model(id="gpt-z"), _Model(id="gpt-a")])
    assert fetch_models("openai", key="k") == ["gpt-a", "gpt-z"]


def test_gemini_filters_chat_models_and_strips_prefix():
    models = [
        _Model(name="models/gemini-2.0-flash", supported_actions=["generateContent"]),
        _Model(name="models/embedding-001", supported_actions=["embedContent"]),
        _Model(name="models/gemini-legacy", supported_actions=None),
    ]
    assert _gemini_chat_models(models) == ["gemini-2.0-flash", "gemini-legacy"]


def test_fetch_requires_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError):
        fetch_models("openai")
