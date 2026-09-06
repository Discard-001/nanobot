"""Tests for lazy provider exports from nanobot.providers."""

from __future__ import annotations

import importlib
import sys

import nanobot


def _restore_parent_binding(original: object | None) -> None:
    """Re-importing nanobot.providers rebinds the `providers` attribute on the
    parent `nanobot` package to a fresh module object, while monkeypatch only
    restores the sys.modules entry at teardown. Restore the parent binding too,
    otherwise the two stay inconsistent and later tests resolving submodules
    through the parent (e.g. monkeypatch string targets) break.
    """
    if original is not None:
        setattr(nanobot, "providers", original)
    else:
        # Nothing will remain in sys.modules after teardown; drop the fresh
        # binding so the parent package does not keep it alive.
        if hasattr(nanobot, "providers"):
            delattr(nanobot, "providers")


def test_importing_providers_package_is_lazy(monkeypatch) -> None:
    original = getattr(nanobot, "providers", None)
    monkeypatch.delitem(sys.modules, "nanobot.providers", raising=False)
    monkeypatch.delitem(sys.modules, "nanobot.providers.anthropic_provider", raising=False)
    monkeypatch.delitem(sys.modules, "nanobot.providers.openai_compat_provider", raising=False)
    monkeypatch.delitem(sys.modules, "nanobot.providers.openai_codex_provider", raising=False)
    monkeypatch.delitem(sys.modules, "nanobot.providers.github_copilot_provider", raising=False)
    monkeypatch.delitem(sys.modules, "nanobot.providers.azure_openai_provider", raising=False)
    monkeypatch.delitem(sys.modules, "nanobot.providers.bedrock_provider", raising=False)

    providers = importlib.import_module("nanobot.providers")

    assert "nanobot.providers.anthropic_provider" not in sys.modules
    assert "nanobot.providers.openai_compat_provider" not in sys.modules
    assert "nanobot.providers.openai_codex_provider" not in sys.modules
    assert "nanobot.providers.github_copilot_provider" not in sys.modules
    assert "nanobot.providers.azure_openai_provider" not in sys.modules
    assert "nanobot.providers.bedrock_provider" not in sys.modules
    assert providers.__all__ == [
        "LLMProvider",
        "LLMResponse",
        "AnthropicProvider",
        "OpenAICompatProvider",
        "OpenAICodexProvider",
        "GitHubCopilotProvider",
        "AzureOpenAIProvider",
        "BedrockProvider",
    ]
    _restore_parent_binding(original)


def test_explicit_provider_import_still_works(monkeypatch) -> None:
    original = getattr(nanobot, "providers", None)
    monkeypatch.delitem(sys.modules, "nanobot.providers", raising=False)
    monkeypatch.delitem(sys.modules, "nanobot.providers.anthropic_provider", raising=False)

    namespace: dict[str, object] = {}
    exec("from nanobot.providers import AnthropicProvider", namespace)

    assert namespace["AnthropicProvider"].__name__ == "AnthropicProvider"
    assert "nanobot.providers.anthropic_provider" in sys.modules
    _restore_parent_binding(original)


def test_openai_codex_supports_progress_deltas() -> None:
    from nanobot.providers.openai_codex_provider import OpenAICodexProvider

    assert OpenAICodexProvider.supports_progress_deltas is True
