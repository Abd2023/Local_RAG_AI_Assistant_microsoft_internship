import pytest

from src import config
from src.providers import FoundryProvider, OllamaProvider, ProviderError, get_provider


def test_provider_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "RAG_PROVIDER", "foundry")
    assert isinstance(get_provider(), FoundryProvider)

    monkeypatch.setattr(config, "RAG_PROVIDER", "ollama")
    assert isinstance(get_provider(), OllamaProvider)


def test_provider_selection_rejects_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "RAG_PROVIDER", "other")
    with pytest.raises(ProviderError, match="Unsupported"):
        get_provider()
