from types import SimpleNamespace

import pytest

from src import config
from src import foundry_client
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


def test_configured_chat_model_defaults_to_phi4_mini() -> None:
    assert config.CHAT_MODEL_ALIAS == "phi-4-mini"
    assert config.OLLAMA_CHAT_MODEL == "phi4-mini"
    assert config.RERANKER_DEVICE == "cpu"
    assert config.CHAT_MAX_TOKENS == 500


def test_foundry_model_load_failure_names_the_configured_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_get_model(*_args, **_kwargs):
        raise foundry_client.FoundryLocalException("model download failed")

    monkeypatch.setattr(foundry_client, "get_chat_model", fail_get_model)

    with pytest.raises(foundry_client.FoundryLocalException, match="phi-4-mini"):
        foundry_client.load_chat_model(
            "phi-4-mini",
            register_execution_providers=False,
            require_gpu=False,
        )


def test_foundry_selects_cpu_variant_when_gpu_is_not_required() -> None:
    gpu = SimpleNamespace(
        id="gpu",
        info=SimpleNamespace(
            runtime=SimpleNamespace(
                device_type="GPU",
                execution_provider="CUDAExecutionProvider",
            )
        ),
    )
    cpu = SimpleNamespace(
        id="cpu",
        info=SimpleNamespace(
            runtime=SimpleNamespace(
                device_type="CPU",
                execution_provider="CPUExecutionProvider",
            )
        ),
    )

    class FakeModel:
        alias = "test-model"
        variants = [gpu, cpu]

        def __init__(self) -> None:
            self.selected = None

        def select_variant(self, variant) -> None:
            self.selected = variant

    model = FakeModel()

    assert foundry_client._select_preferred_model_variant(model, require_gpu=False) is model
    assert model.selected is cpu


def test_foundry_rejects_unregistered_gpu_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    execution_provider = SimpleNamespace(
        name=config.PREFERRED_EXECUTION_PROVIDER,
        is_registered=False,
    )

    class FakeManager:
        def discover_eps(self):
            return [execution_provider]

    monkeypatch.setattr(foundry_client, "platform", SimpleNamespace(system=lambda: "Windows"))
    monkeypatch.setattr(foundry_client, "get_manager", lambda: FakeManager())
    monkeypatch.setattr(
        foundry_client,
        "download_and_register_execution_providers",
        lambda **_kwargs: SimpleNamespace(status="registration failed"),
    )

    with pytest.raises(foundry_client.FoundryLocalException, match="could not be registered"):
        foundry_client.ensure_preferred_gpu_execution_provider()
