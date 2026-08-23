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
    assert config.REQUIRE_CHAT_GPU_MODELS is True
    assert config.REQUIRE_EMBEDDING_GPU_MODELS is False
    assert config.ALLOW_FOUNDRY_CPU_FALLBACK is False
    assert config.UNLOAD_EMBEDDING_MODEL_AFTER_USE is False
    assert config.UNLOAD_CHAT_MODEL_AFTER_USE is True
    assert config.RERANKER_DEVICE == "cpu"
    assert config.CHAT_MAX_TOKENS == 220


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


def test_foundry_chat_load_retries_same_model_on_cpu_after_gpu_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[bool] = []

    class FakeModel:
        is_cached = True
        is_loaded = False

        def __init__(self, *, should_fail: bool) -> None:
            self.should_fail = should_fail
            self.unloaded = False

        def load(self) -> None:
            if self.should_fail:
                raise RuntimeError("cuda allocation failed")
            self.is_loaded = True

        def unload(self) -> None:
            self.unloaded = True

    gpu_model = FakeModel(should_fail=True)
    cpu_model = FakeModel(should_fail=False)

    def fake_get_chat_model(_alias: str, *, require_gpu: bool) -> FakeModel:
        calls.append(require_gpu)
        return gpu_model if require_gpu else cpu_model

    monkeypatch.setattr(foundry_client, "get_chat_model", fake_get_chat_model)

    loaded = foundry_client.load_chat_model(
        "phi-4-mini",
        register_execution_providers=False,
        require_gpu=True,
        allow_cpu_fallback=True,
    )

    assert loaded is cpu_model
    assert calls == [True, False]
    assert gpu_model.unloaded is True


def test_foundry_chat_load_reports_cpu_fallback_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingModel:
        is_cached = True
        is_loaded = False

        def load(self) -> None:
            raise RuntimeError("load failed")

    monkeypatch.setattr(
        foundry_client,
        "get_chat_model",
        lambda _alias, *, require_gpu: FailingModel(),
    )

    with pytest.raises(
        foundry_client.FoundryLocalException,
        match="CPU fallback for the same model alias also failed",
    ):
        foundry_client.load_chat_model(
            "phi-4-mini",
            register_execution_providers=False,
            require_gpu=True,
            allow_cpu_fallback=True,
        )


def test_foundry_unloads_embedding_model_after_single_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeEmbeddingClient:
        def generate_embedding(self, _text: str):
            return SimpleNamespace(data=[SimpleNamespace(embedding=[1.0, 2.0])])

    class FakeModel:
        def __init__(self) -> None:
            self.unloaded = False

        def get_embedding_client(self) -> FakeEmbeddingClient:
            return FakeEmbeddingClient()

        def unload(self) -> None:
            self.unloaded = True

    model = FakeModel()
    monkeypatch.setattr(config, "RAG_PROVIDER", "foundry")
    monkeypatch.setattr(config, "UNLOAD_EMBEDDING_MODEL_AFTER_USE", True)
    monkeypatch.setattr(foundry_client, "load_embedding_model", lambda *_args, **_kwargs: model)

    assert foundry_client.generate_embedding("hello") == [1.0, 2.0]
    assert model.unloaded is True


def test_foundry_unloads_chat_model_after_chat_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeChatClient:
        def __init__(self) -> None:
            self.settings = SimpleNamespace(max_tokens=None, temperature=None)

        def complete_chat(self, _messages):
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="hello"))]
            )

    class FakeModel:
        def __init__(self) -> None:
            self.unloaded = False

        def get_chat_client(self) -> FakeChatClient:
            return FakeChatClient()

        def unload(self) -> None:
            self.unloaded = True

    model = FakeModel()
    monkeypatch.setattr(config, "RAG_PROVIDER", "foundry")
    monkeypatch.setattr(config, "UNLOAD_CHAT_MODEL_AFTER_USE", True)
    monkeypatch.setattr(foundry_client, "load_chat_model", lambda *_args, **_kwargs: model)

    assert foundry_client.complete_chat_messages([{"role": "user", "content": "hello"}]) == "hello"
    assert model.unloaded is True
