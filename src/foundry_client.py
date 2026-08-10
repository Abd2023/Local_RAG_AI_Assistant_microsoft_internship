"""Foundry Local SDK helpers for local chat and embedding inference."""

from __future__ import annotations

import platform
from typing import Any
from typing import Callable

from src import config

try:
    from foundry_local_sdk.exception import FoundryLocalException
except ImportError:  # pragma: no cover - exercised only in non-Windows Docker images
    class FoundryLocalException(RuntimeError):
        """Fallback error type when the Windows Foundry SDK is unavailable."""

ProgressCallback = Callable[[float], None]
ExecutionProviderProgressCallback = Callable[[str, float], None]

_APP_NAME = "Local_RAG_AI_Assistant"


def get_manager() -> Any:
    """Initialize and return the Foundry Local singleton manager."""
    from foundry_local_sdk import Configuration, FoundryLocalManager

    if FoundryLocalManager.instance is None:
        FoundryLocalManager.initialize(Configuration(app_name=_APP_NAME))
    return FoundryLocalManager.instance


def download_and_register_execution_providers(
    names: list[str] | None = None,
    progress_callback: ExecutionProviderProgressCallback | None = None,
) -> object | None:
    """Download/register Windows execution providers when running on Windows."""
    if platform.system() != "Windows":
        return None

    manager = get_manager()
    return manager.download_and_register_eps(names=names, progress_callback=progress_callback)


def ensure_preferred_gpu_execution_provider() -> None:
    """Register the configured GPU execution provider for the current process."""
    if platform.system() != "Windows":
        raise FoundryLocalException("GPU model selection currently requires Windows Foundry Local execution providers.")

    manager = get_manager()
    execution_providers = manager.discover_eps()
    preferred = next(
        (ep for ep in execution_providers if ep.name == config.PREFERRED_EXECUTION_PROVIDER),
        None,
    )
    if preferred is None:
        raise FoundryLocalException(
            f"Preferred execution provider not discovered: {config.PREFERRED_EXECUTION_PROVIDER}"
        )
    if not preferred.is_registered:
        result = download_and_register_execution_providers(names=[config.PREFERRED_EXECUTION_PROVIDER])
        refreshed = next(
            (ep for ep in manager.discover_eps() if ep.name == config.PREFERRED_EXECUTION_PROVIDER),
            None,
        )
        if refreshed is None or not refreshed.is_registered:
            details = getattr(result, "status", result)
            raise FoundryLocalException(
                "Preferred execution provider could not be registered: "
                f"{config.PREFERRED_EXECUTION_PROVIDER}. Details: {details}"
            )


def _is_preferred_gpu_variant(model: Any) -> bool:
    runtime = model.info.runtime
    if runtime is None:
        return False
    return (
        str(runtime.device_type).upper() == "GPU"
        and runtime.execution_provider == config.PREFERRED_EXECUTION_PROVIDER
    )


def _is_cpu_variant(model: Any) -> bool:
    runtime = model.info.runtime
    if runtime is None:
        return False
    return (
        str(runtime.device_type).upper() == "CPU"
        or str(runtime.execution_provider).upper() == "CPUEXECUTIONPROVIDER"
    )


def _select_preferred_model_variant(model: Any, *, require_gpu: bool) -> Any:
    if require_gpu:
        gpu_variant = next((variant for variant in model.variants if _is_preferred_gpu_variant(variant)), None)
        if gpu_variant is not None:
            model.select_variant(gpu_variant)
            return model

        variant_ids = ", ".join(variant.id for variant in model.variants)
        raise FoundryLocalException(
            "No GPU variant found for model alias "
            f"'{model.alias}' with execution provider '{config.PREFERRED_EXECUTION_PROVIDER}'. "
            f"Available variants: {variant_ids}"
        )

    cpu_variant = next((variant for variant in model.variants if _is_cpu_variant(variant)), None)
    if cpu_variant is not None:
        model.select_variant(cpu_variant)
        return model

    variant_ids = ", ".join(variant.id for variant in model.variants)
    raise FoundryLocalException(
        "No CPU variant found for model alias "
        f"'{model.alias}'. Available variants: {variant_ids}"
    )


def _download_and_load_model(
    model: Any,
    progress_callback: ProgressCallback | None = None,
) -> Any:
    if not model.is_cached:
        model.download(progress_callback=progress_callback)
    if not model.is_loaded:
        model.load()
    return model


def _try_unload_model(model: Any | None) -> None:
    if model is None:
        return
    unload = getattr(model, "unload", None)
    if callable(unload):
        try:
            unload()
        except Exception:
            pass


def _load_cpu_fallback_model(
    *,
    model_alias: str,
    get_model: Callable[..., Any],
    progress_callback: ProgressCallback | None,
) -> Any:
    cpu_model = get_model(model_alias, require_gpu=False)
    return _download_and_load_model(cpu_model, progress_callback=progress_callback)


def get_chat_model(
    model_alias: str = config.CHAT_MODEL_ALIAS,
    require_gpu: bool = config.REQUIRE_CHAT_GPU_MODELS,
) -> Any:
    """Return the configured chat model from the Foundry Local catalog."""
    if require_gpu:
        ensure_preferred_gpu_execution_provider()

    manager = get_manager()
    model = manager.catalog.get_model(model_alias)
    if model is None:
        raise FoundryLocalException(f"Chat model alias not found in Foundry Local catalog: {model_alias}")
    return _select_preferred_model_variant(model, require_gpu=require_gpu)


def get_embedding_model(
    model_alias: str = config.EMBEDDING_MODEL_ALIAS,
    require_gpu: bool = config.REQUIRE_EMBEDDING_GPU_MODELS,
) -> Any:
    """Return the configured embedding model from the Foundry Local catalog."""
    if require_gpu:
        ensure_preferred_gpu_execution_provider()

    manager = get_manager()
    model = manager.catalog.get_model(model_alias)
    if model is None:
        raise FoundryLocalException(f"Embedding model alias not found in Foundry Local catalog: {model_alias}")
    return _select_preferred_model_variant(model, require_gpu=require_gpu)


def load_chat_model(
    model_alias: str = config.CHAT_MODEL_ALIAS,
    progress_callback: ProgressCallback | None = None,
    register_execution_providers: bool = config.REQUIRE_CHAT_GPU_MODELS,
    require_gpu: bool = config.REQUIRE_CHAT_GPU_MODELS,
    allow_cpu_fallback: bool = config.ALLOW_FOUNDRY_CPU_FALLBACK,
) -> IModel:
    """Download and load the configured chat model for local inference."""
    model = None
    try:
        if register_execution_providers:
            ensure_preferred_gpu_execution_provider()

        model = get_chat_model(model_alias, require_gpu=require_gpu)
        return _download_and_load_model(model, progress_callback=progress_callback)
    except Exception as exc:
        if require_gpu and allow_cpu_fallback:
            _try_unload_model(model)
            try:
                return _load_cpu_fallback_model(
                    model_alias=model_alias,
                    get_model=get_chat_model,
                    progress_callback=progress_callback,
                )
            except Exception as cpu_exc:
                raise FoundryLocalException(
                    f"Unable to load configured Foundry Local chat model alias '{model_alias}'. "
                    "The GPU variant failed, then the CPU fallback for the same model alias also failed. "
                    "Close other GPU-heavy apps, or set FOUNDRY_REQUIRE_CHAT_GPU=false to use CPU directly. "
                    f"GPU details: {exc}. CPU fallback details: {cpu_exc}"
                ) from cpu_exc
        raise FoundryLocalException(
            f"Unable to load configured Foundry Local chat model alias '{model_alias}'. "
            "Download/cache the model in Foundry Local and verify the selected variant is available. "
            f"Details: {exc}"
        ) from exc


def load_embedding_model(
    model_alias: str = config.EMBEDDING_MODEL_ALIAS,
    progress_callback: ProgressCallback | None = None,
    register_execution_providers: bool = config.REQUIRE_EMBEDDING_GPU_MODELS,
    require_gpu: bool = config.REQUIRE_EMBEDDING_GPU_MODELS,
) -> IModel:
    """Download and load the configured embedding model for local inference."""
    try:
        if register_execution_providers:
            ensure_preferred_gpu_execution_provider()

        model = get_embedding_model(model_alias, require_gpu=require_gpu)
        return _download_and_load_model(model, progress_callback=progress_callback)
    except Exception as exc:
        raise FoundryLocalException(
            f"Unable to load configured Foundry Local embedding model alias '{model_alias}'. "
            "Download/cache the model in Foundry Local and verify the selected variant is available. "
            f"Details: {exc}"
        ) from exc


def complete_chat_prompt(
    prompt: str,
    model_alias: str = config.CHAT_MODEL_ALIAS,
    max_tokens: int = config.CHAT_MAX_TOKENS,
    temperature: float = 0.2,
    register_execution_providers: bool = config.REQUIRE_CHAT_GPU_MODELS,
    require_gpu: bool = config.REQUIRE_CHAT_GPU_MODELS,
) -> str:
    """Send one user prompt to the local chat model and return the response text."""
    if config.RAG_PROVIDER != "foundry":
        from src.providers import complete_chat

        return complete_chat([{"role": "user", "content": prompt}])

    model = load_chat_model(
        model_alias,
        register_execution_providers=register_execution_providers,
        require_gpu=require_gpu,
    )
    chat_client = model.get_chat_client()
    chat_client.settings.max_tokens = max_tokens
    chat_client.settings.temperature = temperature

    completion = chat_client.complete_chat(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ]
    )

    return completion.choices[0].message.content or ""


def complete_chat_messages(
    messages: list[dict[str, str]],
    model_alias: str = config.CHAT_MODEL_ALIAS,
    max_tokens: int = config.CHAT_MAX_TOKENS,
    temperature: float = 0.1,
    register_execution_providers: bool = config.REQUIRE_CHAT_GPU_MODELS,
    require_gpu: bool = config.REQUIRE_CHAT_GPU_MODELS,
) -> str:
    """Send chat messages to the local chat model and return the response text."""
    if config.RAG_PROVIDER != "foundry":
        from src.providers import complete_chat

        return complete_chat(messages)

    model = load_chat_model(
        model_alias,
        register_execution_providers=register_execution_providers,
        require_gpu=require_gpu,
    )
    chat_client = model.get_chat_client()
    chat_client.settings.max_tokens = max_tokens
    chat_client.settings.temperature = temperature

    completion = chat_client.complete_chat(messages)
    return completion.choices[0].message.content or ""


def generate_embedding(
    text: str,
    model_alias: str = config.EMBEDDING_MODEL_ALIAS,
    register_execution_providers: bool = config.REQUIRE_EMBEDDING_GPU_MODELS,
    require_gpu: bool = config.REQUIRE_EMBEDDING_GPU_MODELS,
) -> list[float]:
    """Generate one local embedding vector for a single text string."""
    if config.RAG_PROVIDER != "foundry":
        from src.providers import generate_embedding as provider_generate_embedding

        return provider_generate_embedding(text)

    model = load_embedding_model(
        model_alias,
        register_execution_providers=register_execution_providers,
        require_gpu=require_gpu,
    )
    embedding_client = model.get_embedding_client()
    response = embedding_client.generate_embedding(text)
    return list(response.data[0].embedding)


def generate_embeddings(
    texts: list[str],
    model_alias: str = config.EMBEDDING_MODEL_ALIAS,
    register_execution_providers: bool = config.REQUIRE_EMBEDDING_GPU_MODELS,
    require_gpu: bool = config.REQUIRE_EMBEDDING_GPU_MODELS,
) -> list[list[float]]:
    """Generate local embedding vectors for multiple text strings."""
    if config.RAG_PROVIDER != "foundry":
        from src.providers import generate_embeddings as provider_generate_embeddings

        return provider_generate_embeddings(texts)

    model = load_embedding_model(
        model_alias,
        register_execution_providers=register_execution_providers,
        require_gpu=require_gpu,
    )
    embedding_client = model.get_embedding_client()
    response = embedding_client.generate_embeddings(texts)
    return [list(item.embedding) for item in response.data]


if __name__ == "__main__":
    print(complete_chat_prompt("Say hello in one sentence."))
