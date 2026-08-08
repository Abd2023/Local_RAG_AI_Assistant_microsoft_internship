"""Provider abstraction for Foundry Local and Ollama inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from src import config


class ProviderError(RuntimeError):
    """Raised when a configured local inference provider cannot be used."""


class ModelProvider(Protocol):
    """Common chat and embedding contract used by RAG services."""

    name: str

    def complete_chat(self, messages: list[dict[str, str]]) -> str:
        """Generate one local chat response."""

    def embed(self, text: str) -> list[float]:
        """Generate one embedding vector."""

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate multiple embedding vectors."""


@dataclass
class FoundryProvider:
    """Adapter around the existing Microsoft Foundry Local helpers."""

    name: str = "foundry"

    def complete_chat(self, messages: list[dict[str, str]]) -> str:
        from src.foundry_client import complete_chat_messages

        return complete_chat_messages(messages)

    def embed(self, text: str) -> list[float]:
        from src.foundry_client import generate_embedding

        return generate_embedding(text)

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        from src.foundry_client import generate_embeddings

        return generate_embeddings(list(texts))


@dataclass
class OllamaProvider:
    """Adapter for Ollama's OpenAI-compatible local API."""

    base_url: str = config.OLLAMA_BASE_URL
    chat_model: str = config.OLLAMA_CHAT_MODEL
    embedding_model: str = config.OLLAMA_EMBEDDING_MODEL
    name: str = "ollama"

    def _client(self):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency is installed in normal setup
            raise ProviderError("The openai package is required for the Ollama provider.") from exc
        return OpenAI(base_url=f"{self.base_url.rstrip('/')}/v1", api_key="ollama")

    def complete_chat(self, messages: list[dict[str, str]]) -> str:
        try:
            response = self._client().chat.completions.create(
                model=self.chat_model,
                messages=messages,
                temperature=0.1,
                max_tokens=250,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # pragma: no cover - requires a running Ollama service
            raise ProviderError(f"Ollama chat generation failed: {exc}") from exc

    def embed(self, text: str) -> list[float]:
        return self.embed_many([text])[0]

    def embed_many(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            response = self._client().embeddings.create(
                model=self.embedding_model,
                input=list(texts),
            )
            return [list(item.embedding) for item in response.data]
        except Exception as exc:  # pragma: no cover - requires a running Ollama service
            raise ProviderError(f"Ollama embedding generation failed: {exc}") from exc


def get_provider() -> ModelProvider:
    """Return the configured local model provider."""
    if config.RAG_PROVIDER == "foundry":
        return FoundryProvider()
    if config.RAG_PROVIDER == "ollama":
        return OllamaProvider()
    raise ProviderError(
        f"Unsupported RAG_PROVIDER '{config.RAG_PROVIDER}'. Use 'foundry' or 'ollama'."
    )


def complete_chat(messages: list[dict[str, str]]) -> str:
    """Generate a response through the configured provider."""
    return get_provider().complete_chat(messages)


def generate_embedding(text: str) -> list[float]:
    """Generate one vector through the configured provider."""
    return get_provider().embed(text)


def generate_embeddings(texts: Sequence[str]) -> list[list[float]]:
    """Generate multiple vectors through the configured provider."""
    return get_provider().embed_many(texts)
