"""Cross-encoder reranking for retrieved document chunks."""

from __future__ import annotations

from dataclasses import replace
from typing import Protocol, Sequence

from src import config
from src.retrieval import RetrievalResult


class RerankerUnavailableError(RuntimeError):
    """Raised when the configured cross-encoder cannot be loaded."""


class PairScorer(Protocol):
    """Minimal scorer interface used by production and tests."""

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        """Score query/document pairs in input order."""


class CrossEncoderScorer:
    """Lazy wrapper around sentence-transformers CrossEncoder."""

    def __init__(
        self,
        model_name: str = config.RERANKER_MODEL,
        device: str = config.RERANKER_DEVICE,
        batch_size: int = config.RERANKER_BATCH_SIZE,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - dependency is optional in unit tests
            raise RerankerUnavailableError(
                "sentence-transformers is required for cross-encoder reranking."
            ) from exc

        selected_device = None if self.device.lower() == "auto" else self.device
        try:
            self._model = CrossEncoder(self.model_name, device=selected_device)
        except Exception as exc:  # pragma: no cover - depends on local model cache/runtime
            raise RerankerUnavailableError(
                f"Unable to load reranker model '{self.model_name}': {exc}"
            ) from exc
        return self._model

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        if not documents:
            return []
        model = self._load()
        pairs = [(query, document) for document in documents]
        try:
            scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:  # pragma: no cover - depends on local model runtime
            raise RerankerUnavailableError(f"Reranker scoring failed: {exc}") from exc
        return [float(score) for score in scores]


_DEFAULT_SCORER: CrossEncoderScorer | None = None


def get_default_scorer() -> CrossEncoderScorer:
    """Return one lazy scorer instance per process so weights are loaded once."""
    global _DEFAULT_SCORER
    if _DEFAULT_SCORER is None:
        _DEFAULT_SCORER = CrossEncoderScorer()
    return _DEFAULT_SCORER


def rerank_results(
    query: str,
    results: Sequence[RetrievalResult],
    *,
    top_k: int = config.TOP_K,
    scorer: PairScorer | None = None,
) -> tuple[list[RetrievalResult], str]:
    """Rerank vector candidates and return results plus the ranking status."""
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    candidates = list(results)
    if not config.RERANKER_ENABLED:
        return [replace(result, ranking_method="vector") for result in candidates[:top_k]], "disabled"
    if len(candidates) <= 1:
        return [replace(result, ranking_method="vector") for result in candidates[:top_k]], "skipped_single_candidate"

    active_scorer = scorer or get_default_scorer()
    try:
        scores = active_scorer.score(query, [result.content for result in candidates])
    except RerankerUnavailableError:
        if config.RERANKER_STRICT:
            raise
        return [replace(result, ranking_method="vector") for result in candidates[:top_k]], "unavailable_fallback"

    if len(scores) != len(candidates):
        raise RerankerUnavailableError(
            f"Reranker returned {len(scores)} scores for {len(candidates)} candidates."
        )

    reranked = [
        replace(result, rerank_score=score, ranking_method="cross_encoder")
        for result, score in zip(candidates, scores, strict=True)
    ]
    reranked.sort(
        key=lambda result: (
            result.rerank_score if result.rerank_score is not None else float("-inf"),
            result.similarity,
        ),
        reverse=True,
    )
    return reranked[:top_k], "cross_encoder"
