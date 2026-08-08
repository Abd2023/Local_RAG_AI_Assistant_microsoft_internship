import pytest

from src import config
from src.reranker import rerank_results
from src.retrieval import RetrievalResult


def make_result(source_name: str, similarity: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=1,
        source_name=source_name,
        chunk_index=0,
        content=f"Content from {source_name}",
        similarity=similarity,
    )


class FakeScorer:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores

    def score(self, query: str, documents: list[str]) -> list[float]:
        assert query == "setup"
        assert len(documents) == len(self.scores)
        return self.scores


def test_reranker_can_move_a_lower_vector_result_to_the_top(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "RERANKER_ENABLED", True)
    results, status = rerank_results(
        "setup",
        [make_result("grading.md", 0.95), make_result("tools_and_setup.md", 0.80)],
        top_k=1,
        scorer=FakeScorer([0.1, 0.9]),
    )

    assert status == "cross_encoder"
    assert results[0].source_name == "tools_and_setup.md"
    assert results[0].rerank_score == pytest.approx(0.9)
    assert results[0].similarity == pytest.approx(0.80)


def test_reranker_is_injectable_and_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "RERANKER_ENABLED", False)
    results, status = rerank_results(
        "setup",
        [make_result("first.md", 0.9), make_result("second.md", 0.8)],
        top_k=1,
        scorer=FakeScorer([0.0, 1.0]),
    )

    assert status == "disabled"
    assert results[0].source_name == "first.md"
