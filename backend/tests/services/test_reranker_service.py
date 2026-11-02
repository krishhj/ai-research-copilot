from uuid import uuid4

import pytest

from app.services.reranker_service import RerankerService
from app.storage.chroma import VectorSearchResult


class FakeCrossEncoder:
    """Fake reranker for testing"""

    def __init__(self, scores_by_text: dict[str, float]) -> None:
        self.scores_by_text = scores_by_text
        self.pairs: list[tuple[str, str]] = []

    def predict(self, inputs):
        self.pairs = inputs
        return [self.scores_by_text[passage] for _, passage in inputs]


def make_result(text: str, score: float) -> VectorSearchResult:
    return VectorSearchResult(
        chunk_id=uuid4(),
        paper_id=uuid4(),
        page_number=1,
        text=text,
        score=score,
    )


def test_rerank_orders_results_by_cross_encoder_score():
    model = FakeCrossEncoder({"weak": 0.1, "strong": 0.9, "medium": 0.5})
    service = RerankerService(model=model)

    results = (
        make_result("weak", score=0.8),
        make_result("strong", score=0.4),
        make_result("medium", score=0.6),
    )

    reranked = service.rerank("What is RAG?", results, top_k=3)

    assert [result.text for result in reranked] == ["strong", "medium", "weak"]
    assert model.pairs == [
        ("What is RAG?", "weak"),
        ("What is RAG?", "strong"),
        ("What is RAG?", "medium"),
    ]


def test_rerank_keeps_only_top_k_and_uses_reranker_scores():
    service = RerankerService(FakeCrossEncoder({"a": 0.2, "b": 0.7, "c": 0.4}))

    results = (make_result("a", 0.9), make_result("b", 0.1), make_result("c", 0.5))

    reranked = service.rerank("question", results, top_k=2)

    assert [result.text for result in reranked] == ["b", "c"]
    assert [result.score for result in reranked] == [0.7, 0.4]


def test_rerank_empty_results_returns_empty_tuple():
    service = RerankerService(FakeCrossEncoder({}))

    assert service.rerank("question", (), top_k=5) == ()


def test_rerank_rejects_wrong_number_of_scores():
    class BrokenCrossEncoder:
        def predict(self, inputs):
            return [0.5]

    service = RerankerService(BrokenCrossEncoder())

    with pytest.raises(RuntimeError):
        service.rerank("question", (make_result("a", 0.5), make_result("b", 0.5)), top_k=2)
