from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from app.storage.chroma import VectorSearchResult


class CrossEncoderModel(Protocol):
    """Model interface used to score (question, passage) pairs"""

    def predict(self, inputs: list[tuple[str, str]]) -> Sequence[float]:
        """Return one relevance score for every pair"""


class RerankerService:
    """Re-order search results using a cross-encoder model"""

    def __init__(self, model: CrossEncoderModel) -> None:
        self._model = model

    def rerank(
        self,
        question: str,
        results: Sequence[VectorSearchResult],
        top_k: int,
    ) -> tuple[VectorSearchResult, ...]:
        """Return the top_k results with the highest reranker scores"""
        if not results:
            return ()

        scores = self._model.predict([(question, result.text) for result in results])

        if len(scores) != len(results):
            raise RuntimeError("The reranker model returned an unexpected result.")

        ranked = sorted(zip(results, scores), key=lambda pair: pair[1], reverse=True)

        return tuple(replace(result, score=float(score)) for result, score in ranked[:top_k])
