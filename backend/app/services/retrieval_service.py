from uuid import UUID

from app.models.query import Query
from app.services.embedding_service import EmbeddingService
from app.services.reranker_service import RerankerService
from app.storage.chroma import ChromaVectorStore, VectorSearchResult


class RetrievalService:
    """Retrieve semantic-search results only from the user's papers."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: ChromaVectorStore,
        reranker_service: RerankerService | None = None,
        rerank_candidates: int = 20,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store
        self._reranker_service = reranker_service
        self._rerank_candidates = rerank_candidates

    def search(
        self,
        query: Query,
        owner_id: UUID,
    ) -> tuple[VectorSearchResult, ...]:
        """Find chunks relevant to a question for one owner."""
        query_vector = self._embedding_service.embed_query(query.question)

        # Fetch more chunks than needed so the reranker has something to choose from
        top_k = query.top_k
        if self._reranker_service:
            top_k = max(self._rerank_candidates, query.top_k)

        results = self._vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            owner_id=owner_id,
            paper_id=query.paper_id,
        )

        if self._reranker_service:
            return self._reranker_service.rerank(query.question, results, query.top_k)

        return results
