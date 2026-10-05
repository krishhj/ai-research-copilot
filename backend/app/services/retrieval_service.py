from uuid import UUID

from app.models.query import Query
from app.services.embedding_service import EmbeddingService
from app.storage.chroma import ChromaVectorStore, VectorSearchResult


class RetrievalService:
    """Retrieve semantic-search results only from the user's papers."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        vector_store: ChromaVectorStore,
    ) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def search(
        self,
        query: Query,
        owner_id: UUID,
    ) -> tuple[VectorSearchResult, ...]:
        """Find chunks relevant to a question for one owner."""
        query_vector = self._embedding_service.embed_query(query.question)

        return self._vector_store.search(
            query_vector=query_vector,
            top_k=query.top_k,
            owner_id=owner_id,
            paper_id=query.paper_id,
        )