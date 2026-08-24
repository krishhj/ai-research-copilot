from app.models.query import Query
from app.services.embedding_service import EmbeddingService
from app.storage.chroma import ChromaVectorStore, VectorSearchResult

class RetrievalService:
    """Retrieve relevant paper chunks using semantic search"""

    def __init__(self, embedding_service: EmbeddingService, vector_store: ChromaVectorStore) -> None:
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def search(self, query: Query) -> tuple[VectorSearchResult, ...]:
        """Find chunks relevant to user's question"""
        query_vector = self._embedding_service.embed_query(query.question)

        return self._vector_store.search(
            query_vector=query_vector,
            top_k=query.top_k,
            paper_id=query.paper_id
        )