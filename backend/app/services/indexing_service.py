from uuid import UUID

from app.services.embedding_service import EmbeddingService
from app.storage.chroma import ChromaVectorStore
from app.storage.sqlite import SQLitePaperRepository

class IndexingService:
    """Create and store embeddings for processed paper chunks"""

    def __init__(self, paper_repository: SQLitePaperRepository, embedding_service: EmbeddingService, vector_store: ChromaVectorStore) -> None:
        self._paper_repository = paper_repository
        self._embedding_service = embedding_service
        self._vector_store = vector_store

    def index_paper(self, paper_id: UUID) -> int:
        """Embed a paper's chunks and store them in ChromaDB"""
        chunks = tuple(
            self._paper_repository.list_chunks_by_paper_id(paper_id)
        )

        if not chunks:
            return 0

        embedded_chunks = self._embedding_service.embed_chunks(chunks)

        self._vector_store.upsert(
            chunks=chunks, 
            embedded_chunks=embedded_chunks
        )

        return self._paper_repository.mark_chunks_embedded(paper_id=paper_id)