from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import chromadb

from app.models.chunk import Chunk
from app.services.embedding_service import EmbeddedChunk


@dataclass(frozen=True)
class VectorSearchResult:
    """One chunk returned from vector similarity search."""

    chunk_id: UUID
    paper_id: UUID
    page_number: int
    text: str
    score: float


class ChromaVectorStore:
    """Persist vector embeddings with user ownership metadata."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str = "paper_chunks",
    ) -> None:
        self._client = chromadb.PersistentClient(
            path=str(persist_directory),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        chunks: tuple[Chunk, ...],
        embedded_chunks: tuple[EmbeddedChunk, ...],
        owner_id: UUID,
    ) -> None:
        """Store vectors and associate every chunk with its owner."""
        if len(chunks) != len(embedded_chunks):
            raise ValueError(
                "Each chunk must have exactly one embedding.",
            )

        embeddings_by_chunk_id = {
            embedded_chunk.chunk_id: embedded_chunk.vector
            for embedded_chunk in embedded_chunks
        }

        if {chunk.id for chunk in chunks} != set(embeddings_by_chunk_id):
            raise ValueError("Chunk IDs and embedding IDs must match.")

        self._collection.upsert(
            ids=[str(chunk.id) for chunk in chunks],
            embeddings=[
                list(embeddings_by_chunk_id[chunk.id])
                for chunk in chunks
            ],
            documents=[chunk.chunk_text for chunk in chunks],
            metadatas=[
                {
                    "user_id": str(owner_id),
                    "paper_id": str(chunk.paper_id),
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                }
                for chunk in chunks
            ],
        )

    def delete_by_paper_id(
        self,
        paper_id: UUID,
        owner_id: UUID,
    ) -> None:
        """Delete only vectors that match both paper and owner."""
        self._collection.delete(
            where={
                "$and": [
                    {"paper_id": str(paper_id)},
                    {"user_id": str(owner_id)},
                ]
            }
        )

    def search(
        self,
        query_vector: tuple[float, ...],
        top_k: int,
        owner_id: UUID,
        paper_id: UUID | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Search only chunks owned by the authenticated user."""
        filters: list[dict[str, str]] = [
            {"user_id": str(owner_id)},
        ]

        if paper_id:
            filters.append({"paper_id": str(paper_id)})

        where = (
            filters[0]
            if len(filters) == 1
            else {"$and": filters}
        )

        result = self._collection.query(
            query_embeddings=[list(query_vector)],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        return tuple(
            VectorSearchResult(
                chunk_id=UUID(chunk_id),
                paper_id=UUID(metadata["paper_id"]),
                page_number=int(metadata["page_number"]),
                text=document,
                score=max(0.0, min(1.0, 1.0 - float(distance))),
            )
            for chunk_id, document, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        )
    def count_by_paper_id(self, paper_id: UUID) -> int:
        """Return the number of stored vectors for a given paper (any owner)."""
        result = self._collection.get(
            where={"paper_id": str(paper_id)},
            include=[],
        )
        return len(result["ids"])
