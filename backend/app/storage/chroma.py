from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

import chromadb

from app.models.chunk import Chunk
from app.services.embedding_service import EmbeddedChunk

@dataclass(frozen=True)
class VectorSearchResult:
    """One chunk returned from vector similarity search"""
    chunk_id: UUID        
    paper_id: UUID
    page_number : int
    text: str
    score: float

class ChromaVectorStore:
    """Persist and manage chunk embeddings in ChromaDB"""

    def __init__(self, persist_directory: Path, collection_name: str = "paper_chunks") -> None:
        self._client = chromadb.PersistentClient(path=str(persist_directory))
        self._collection = self._client.get_or_create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def upsert(self, chunks: tuple[Chunk, ...], embedded_chunks: tuple[EmbeddedChunk, ...]) -> None:
        """Store and update chunks and their precomputed embeddings"""
        if len(chunks) != len(embedded_chunks):
            raise ValueError("Each chunk must have exactly one embedding")

        embeddings_by_chunk_id = {
            embedded_chunk.chunk_id: embedded_chunk.vector
            for embedded_chunk in embedded_chunks
        }

        if {chunk.id for chunk in chunks} != set(embeddings_by_chunk_id):
            raise ValueError("Chunk IDs and Embedding IDs must match")

        self._collection.upsert(
            ids = [str(chunk.id) for chunk in chunks],
            embeddings=[
                list(embeddings_by_chunk_id[chunk.id])
                for chunk in chunks
            ],
            documents=[chunk.chunk_text for chunk in chunks],
            metadatas=[
                {
                    "paper_id": str(chunk.paper_id),
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index
                }
                for chunk in chunks
            ]
        )

    def count_by_paper_id(self, paper_id: UUID) -> int :
        """Return the number of vectors stored for one paper"""
        result = self._collection.get(where={"paper_id": str(paper_id)})
        return len(result["ids"])

    def delete_by_paper_id(self, paper_id: UUID) -> None:
        """Delete all vectors belonging to one paper"""
        self._collection.delete(where={"paper_id": str(paper_id)})

    def search(self, query_vector: tuple[float, ...], top_k: int, paper_id: UUID | None = None)-> tuple[VectorSearchResult, ...]:
        """Return the chunks most similar to a query vector"""
        where = {"paper_id": str(paper_id)} if paper_id else None

        result = self._collection.query(
            query_embeddings=[list(query_vector)],
            n_results= top_k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        chunk_ids = result["ids"][0]
        documents = result["documents"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        return tuple(
            VectorSearchResult(
                chunk_id= UUID(chunk_id),
                paper_id=UUID(metadata["paper_id"]),
                page_number= int(metadata["page_number"]),
                text = document,
                score= max(0.0, min(1.0, 1.0 - float(distance)))
            )
            for chunk_id, document, metadata, distance in zip(
                chunk_ids,
                documents,
                metadatas,
                distances,
                strict=True
            )
        )
