from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol
from uuid import UUID

from app.models.chunk import Chunk


class DocumentEmbeddingModel(Protocol):
    """Model interface used to embed document text"""

    def encode_document(self, sentences: list[str], *, normalize_embeddings: bool) -> Sequence[Sequence[float]]:
        """Convert document text into embedding vectors"""

@dataclass(frozen=True)
class EmbeddedChunk:
    """A chunk paired with its vector embedding"""

    chunk_id: UUID
    vector: tuple[float, ...]

class EmbeddingService:
    """Create normalized vector embeddings for document chunks"""

    def __init__(self, model: DocumentEmbeddingModel) -> None:
        self._model = model

    def embed_chunks(self, chunks: Sequence[Chunk]) -> tuple[EmbeddedChunk, ...]:
        """Create one embedding vector for every chunk"""
        if not chunks:
            return ()

        embeddings = self._model.encode_document([chunk.chunk_text for chunk in chunks], normalize_embeddings= True)

        if len(embeddings) != len(chunks):
            raise RuntimeError("The embedding model returned an unexpected result.")

        return tuple(
            EmbeddedChunk(
                chunk_id=chunk.id,
                vector= tuple(float(value) for value in embedding)
            )
            for chunk, embedding in zip(chunks, embeddings, strict= True)
        )

    def embed_query(self, question: str) -> tuple[float, ...]:
        """Create a normalized embedding for one user question"""
        if not question.strip():
            raise ValueError("Question cannot be empty")

        embedding = self._model.encode_query([question], normalize_embeddings = True)

        return tuple(float(value) for value in embedding[0])