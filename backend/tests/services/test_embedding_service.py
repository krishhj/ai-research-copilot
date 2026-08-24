from uuid import uuid4

from app.models.chunk import Chunk
from app.services.embedding_service import EmbeddingService

class FakeEmbeddingModel:
    """Fake model for testing"""

    def __init__(self) -> None:
        self.sentences: list[str] = []
        self.normalized = False

    def encode_document(self, sentences: list[str], *, normalize_embeddings: bool) -> list[list[float]]:
        self.sentences = sentences
        self.normalized = normalize_embeddings

        return [[0.1,0.2,0.3] for _ in sentences]

    def encode_query(self, sentences, *, normalize_embeddings):
        return [[0.9, 0.8, 0.7] for _ in sentences]

def test_embed_chunks_returns_vector_for_each_chunk():
    model = FakeEmbeddingModel()
    service = EmbeddingService(model=model)

    chunks = [
        Chunk(
            paper_id=uuid4(),
            chunk_index=0,
            page_number=1,
            chunk_text="Transformers use self-attention.",
            token_count=4
        ),
        Chunk(
            paper_id=uuid4(),
            chunk_index=1,
            page_number=2,
            chunk_text="Embeddings represent semantic meaning.",
            token_count=4
        ),
    ]

    embedded_chunks = service.embed_chunks(chunks=chunks)

    assert len(embedded_chunks) == 2
    assert embedded_chunks[0].chunk_id == chunks[0].id
    assert embedded_chunks[0].vector == (0.1,0.2,0.3)
    assert model.sentences == [
        "Transformers use self-attention.",
        "Embeddings represent semantic meaning."
    ]
    assert model.normalized is True

def test_embed_empty_chunks_returns_empty_tuple():
    service = EmbeddingService(model=FakeEmbeddingModel())

    assert service.embed_chunks([]) == ()

def test_embed_query_returns_one_vector():
    service = EmbeddingService(FakeEmbeddingModel())

    vector = service.embed_query("What is self-attention?")

    assert vector == (0.9, 0.8, 0.7)