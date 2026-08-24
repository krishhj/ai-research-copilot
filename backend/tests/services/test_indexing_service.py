from uuid import uuid4

from app.models.chunk import Chunk
from app.models.paper import Paper, PaperMetaData, ProcessingMetadata
from app.services.embedding_service import EmbeddingService
from app.services.indexing_service import IndexingService
from app.storage.sqlite import SQLitePaperRepository

class FakeEmbeddingModel:
    def encode_document(self, sentences, *, normalize_embeddings):
        return [[0.1, 0.2, 0.3] for _ in sentences]


class FakeVectorStore:
    def __init__(self):
        self.saved_chunks = ()

    def upsert(self, chunks, embedded_chunks):
        self.saved_chunks = chunks

def test_index_paper_stores_vectors_and_marks_chunks_embedded(tmp_path):
    repository = SQLitePaperRepository(tmp_path / "papers.db")

    paper = Paper(
        metadata=PaperMetaData(title="Attention Is All You Need"),
        processing=ProcessingMetadata(stored_filename="attention.pdf")
        )
    repository.add(paper)

    chunks = (
        Chunk(
            paper_id=paper.id,
            chunk_index=0,
            page_number=1,
            chunk_text="Transformers use self-attention.",
            token_count=4
        ),
    )
    repository.save_processing_result(
        paper_id=paper.id,
        total_pages=1,
        chunks=chunks
    )

    vector_store = FakeVectorStore()
    service = IndexingService(
        paper_repository=repository,
        embedding_service=EmbeddingService(FakeEmbeddingModel()),
        vector_store=vector_store
    )

    indexed_chunks = service.index_paper(paper.id)

    saved_chunks = repository.list_chunks_by_paper_id(paper.id)

    assert indexed_chunks == 1
    assert vector_store.saved_chunks == chunks
    assert saved_chunks[0].embedding_created is True
    