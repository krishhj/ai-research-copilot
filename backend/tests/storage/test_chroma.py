from uuid import uuid4

from app.models.chunk import Chunk
from app.services.embedding_service import EmbeddedChunk
from app.storage.chroma import ChromaVectorStore

def test_upsert_and_delete_vectors_for_paper(tmp_path):
    vector_store = ChromaVectorStore(persist_directory=tmp_path/ "chroma")
    paper_id = uuid4()

    chunks = (
        Chunk(
            paper_id=paper_id,
            chunk_index=0,
            page_number=1,
            chunk_text="Transformers use self-attention.",
            token_count=4
        ),
        Chunk(
            paper_id=paper_id,
            chunk_index=1,
            page_number=2,
            chunk_text="Embeddings represent semantic meaning.",
            token_count=4
        ),
    )

    embedded_chunks = (
        EmbeddedChunk(
            chunk_id= chunks[0].id,
            vector=(0.1, 0.2, 0.3)
        ),
        EmbeddedChunk(
            chunk_id= chunks[1].id,
            vector=(0.4, 0.5, 0.6)
        ),
    )

    vector_store.upsert(chunks=chunks, embedded_chunks=embedded_chunks)

    assert vector_store.count_by_paper_id(paper_id=paper_id) == 2

    vector_store.delete_by_paper_id(paper_id)

    assert vector_store.count_by_paper_id(paper_id) == 0
