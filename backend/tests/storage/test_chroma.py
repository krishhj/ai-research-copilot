from uuid import UUID, uuid4

from app.models.chunk import Chunk
from app.services.embedding_service import EmbeddedChunk
from app.storage.chroma import ChromaVectorStore

def test_upsert_and_delete_vectors_for_paper(tmp_path):
    vector_store = ChromaVectorStore(persist_directory=tmp_path/ "chroma")
    paper_id = uuid4()
    owner_id = UUID("00000000-0000-0000-0000-000000000001")

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

    vector_store.upsert(chunks=chunks, embedded_chunks=embedded_chunks, owner_id=owner_id)

    assert vector_store.count_by_paper_id(paper_id=paper_id) == 2

    vector_store.delete_by_paper_id(paper_id, owner_id)

    assert vector_store.count_by_paper_id(paper_id) == 0

def test_search_returns_only_vectors_owned_by_current_user(tmp_path):
    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
    )

    owner_a = uuid4()
    owner_b = uuid4()

    owner_a_chunk = Chunk(
        paper_id=uuid4(),
        chunk_index=0,
        page_number=1,
        chunk_text="Owner A private research content.",
        token_count=5,
    )

    owner_b_chunk = Chunk(
        paper_id=uuid4(),
        chunk_index=0,
        page_number=1,
        chunk_text="Owner B private research content.",
        token_count=5,
    )

    vector_store.upsert(
        chunks=(owner_a_chunk,),
        embedded_chunks=(
            EmbeddedChunk(
                chunk_id=owner_a_chunk.id,
                vector=(1.0, 0.0, 0.0),
            ),
        ),
        owner_id=owner_a,
    )

    vector_store.upsert(
        chunks=(owner_b_chunk,),
        embedded_chunks=(
            EmbeddedChunk(
                chunk_id=owner_b_chunk.id,
                vector=(1.0, 0.0, 0.0),
            ),
        ),
        owner_id=owner_b,
    )

    results = vector_store.search(
        query_vector=(1.0, 0.0, 0.0),
        top_k=5,
        owner_id=owner_a,
    )

    assert len(results) == 1
    assert results[0].chunk_id == owner_a_chunk.id
    assert results[0].paper_id == owner_a_chunk.paper_id