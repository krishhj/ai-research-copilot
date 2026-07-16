from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.models.chunk import Chunk


# Test 1 — Create a Chunk
def test_create_chunk():
    chunk = Chunk(
        paper_id=uuid4(),
        chunk_index=0,
        page_number=1,
        chunk_text="This is a test chunk.",
        token_count=120,
    )

    assert chunk.chunk_index == 0
    assert chunk.page_number == 1
    assert chunk.chunk_text == "This is a test chunk."


# Test 2 — UUID Generation
def test_chunk_has_uuid():
    chunk = Chunk(
        paper_id=uuid4(),
        chunk_index=0,
        page_number=1,
        chunk_text="Hello",
        token_count=10,
    )

    assert isinstance(chunk.id, UUID)


# Test 3 — Default Embedding Status
def test_default_embedding_status():
    chunk = Chunk(
        paper_id=uuid4(),
        chunk_index=0,
        page_number=1,
        chunk_text="Hello",
        token_count=10,
    )

    assert chunk.embedding_created is False


# Test 4 — Invalid Page Number
def test_invalid_page_number():
    with pytest.raises(ValidationError):
        Chunk(
            paper_id=uuid4(),
            chunk_index=0,
            page_number=0,
            chunk_text="Hello",
            token_count=10,
        )


# Test 5 — Invalid Chunk Index
def test_invalid_chunk_index():
    with pytest.raises(ValidationError):
        Chunk(
            paper_id=uuid4(),
            chunk_index=-1,
            page_number=1,
            chunk_text="Hello",
            token_count=10,
        )
