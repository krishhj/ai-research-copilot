from uuid import uuid4

import pytest

from app.services.pdf_parser import ExtractedPage
from app.services.text_chunker import TextChunker

def test_chunk_document_creates_overlapping_chunks():
    pages = (
        ExtractedPage(
            page_number=1,
            text="one two three four five six seven eight nine ten"
        ),
    )
    chunker = TextChunker(max_words=4, overlap_words=1)

    chunks = chunker.chunk_document(paper_id=uuid4(), pages=pages)

    assert len(chunks) == 3
    assert chunks[0].chunk_text == "one two three four"
    assert chunks[1].chunk_text == "four five six seven"
    assert chunks[2].chunk_text == "seven eight nine ten"
    assert chunks[0].page_number == 1

def test_chunk_document_preserves_page_numbers():
    pages = (ExtractedPage(page_number=1, text="first page"),
             ExtractedPage(page_number=2, text="second page"))

    chunker = TextChunker(max_words=10, overlap_words=0)

    chunks = chunker.chunk_document(paper_id=uuid4(), pages=pages)

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1

def test_chunker_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        TextChunker(max_words=100, overlap_words=100)
