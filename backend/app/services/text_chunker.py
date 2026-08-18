from uuid import UUID

from app.models.chunk import Chunk
from app.services.pdf_parser import ExtractedPage

class TextChunker:
    """Split extracted page text into overlapping chunks"""

    def __init__(self, max_words: int = 200, overlap_words: int = 40) -> None:
        if max_words <= 0:
            raise ValueError("max_words must be greater than zero")

        if overlap_words < 0 or overlap_words >= max_words:
            raise ValueError("overlap_words must be zero or greater and less than max_words")

        self._max_words = max_words
        self._overlap_words = overlap_words

    def chunk_document(self, paper_id: UUID, pages: tuple[ExtractedPage, ...],) -> list[Chunk]:
        """Create page-aware chunks from extracted document pages"""
        chunks: list[Chunk] = []

        for page in pages:
            words = page.text.split()

            if not words:
                continue

            start = 0
            while start < len(words):
                end = min(start + self._max_words, len(words))
                chunk_words = words[start:end]

                chunks.append(Chunk(paper_id=paper_id, chunk_index=len(chunks),page_number=page.page_number, chunk_text=" ".join(chunk_words),token_count=len(chunk_words)))

                if end == len(words):
                    break

                start = end - self._overlap_words

        return chunks