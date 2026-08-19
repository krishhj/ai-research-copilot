from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from app.models.chunk import Chunk
from app.services.pdf_parser import ExtractedPage, PDFParser
from app.services.text_cleaner import TextCleaner
from app.services.text_chunker import TextChunker

@dataclass(frozen=True)
class ProcessedDocument:
    """Results of processing one research-paper PDF"""

    total_pages: int
    chunks: tuple[Chunk, ...]

class DocumentProcessor:
    """Extract, clean and chunk a PDF document"""

    def __init__(self, pdf_parser: PDFParser, text_cleaner: TextCleaner, text_chunker: TextChunker) -> None:
        self._pdf_parser = pdf_parser
        self._text_cleaner = text_cleaner
        self._text_chunker = text_chunker

    def process(self, pdf_path: Path, paper_id: UUID) -> ProcessedDocument:
        """Process a PDF into clean, page-aware text chunks"""
        parsed_document = self._pdf_parser.extract(pdf_path)

        cleaned_pages = tuple(
            ExtractedPage(
                page_number=page.page_number,
                text= self._text_cleaner.clean(page.text),
            )
            for page in parsed_document.pages
        )

        chunks = self._text_chunker.chunk_document(paper_id=paper_id, pages=cleaned_pages)

        return ProcessedDocument(total_pages=parsed_document.page_count, chunks= tuple(chunks))