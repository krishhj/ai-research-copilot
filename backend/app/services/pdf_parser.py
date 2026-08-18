from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.core.exceptions import PDFProcessingError


@dataclass(frozen=True)
class ExtractedPage:
    """Text extracted from one PDF page"""

    page_number: int
    text: str

@dataclass(frozen=True)
class ParsedDocument:
    """Text extracted from an entire page"""
    pages: tuple[ExtractedPage, ...]

    @property
    def page_count(self) -> int:
        """Return the number of pages in PDF"""
        return len(self.pages)

class PDFParser:
    """Extract text from PDF files"""

    def extract(self, pdf_path: Path) -> ParsedDocument:
        """Extract page-level text from a PDF"""
        try:
            with pymupdf.open(pdf_path) as document:
                pages = tuple(
                    ExtractedPage(
                        page_number=page_number,
                        text= page.get_text("text", sort=True).strip()
                    )
                    for page_number, page in enumerate(document, start=1)
                )
        except(OSError, ValueError, pymupdf.FileDataError) as error:
            raise PDFProcessingError(
                f"Could not extract teaxt from PDF: {pdf_path.name}"
            ) from error

        return ParsedDocument(pages=pages)
