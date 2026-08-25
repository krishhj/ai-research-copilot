import re
from pathlib import Path

import pymupdf

from app.core.exceptions import PDFProcessingError
from app.models.paper import PaperMetaData

class MetadataExtractor:
    """Extract available metadata from a research-paper PDF"""

    def extract(self, pdf_path: Path) -> PaperMetaData:
        """Extract metadata from embedded fields and first-page text"""
        try:
            with pymupdf.open(pdf_path) as document:
                metadata = document.metadata or {}
                first_page_text = (
                    document[0].get_text("text", sort=True)
                    if document.page_count
                    else ""
                )
        except (OSError, ValueError, pymupdf.FileDataError) as error:
            raise PDFProcessingError(
                f"Could not extract metadata from: {pdf_path.name}"
            ) from error

        title = self._clean(metadata.get("title"))
        authors = self._parse_authors(metadata.get("author"))

        lines = self._meaningful_lines(first_page_text)

        if title is None and lines:
            title = lines[0]

        if not authors and len(lines) > 1:
            authors = self._parse_authors(lines[1])

        return PaperMetaData(
            title = title, 
            authors = authors, 
            abstract = self._extract_abstract(first_page_text),
            year = self._extract_year(metadata.get("creationDate"), first_page_text),
            doi = self._extract_doi(first_page_text),
            keywords = self._extract_keywords(first_page_text)
        )

    @staticmethod
    def _clean(value: str | None) -> str | None:
        """Normalize a possible text value."""
        if not value:
            return None

        cleaned_value = " ".join(value.split())
        return cleaned_value or None

    def _meaningful_lines(self, text: str) -> list[str]:
        """Return non_empty normalized lines from first-page text"""
        return [
            cleaned_line
            for line in text.splitlines()
            if (cleaned_line := self._clean(line))
        ]

    def _parse_authors(self, value: str | None) -> list[str]:
        """Split common author-list formats"""
        cleaned_value = self._clean(value=value)

        if cleaned_value is None:
            return []

        return [
            author.strip()
            for author in re.split(
                r"\s*;\s*|\s+\band\b\s+",
                cleaned_value,
                flags=re.IGNORECASE
            )
            if author.strip()
        ]

    @staticmethod
    def _extract_doi(text: str) -> str | None:
        """Extract a DOI when one appears in the text"""
        match = re.search(
            r"10\.\d{4,9}/[-._;()/:A-Z0-9]+",
            text, 
            flags=re.IGNORECASE
        )

        return match.group(0).rstrip(".,;)") if match else None

    @staticmethod
    def _extract_year(creation_date: str | None, first_page_text: str) -> int | None:
        """Extract a likely publication year"""
        source_text = f"{creation_date or ''} {first_page_text}"

        match = re.search(r"\b(?:19|20)\d{2}\b", source_text)

        return int(match.group(0)) if match else None

    @staticmethod
    def _extract_abstract(text: str) -> str | None:
        """Extract text between Abstract and the next major section."""
        match = re.search(
            r"\babstract\b\s*[:.-]?\s*"
            r"(?P<abstract>.*?)"
            r"(?=\n(?:\d+\.?\s+)?"
            r"(?:introduction|keywords?|index terms)\b|\Z)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )

        if match is None:
            return None

        return " ".join(match.group("abstract").split())

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract comma-separated keywords when present"""
        match = re.search(
            r"\b(?:keywords?|index terms)\b\s*[:.-]?\s*(.+?)(?=\n|\Z)",
            text,
            flags=re.IGNORECASE
        )

        if not match:
            return []

        return[
            keyword.strip()
            for keyword in re.split(r"[,;]", match.group(1))
            if keyword.strip()
        ]