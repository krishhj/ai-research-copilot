import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

from app.core.exceptions import PDFProcessingError
from app.models.paper import PaperMetaData


@dataclass(frozen=True)
class PageLine:
    """One text line from a PDF page, with layout information."""

    text: str
    x0: float
    y0: float
    y1: float
    font_size: float


class MetadataExtractor:
    """Extract best-effort metadata from research-paper PDFs."""

    _AFFILIATION_PATTERN = re.compile(
        r"\b(?:"
        r"university|department|faculty|institute|institution|"
        r"laboratory|laboratories|lab|school|college|"
        r"research|facebook ai|google research|"
        r"microsoft research|openai|correspondence"
        r")\b",
        flags=re.IGNORECASE,
    )

    def extract(self, pdf_path: Path) -> PaperMetaData:
        """Extract embedded metadata and reliable first-page metadata."""
        try:
            with pymupdf.open(pdf_path) as document:
                metadata = document.metadata or {}

                if document.page_count:
                    first_page = document[0]
                    first_page_text = first_page.get_text("text", sort=True)
                    page_lines = self._extract_page_lines(first_page)
                else:
                    first_page_text = ""
                    page_lines = []

        except (OSError, ValueError, pymupdf.FileDataError) as error:
            raise PDFProcessingError(
                f"Could not extract metadata from: {pdf_path.name}"
            ) from error

        fallback_title, fallback_authors = (
            self._extract_title_and_authors(page_lines)
        )

        embedded_title = self._clean(metadata.get("title"))
        embedded_authors = self._parse_authors(metadata.get("author"))

        title = (
            embedded_title
            if self._is_valid_title(embedded_title)
            else fallback_title
        )
        authors = (
            embedded_authors
            if self._are_valid_authors(embedded_authors)
            else fallback_authors
        )

        return PaperMetaData(
            title=title,
            authors=authors,
            abstract=self._extract_abstract(first_page_text),
            year=self._extract_year(
                metadata.get("creationDate"),
                first_page_text,
            ),
            doi=self._extract_doi(first_page_text),
            keywords=self._extract_keywords(first_page_text),
        )

    def _extract_page_lines(self, page: pymupdf.Page) -> list[PageLine]:
        """Read page lines while preserving their position and font size."""
        page_data = page.get_text("dict", sort=True)
        page_lines: list[PageLine] = []

        for block in page_data["blocks"]:
            if block["type"] != 0:
                continue

            for line in block["lines"]:
                text = self._clean(
                    "".join(span["text"] for span in line["spans"])
                )
                if text is None:
                    continue

                font_size = max(
                    span["size"] for span in line["spans"]
                )

                x0, y0, _, y1 = line["bbox"]
                page_lines.append(
                    PageLine(
                        text=text,
                        x0=x0,
                        y0=y0,
                        y1=y1,
                        font_size=font_size,
                    )
                )

        return sorted(page_lines, key=lambda line: (line.y0, line.x0))

    def _extract_title_and_authors(
        self,
        page_lines: list[PageLine],
    ) -> tuple[str | None, list[str]]:
        """Extract title and authors from the header above the abstract."""
        abstract_index = next(
            (
                index
                for index, line in enumerate(page_lines)
                if line.text.lower() == "abstract"
            ),
            len(page_lines),
        )

        header_lines = [
            line
            for line in page_lines[:abstract_index]
            if not line.text.lower().startswith("arxiv:")
        ]

        if not header_lines:
            return None, []

        largest_font = max(line.font_size for line in header_lines)

        title_lines: list[PageLine] = []
        for line in header_lines:
            if line.font_size >= largest_font - 0.25:
                title_lines.append(line)
            elif title_lines:
                break

        if not title_lines:
            title_lines = [header_lines[0]]

        title_end_index = header_lines.index(title_lines[-1])

        # Some test PDFs and simpler PDFs use one font size everywhere.
        # In that case, find the first line that clearly resembles authors.
        if len(title_lines) == len(header_lines):
            author_index = next(
                (
                    index
                    for index, line in enumerate(header_lines[1:], start=1)
                    if self._looks_like_author_line(line.text)
                ),
                None,
            )

            if author_index is not None:
                title_lines = header_lines[:author_index]
                title_end_index = author_index - 1
            else:
                title_lines = [header_lines[0]]
                title_end_index = 0

        title = self._clean(
            " ".join(line.text for line in title_lines)
        )

        author_lines = header_lines[title_end_index + 1:]
        authors = self._extract_authors_from_lines(author_lines)

        return title, authors

    def _extract_authors_from_lines(
        self,
        lines: list[PageLine],
    ) -> list[str]:
        """Extract names while excluding affiliation and contact details."""
        authors: list[str] = []

        for line in lines:
            text = line.text

            if "@" in text or "www." in text.lower():
                continue

            affiliation_match = self._AFFILIATION_PATTERN.search(text)
            if affiliation_match:
                text = text[:affiliation_match.start()]

            authors.extend(self._parse_author_block(text))

        return self._deduplicate(authors)

    @staticmethod
    def _clean(value: str | None) -> str | None:
        """Normalize whitespace in a possible text value."""
        if not value:
            return None

        cleaned_value = " ".join(value.split())
        return cleaned_value or None

    def _parse_authors(self, value: str | None) -> list[str]:
        """Split embedded author metadata into individual names."""
        cleaned_value = self._clean(value)

        if cleaned_value is None:
            return []

        return self._parse_author_block(cleaned_value)

    def _parse_author_block(self, author_block: str) -> list[str]:
        """Parse a visible author line or block into person names."""
        without_markers = re.sub(
            r"[\*\†\‡\⋆\∗]+|\d+(?=[,\s]|$)",
            "",
            author_block,
        )

        candidates = re.split(
            r"\s*,\s*|\s*;\s*|\s+\band\b\s+",
            without_markers,
            flags=re.IGNORECASE,
        )

        return [
            candidate
            for item in candidates
            if (candidate := self._clean(item))
            and self._looks_like_person_name(candidate)
        ]

    @staticmethod
    def _looks_like_person_name(value: str) -> bool:
        """Reject emails, URLs, affiliations, and ordinary prose."""
        if "@" in value or "://" in value:
            return False

        words = re.findall(
            r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ.'-]*",
            value,
        )

        if not 2 <= len(words) <= 4:
            return False

        return all(
            word[0].isupper() or len(word) == 1
            for word in words
        )

    def _looks_like_author_line(self, line: str) -> bool:
        """Identify a likely visible author line."""
        return (
            "," in line
            or " and " in line.lower()
            or bool(re.search(r"[\*\†\‡\⋆\∗]", line))
            or self._looks_like_person_name(line)
        )

    @staticmethod
    def _deduplicate(values: list[str]) -> list[str]:
        """Keep author order while removing accidental duplicates."""
        seen: set[str] = set()
        unique_values: list[str] = []

        for value in values:
            key = value.casefold()
            if key not in seen:
                seen.add(key)
                unique_values.append(value)

        return unique_values

    @staticmethod
    def _is_valid_title(title: str | None) -> bool:
        """Reject unusable embedded PDF titles."""
        return bool(
            title
            and len(title) <= 250
            and "@" not in title
            and "abstract" not in title.lower()
        )

    def _are_valid_authors(self, authors: list[str]) -> bool:
        """Check whether embedded author metadata looks usable."""
        return bool(authors) and all(
            self._looks_like_person_name(author)
            for author in authors
        )

    @staticmethod
    def _extract_doi(text: str) -> str | None:
        """Extract a DOI when one appears in the text."""
        match = re.search(
            r"10\.\d{4,9}/[-._;()/:A-Z0-9]+",
            text,
            flags=re.IGNORECASE,
        )
        return match.group(0).rstrip(".,;)") if match else None

    @staticmethod
    def _extract_year(
        creation_date: str | None,
        first_page_text: str,
    ) -> int | None:
        """Prefer the arXiv submission year over years in citations."""
        arxiv_match = re.search(
            r"arXiv:\d{4}\.\d{4,5}v?\d*.*?"
            r"\b((?:19|20)\d{2})\b",
            first_page_text,
            flags=re.IGNORECASE,
        )
        if arxiv_match:
            return int(arxiv_match.group(1))

        creation_match = re.search(
            r"\b((?:19|20)\d{2})\b",
            creation_date or "",
        )
        if creation_match:
            return int(creation_match.group(1))

        page_match = re.search(
            r"\b((?:19|20)\d{2})\b",
            first_page_text,
        )
        return int(page_match.group(1)) if page_match else None

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

        return (
            " ".join(match.group("abstract").split())
            if match
            else None
        )

    @staticmethod
    def _extract_keywords(text: str) -> list[str]:
        """Extract comma-separated keywords when present."""
        match = re.search(
            r"\b(?:keywords?|index terms)\b\s*[:.-]?\s*(.+?)(?=\n|\Z)",
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            return []

        return [
            keyword.strip()
            for keyword in re.split(r"[,;]", match.group(1))
            if keyword.strip()
        ]