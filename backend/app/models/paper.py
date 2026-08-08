from datetime import UTC, datetime
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from app.models.enums import PaperStatus


class PaperMetaData(BaseModel):
    """Metadata describing a research paper"""

    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    year: int | None = None
    doi: str | None = None
    journal: str | None = None
    keywords: list[str] = Field(default_factory=list)


class ProcessingMetadata(BaseModel):
    """Metadata describing the paper inside our application"""

    stored_filename: str
    total_pages: int = 0
    total_chunks: int = 0
    uploaded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    status: PaperStatus = PaperStatus.UPLOADED


class Paper(BaseModel):
    """Represent a research paper in the system"""

    id: UUID = Field(default_factory=uuid4)
    metadata: PaperMetaData
    processing: ProcessingMetadata
