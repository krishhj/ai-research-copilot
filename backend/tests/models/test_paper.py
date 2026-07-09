"""
Test 1 - Paper Creation
"""
from app.models.paper import (
    Paper,
    PaperMetaData,
    ProcessingMetadata,
)
from app.models.enums import PaperStatus

def test_create_paper():
    metadata = PaperMetaData(
        title = "Attention Is All You Need",
        authors=["Ashish Vaswani"],
        year=2017,
    )
    processing = ProcessingMetadata(
        filename = "attention.pdf"
    )
    paper = Paper(
        metadata=metadata,
        processing=processing
    )

    assert paper.metadata.title == "Attention Is All You Need"
    assert paper.processing.filename == "attention.pdf"
    assert paper.processing.status == PaperStatus.UPLOADED

"""
Test 2 - UUID Generation
"""
from uuid import UUID

def test_paper_has_uuid():
    metadata = PaperMetaData(title="Paper")
    processing = ProcessingMetadata(filename="paper.pdf")
    paper = Paper(
        metadata=metadata,
        processing=processing,
    )

    assert isinstance(paper.id, UUID)

"""
Test 3 - Timestamp
"""
from datetime import datetime
def test_upload_time_created():
    processing = ProcessingMetadata(filename="paper.pdf")

    assert isinstance(processing.uploaded_at, datetime)

"""
Test 4 - Invalid Year
"""
import pytest
from pydantic import ValidationError

def test_inavlid_year():
    with pytest.raises(ValidationError):
        PaperMetaData(
            title="Paper",
            year="hello",
        )

"""
Test 5 - Default status
"""
def test_default_status():
    processing = ProcessingMetadata(filename="paper.pdf")
    assert processing.status == PaperStatus.UPLOADED