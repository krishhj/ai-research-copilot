import pytest

from app.services.paper_service import PaperService
from app.storage.file_storage import FileStorage

def test_upload_paper_saves_pdf_and_returns_paper(tmp_path):
    storage = FileStorage(base_directory=tmp_path)
    service = PaperService(file_storage=storage)

    paper = service.upload_paper(
        content=b"sample PDF content",
        original_filename="attention.pdf",
    )

    saved_file = tmp_path / paper.processing.stored_filename

    assert saved_file.exists()
    assert saved_file.read_bytes() == b"sample PDF content"
    assert paper.processing.stored_filename == f"{paper.id}.pdf"

def test_upload_paper_rejects_non_pdf_file(tmp_path):
    storage = FileStorage(base_directory=tmp_path)
    service = PaperService(file_storage=storage)

    with pytest.raises(ValueError, match="Only PDF files are Supported."):
        service.upload_paper(
            content=b"not a PDF",
            original_filename="attention.docx"
            )