import pytest

from app.services.document_processor import DocumentProcessor
from app.services.metadata_extractor import MetadataExtractor
from app.services.paper_service import PaperService
from app.services.pdf_parser import PDFParser
from app.services.text_cleaner import TextCleaner
from app.services.text_chunker import TextChunker
from app.storage.chroma import ChromaVectorStore
from app.storage.file_storage import FileStorage
from app.storage.sqlite import SQLitePaperRepository

def test_upload_paper_saves_pdf_and_returns_paper(tmp_path):
    storage = FileStorage(base_directory=tmp_path)
    repository = SQLitePaperRepository(
        database_path=tmp_path / "papers.db",
    )
    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
    )
    service = PaperService(
        file_storage=storage,
        paper_repository=repository,
        document_processor=DocumentProcessor(
            pdf_parser=PDFParser(),
            text_cleaner=TextCleaner(),
            text_chunker=TextChunker(),
            metadata_extractor=MetadataExtractor(),
        ),
        vector_store=vector_store,
    )

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
    repository = SQLitePaperRepository(
        database_path=tmp_path / "papers.db",
    )
    vector_store = ChromaVectorStore(
        persist_directory=tmp_path / "chroma",
    )
    service = PaperService(
        file_storage=storage,
        paper_repository=repository,
        document_processor=DocumentProcessor(
            pdf_parser=PDFParser(),
            text_cleaner=TextCleaner(),
            text_chunker=TextChunker(),
            metadata_extractor=MetadataExtractor(),
        ),
        vector_store=vector_store,
    )

    with pytest.raises(ValueError, match="Only PDF files are Supported."):
        service.upload_paper(
            content=b"not a PDF",
            original_filename="attention.docx"
            )