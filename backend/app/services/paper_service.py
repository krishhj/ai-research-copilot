from pathlib import Path
from uuid import uuid4, UUID

from app.core.exceptions import PDFProcessingError
from app.models.enums import PaperStatus
from app.services.document_processor import DocumentProcessor
from app.models.paper import Paper, PaperMetaData, ProcessingMetadata
from app.storage.file_storage import FileStorage
from app.storage.sqlite import SQLitePaperRepository

class PaperService():
    """Service responsible for Paper related operations"""

    def __init__(self, file_storage: FileStorage, paper_repository: SQLitePaperRepository, document_processor: DocumentProcessor) -> None:
        self._file_storage = file_storage
        self._paper_repository = paper_repository
        self._document_processor = document_processor

    def upload_paper(self, content: bytes, original_filename: str) -> Paper:
        """Store an uploaded PDF and create its paper domain objects"""
        file_extension = Path(original_filename).suffix.lower()

        if file_extension != ".pdf":
            raise ValueError("Only PDF files are Supported.")

        paper_id = uuid4()
        stored_filename = f"{paper_id}{file_extension}"

        self._file_storage.save(content=content, stored_filename=stored_filename)

        paper = Paper(                                      # ← step 1: build the Paper first
        id=paper_id,
        metadata=PaperMetaData(),
        processing=ProcessingMetadata(stored_filename=stored_filename),
        )
        
        self._paper_repository.add(paper)

        return paper

    def list_papers(self) -> list[Paper]:
        """Return all uploaded papers."""
        return self._paper_repository.list_all()

    def get_paper(self, paper_id: UUID) -> Paper | None:
        """Return one uploaded paper by ID"""
        return self._paper_repository.get_by_id(paper_id)

    def delete_paper(self, paper_id: UUID) -> bool:
        """Delete a paper and its stored PDF"""
        paper = self.get_paper(paper_id=paper_id)\
        
        if paper is None:
            return False

        self._file_storage.delete(paper.processing.stored_filename)
        self._paper_repository.delete(paper_id=paper_id)
        
        return True

    def process_paper(self, paper_id: UUID) -> Paper | None:
        """Extract and persist searchable chunks for one paper"""
        paper = self.get_paper(paper_id)

        if paper is None:
            return None

        self._paper_repository.update_status(paper_id, PaperStatus.PROCESSING)

        try:
            processed_document = self._document_processor.process(
                pdf_path= self._file_storage.get_path(paper.processing.stored_filename),
                paper_id=paper.id
            )

            self._paper_repository.save_processing_result(
                paper_id=paper.id, 
                total_pages=processed_document.total_pages,
                chunks=processed_document.chunks,
            )
        except PDFProcessingError:
            self._paper_repository.update_status(paper_id=paper_id,status= PaperStatus.FAILED)
            raise

        return self.get_paper(paper_id=paper_id)