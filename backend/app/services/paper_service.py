from pathlib import Path
from uuid import uuid4, UUID

from app.models.paper import Paper, PaperMetaData, ProcessingMetadata
from app.storage.file_storage import FileStorage
from app.storage.sqlite import SQLitePaperRepository

class PaperService():
    """Service responsible for Paper related operations"""

    def __init__(self, file_storage: FileStorage, paper_repository: SQLitePaperRepository) -> None:
        self._file_storage = file_storage
        self._paper_repository = paper_repository

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