from pathlib import Path
from uuid import uuid4

from app.models.paper import Paper, PaperMetaData, ProcessingMetadata
from app.storage.file_storage import FileStorage

class PaperService():
    """Service responsible for Paper related operations"""

    def __init__(self, file_storage: FileStorage) -> None:
        self._file_storage = file_storage

    def upload_paper(self, content: bytes, original_filename: str) -> Paper:
        """Store an uploaded PDF and create its paper domain objects"""
        file_extension = Path(original_filename).suffix.lower()

        if file_extension != ".pdf":
            raise ValueError("Only PDF files are Supported.")

        paper_id = uuid4()
        stored_filename = f"{paper_id}{file_extension}"

        self._file_storage.save(content=content, stored_filename=stored_filename)

        return Paper(id=paper_id, metadata=PaperMetaData(), processing=ProcessingMetadata(stored_filename=stored_filename))