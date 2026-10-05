import hashlib
from pathlib import Path
from uuid import UUID, uuid4

from app.core.exceptions import DuplicatePaperError, PDFProcessingError
from app.models.enums import PaperStatus
from app.models.paper import Paper, PaperMetaData, ProcessingMetadata
from app.services.document_processor import DocumentProcessor
from app.storage.chroma import ChromaVectorStore
from app.storage.file_storage import FileStorage
from app.storage.sqlite import SQLitePaperRepository


class PaperService:
    """Manage papers while enforcing ownership."""

    def __init__(
        self,
        file_storage: FileStorage,
        paper_repository: SQLitePaperRepository,
        document_processor: DocumentProcessor,
        vector_store: ChromaVectorStore,
    ) -> None:
        self._file_storage = file_storage
        self._paper_repository = paper_repository
        self._document_processor = document_processor
        self._vector_store = vector_store

    def upload_paper(
        self,
        content: bytes,
        original_filename: str,
        owner_id: UUID,
    ) -> Paper:
        """Store a PDF for its authenticated owner."""
        if Path(original_filename).suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")

        content_hash = hashlib.sha256(content).hexdigest()

        existing_paper = self._paper_repository.get_by_content_hash_and_owner(
            content_hash=content_hash,
            owner_id=owner_id,
        )

        if existing_paper is not None:
            raise DuplicatePaperError(
                "This paper is already in your research library."
            )

        paper_id = uuid4()
        stored_filename = f"{paper_id}.pdf"

        self._file_storage.save(
            content=content,
            stored_filename=stored_filename,
        )

        paper = Paper(
            id=paper_id,
            metadata=PaperMetaData(),
            processing=ProcessingMetadata(
                stored_filename=stored_filename,
            ),
        )

        self._paper_repository.add(
            paper=paper,
            owner_id=owner_id,
            content_hash=content_hash,
        )
        return paper

    def list_papers(self, owner_id: UUID) -> list[Paper]:
        """Return only the authenticated user's papers."""
        return self._paper_repository.list_by_owner(owner_id)

    def get_paper(
        self,
        paper_id: UUID,
        owner_id: UUID,
    ) -> Paper | None:
        """Return a paper only when it belongs to the owner."""
        return self._paper_repository.get_by_id_and_owner(
            paper_id,
            owner_id,
        )

    def delete_paper(
        self,
        paper_id: UUID,
        owner_id: UUID,
    ) -> bool:
        """Delete a paper only when it belongs to the owner."""
        paper = self.get_paper(paper_id, owner_id)

        if paper is None:
            return False

        self._vector_store.delete_by_paper_id(paper_id, owner_id)
        self._file_storage.delete(paper.processing.stored_filename)

        return self._paper_repository.delete_by_id_and_owner(
            paper_id,
            owner_id,
        )

    def process_paper(
        self,
        paper_id: UUID,
        owner_id: UUID,
    ) -> Paper | None:
        """Process a paper only when it belongs to the owner."""
        paper = self.get_paper(paper_id, owner_id)

        if paper is None:
            return None

        self._paper_repository.update_status(
            paper_id,
            owner_id,
            PaperStatus.PROCESSING,
        )

        try:
            processed_document = self._document_processor.process(
                pdf_path=self._file_storage.get_path(
                    paper.processing.stored_filename,
                ),
                paper_id=paper.id,
            )

            self._paper_repository.save_processing_result(
                paper_id=paper.id,
                owner_id=owner_id,
                total_pages=processed_document.total_pages,
                chunks=processed_document.chunks,
                metadata=processed_document.metadata,
            )
        except PDFProcessingError:
            self._paper_repository.update_status(
                paper_id,
                owner_id,
                PaperStatus.FAILED,
            )
            raise

        return self.get_paper(paper_id, owner_id)