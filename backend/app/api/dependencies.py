from app.core.constants import PAPERS_DIR
from app.services.paper_service import PaperService
from app.storage.file_storage import FileStorage


def get_paper_service() -> PaperService:
    """Provide a PaperService with local file storage"""
    file_storage = FileStorage(base_directory=PAPERS_DIR)
    return PaperService(file_storage=file_storage)