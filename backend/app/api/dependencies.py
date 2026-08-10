from app.core.constants import PAPERS_DIR, PAPERS_DATABASE_PATH
from app.services.paper_service import PaperService
from app.storage.file_storage import FileStorage
from app.storage.sqlite import SQLitePaperRepository


def get_paper_service() -> PaperService:
    """Provide a PaperService with local file storage"""
    file_storage = FileStorage(base_directory=PAPERS_DIR)
    paper_repository = SQLitePaperRepository(database_path= PAPERS_DATABASE_PATH)

    return PaperService(file_storage=file_storage, paper_repository=paper_repository)