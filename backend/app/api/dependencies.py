from app.services.paper_service import PaperService


def get_paper_service() -> PaperService:
    """Provide a PaperService instance"""
    return PaperService()