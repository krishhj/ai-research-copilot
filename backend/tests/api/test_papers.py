import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_paper_service
from app.main import create_app
from app.services.paper_service import PaperService
from app.storage.file_storage import FileStorage
from app.storage.sqlite import SQLitePaperRepository

@pytest.fixture
def client(tmp_path):
    """Create a test client that stores uploaded files temporarily"""
    app = create_app()

    def get_test_paper_service() -> PaperService:
        storage  = FileStorage(base_directory=tmp_path)
        repository = SQLitePaperRepository(
            database_path=tmp_path / "papers.db",
        )
        return PaperService(
            file_storage=storage,
            paper_repository=repository,
        )

    app.dependency_overrides[get_paper_service] = get_test_paper_service

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()

def test_upload_pdf_returns_created_response(client):
    response = client.post(
        "api/v1/papers",
        files={
            "file":(
                "attention.pdf",
                b"sample PDF content",
                "applications/pdf",
            ),
        },
    )

    assert response.status_code == 201

    data = response.json()
    assert data["message"] == "Paper uploaded successfully"
    assert data["paper"]["processing"]["stored_filename"].endswith(".pdf")

def test_upload_non_pdf_returns_bad_request(client):
    response = client.post(
        "api/v1/papers",
        files={
            "file": (
                "notes.txt",
                b"plain text",
                "text/plain",
            ),
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only PDF files are Supported."

def test_list_papers_returns_uploaded_paper(client):
    client.post(
        "/api/v1/papers",
        files={
            "file": (
                "attention.pdf",
                b"sample PDF content",
                "application/pdf"
            ),
        },
    )

    response = client.get("/api/v1/papers")

    assert response.status_code == 200

    papers = response.json()["papers"]
    assert len(papers) == 1
    assert papers[0]["processing"]["stored_filename"].endswith(".pdf")