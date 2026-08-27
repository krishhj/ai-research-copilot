import pytest
import pymupdf
from fastapi.testclient import TestClient

from app.api.dependencies import get_paper_service
from app.main import create_app
from app.services.document_processor import DocumentProcessor
from app.services.metadata_extractor import MetadataExtractor
from app.services.paper_service import PaperService
from app.services.pdf_parser import PDFParser
from app.services.text_cleaner import TextCleaner
from app.services.text_chunker import TextChunker
from app.storage.chroma import ChromaVectorStore
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
        vector_store = ChromaVectorStore(
            persist_directory=tmp_path / "chroma",
        )
        return PaperService(
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

def test_get_paper_returns_uploaded_paper(client):
    upload_response = client.post(
        "/api/v1/papers",
        files={
            "file": (
                "attention.pdf",
                b"sample PDF content",
                "application/pdf",
            ),
        },
    )
    paper_id = upload_response.json()["paper"]["id"]

    response = client.get(f"/api/v1/papers/{paper_id}")

    assert response.status_code == 200
    assert response.json()["paper"]["id"] == paper_id


def test_get_missing_paper_returns_not_found(client):
    response = client.get(
        "/api/v1/papers/00000000-0000-0000-0000-000000000000",
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Paper not found."

def test_delete_paper_removes_metadata_and_file(client):
    upload_response = client.post(
        "/api/v1/papers",
        files={
            "file": (
                "attention.pdf",
                b"sample PDF content",
                "application/pdf",
            ),
        },
    )
    paper_id = upload_response.json()["paper"]["id"]

    delete_response = client.delete(f"/api/v1/papers/{paper_id}")

    assert delete_response.status_code == 200
    assert delete_response.json()["message"] == "Paper deleted successfully"

    get_response = client.get(f"/api/v1/papers/{paper_id}")
    assert get_response.status_code == 404

    list_response = client.get("/api/v1/papers")
    assert list_response.json()["papers"] == []


def test_delete_missing_paper_returns_not_found(client):
    response = client.delete(
        "/api/v1/papers/00000000-0000-0000-0000-000000000000",
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Paper not found."

def test_process_paper_creates_chunks(client, tmp_path):
    pdf_path = tmp_path / "attention.pdf"

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72,72), "Transformers use self-attention for sequence modelling.")
    document.save(pdf_path)
    document.close()

    upload_response = client.post(
        "/api/v1/papers",
        files={
            "file": (
                "attention.pdf",
                pdf_path.read_bytes(),
                "application/pdf"
                ),
            }
        )
    paper_id = upload_response.json()["paper"]["id"]

    response = client.post(f"/api/v1/papers/{paper_id}/process")

    assert response.status_code == 200

    paper = response.json()["paper"]
    assert paper["processing"]["status"] == "processed"
    assert paper["processing"]["total_pages"] == 1
    assert paper["processing"]["total_chunks"] > 0