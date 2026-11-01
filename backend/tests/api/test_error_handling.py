import pymupdf
import pytest
from uuid import UUID
from fastapi.testclient import TestClient

from app.api.dependencies import get_chat_service, get_indexing_service, get_paper_service, get_retrieval_service
from app.core.auth import AuthenticatedUser, get_current_user
from app.core.constants import MAX_UPLOAD_SIZE_MB
from app.core.exceptions import LLMError
from app.main import create_app
from app.services.chat_service import ChatService
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.indexing_service import IndexingService
from app.services.metadata_extractor import MetadataExtractor
from app.services.paper_service import PaperService
from app.services.pdf_parser import PDFParser
from app.services.retrieval_service import RetrievalService
from app.services.text_cleaner import TextCleaner
from app.services.text_chunker import TextChunker
from app.storage.chroma import ChromaVectorStore
from app.storage.file_storage import FileStorage
from app.storage.sqlite import SQLitePaperRepository

TEST_USER_ID = UUID("00000000-0000-0000-0000-000000000001")


class FakeEmbeddingModel:
    def encode_document(self, sentences, *, normalize_embeddings):
        return [[1.0, 0.0] for _ in sentences]

    def encode_query(self, sentences, *, normalize_embeddings):
        return [[1.0, 0.0] for _ in sentences]


class FakeLLMService:
    def __init__(self):
        self.should_fail = False

    def generate(self, system_prompt, user_prompt):
        if self.should_fail:
            raise LLMError("Failed to generate an answer.")
        return "A grounded answer."


@pytest.fixture
def services(tmp_path):
    storage = FileStorage(base_directory=tmp_path / "files")
    repository = SQLitePaperRepository(database_path=tmp_path / "papers.db")
    vector_store = ChromaVectorStore(persist_directory=tmp_path / "chroma")
    embedding_service = EmbeddingService(model=FakeEmbeddingModel())
    retrieval_service = RetrievalService(embedding_service=embedding_service, vector_store=vector_store)
    llm_service = FakeLLMService()

    return {
        "storage": storage,
        "vector_store": vector_store,
        "llm": llm_service,
        "paper": PaperService(
            file_storage=storage,
            paper_repository=repository,
            document_processor=DocumentProcessor(
                pdf_parser=PDFParser(),
                text_cleaner=TextCleaner(),
                text_chunker=TextChunker(),
                metadata_extractor=MetadataExtractor(),
            ),
            vector_store=vector_store,
        ),
        "indexing": IndexingService(
            paper_repository=repository,
            embedding_service=embedding_service,
            vector_store=vector_store,
        ),
        "retrieval": retrieval_service,
        "chat": ChatService(
            retrieval_service=retrieval_service,
            paper_repository=repository,
            llm_service=llm_service,
        ),
    }


@pytest.fixture
def client(services):
    app = create_app()
    app.dependency_overrides[get_paper_service] = lambda: services["paper"]
    app.dependency_overrides[get_indexing_service] = lambda: services["indexing"]
    app.dependency_overrides[get_retrieval_service] = lambda: services["retrieval"]
    app.dependency_overrides[get_chat_service] = lambda: services["chat"]
    app.dependency_overrides[get_current_user] = lambda: AuthenticatedUser(
        id=TEST_USER_ID,
        email="test@example.com",
    )

    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def make_pdf(text: str | None = None) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    if text:
        page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


def upload(client, content: bytes) -> str:
    response = client.post(
        "/api/v1/papers",
        files={"file": ("paper.pdf", content, "application/pdf")},
    )
    return response.json()["paper"]["id"]


def test_reprocessing_removes_outdated_vectors(client, services):
    paper_id = upload(client, make_pdf("Transformers use self-attention."))
    client.post(f"/api/v1/papers/{paper_id}/process")
    client.post(f"/api/v1/papers/{paper_id}/index")
    assert services["vector_store"].count_by_paper_id(UUID(paper_id)) == 1

    client.post(f"/api/v1/papers/{paper_id}/process")

    assert services["vector_store"].count_by_paper_id(UUID(paper_id)) == 0


def test_delete_succeeds_when_stored_file_is_missing(client, services):
    paper_id = upload(client, make_pdf("Transformers use self-attention."))
    services["storage"].delete(f"{paper_id}.pdf")

    response = client.delete(f"/api/v1/papers/{paper_id}")

    assert response.status_code == 200
    assert client.get(f"/api/v1/papers/{paper_id}").status_code == 404


def test_confidence_is_zero_without_retrieved_chunks(client):
    paper_id = upload(client, make_pdf("Transformers use self-attention."))
    client.post(f"/api/v1/papers/{paper_id}/process")

    response = client.post(
        "/api/v1/chat",
        json={"question": "Who wrote this?", "paper_id": paper_id},
    )

    assert response.status_code == 200
    assert response.json()["confidence"] == 0.0


@pytest.mark.parametrize("path", ["/api/v1/search", "/api/v1/chat"])
def test_blank_question_returns_validation_error(client, path):
    response = client.post(path, json={"question": "   "})

    assert response.status_code == 422


def test_upload_over_size_limit_is_rejected(client):
    response = client.post(
        "/api/v1/papers",
        files={
            "file": (
                "paper.pdf",
                b"0" * (MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1),
                "application/pdf",
            ),
        },
    )

    assert response.status_code == 400
    assert client.get("/api/v1/papers").json()["papers"] == []


def test_indexing_pdf_without_text_returns_clear_error(client):
    paper_id = upload(client, make_pdf())
    client.post(f"/api/v1/papers/{paper_id}/process")

    response = client.post(f"/api/v1/papers/{paper_id}/index")

    assert response.status_code == 422
    assert "No text could be extracted" in response.json()["detail"]


def test_llm_failure_returns_readable_error_with_cors_header(client, services):
    paper_id = upload(client, make_pdf("Transformers use self-attention."))
    client.post(f"/api/v1/papers/{paper_id}/process")
    client.post(f"/api/v1/papers/{paper_id}/index")
    services["llm"].should_fail = True

    response = client.post(
        "/api/v1/chat",
        json={"question": "What is attention?"},
        headers={"Origin": "http://localhost:5173"},
    )

    assert response.status_code == 503
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
