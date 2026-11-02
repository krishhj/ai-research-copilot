from uuid import UUID, uuid4

from app.models.query import Query
from app.services.embedding_service import EmbeddingService
from app.services.reranker_service import RerankerService
from app.services.retrieval_service import RetrievalService
from app.storage.chroma import VectorSearchResult

TEST_OWNER_ID = UUID("00000000-0000-0000-0000-000000000001")


class FakeEmbeddingModel:
    def encode_query(self, sentences, *, normalize_embeddings):
        return [[0.1, 0.2, 0.3] for _ in sentences]


class FakeVectorStore:
    """Fake vector store that returns top_k results"""

    def __init__(self) -> None:
        self.requested_top_k = 0

    def search(self, query_vector, top_k, owner_id, paper_id=None):
        self.requested_top_k = top_k
        return tuple(
            VectorSearchResult(
                chunk_id=uuid4(),
                paper_id=uuid4(),
                page_number=index + 1,
                text=f"chunk {index}",
                score=1.0 - index / 100,
            )
            for index in range(top_k)
        )


class ReverseCrossEncoder:
    """Gives later chunks higher scores, so the order flips"""

    def predict(self, inputs):
        return [index / len(inputs) for index in range(len(inputs))]


def make_service(vector_store, reranker_service=None) -> RetrievalService:
    return RetrievalService(
        embedding_service=EmbeddingService(FakeEmbeddingModel()),
        vector_store=vector_store,
        reranker_service=reranker_service,
        rerank_candidates=20,
    )


def test_search_without_reranker_uses_query_top_k():
    vector_store = FakeVectorStore()
    service = make_service(vector_store)

    results = service.search(Query(question="What is RAG?", top_k=5), TEST_OWNER_ID)

    assert vector_store.requested_top_k == 5
    assert [result.text for result in results] == [f"chunk {index}" for index in range(5)]


def test_search_with_reranker_fetches_candidates_then_keeps_top_k():
    vector_store = FakeVectorStore()
    service = make_service(vector_store, RerankerService(ReverseCrossEncoder()))

    results = service.search(Query(question="What is RAG?", top_k=5), TEST_OWNER_ID)

    assert vector_store.requested_top_k == 20
    assert len(results) == 5
    assert results[0].text == "chunk 19"


def test_search_with_reranker_never_fetches_fewer_than_top_k():
    vector_store = FakeVectorStore()
    service = RetrievalService(
        embedding_service=EmbeddingService(FakeEmbeddingModel()),
        vector_store=vector_store,
        reranker_service=RerankerService(ReverseCrossEncoder()),
        rerank_candidates=3,
    )

    service.search(Query(question="What is RAG?", top_k=8), TEST_OWNER_ID)

    assert vector_store.requested_top_k == 8
