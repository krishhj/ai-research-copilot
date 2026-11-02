"""
Compare plain vector search with vector search + reranker.

Run from backend/:
    python -m evaluation.evaluate_retrieval data/papers/<paper>.pdf
"""

import json
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from uuid import UUID, uuid4

from app.api.dependencies import get_embedding_service, get_reranker_service
from app.models.query import Query
from app.services.document_processor import DocumentProcessor
from app.services.metadata_extractor import MetadataExtractor
from app.services.pdf_parser import PDFParser
from app.services.retrieval_service import RetrievalService
from app.services.text_chunker import TextChunker
from app.services.text_cleaner import TextCleaner
from app.storage.chroma import ChromaVectorStore
from evaluation.metrics import first_relevant_rank, mean_reciprocal_rank, recall_at_k

OWNER_ID = UUID("00000000-0000-0000-0000-000000000001")
QUESTIONS_FILE = Path(__file__).parent / "rag_paper_questions.json"


def evaluate(name, retrieval_service, questions):
    """Ask every question and print the scores"""
    ranks = []
    start = perf_counter()

    for item in questions:
        results = retrieval_service.search(Query(question=item["question"]), OWNER_ID)
        pages = [result.page_number for result in results]
        ranks.append(first_relevant_rank(pages, item["expected_pages"]))

    latency_ms = (perf_counter() - start) * 1000 / len(questions)

    print(
        f"{name:<16} Recall@1={recall_at_k(ranks, 1):.2f}  Recall@3={recall_at_k(ranks, 3):.2f}  "
        f"Recall@5={recall_at_k(ranks, 5):.2f}  MRR={mean_reciprocal_rank(ranks):.2f}  "
        f"Latency={latency_ms:.0f}ms"
    )
    return ranks


def main():
    pdf_path = Path(sys.argv[1])
    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))["questions"]

    embedding_service = get_embedding_service()
    reranker_service = get_reranker_service()
    if reranker_service is None:
        sys.exit("Set RERANK_ENABLED=true to run the evaluation.")

    processor = DocumentProcessor(
        pdf_parser=PDFParser(),
        text_cleaner=TextCleaner(),
        text_chunker=TextChunker(),
        metadata_extractor=MetadataExtractor(),
    )

    # Use a temporary vector store so the real app data is not touched.
    # Chroma keeps files open on Windows, so cleanup errors are ignored.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        vector_store = ChromaVectorStore(persist_directory=Path(temp_dir))
        document = processor.process(pdf_path, paper_id=uuid4())
        vector_store.upsert(
            chunks=document.chunks,
            embedded_chunks=embedding_service.embed_chunks(document.chunks),
            owner_id=OWNER_ID,
        )

        dense = RetrievalService(embedding_service, vector_store)
        reranked = RetrievalService(embedding_service, vector_store, reranker_service)

        # The first call to each model is slow, so don't count it
        reranked.search(Query(question="warm up"), OWNER_ID)

        dense_ranks = evaluate("Dense only", dense, questions)
        reranked_ranks = evaluate("Dense + rerank", reranked, questions)

    print("\nQuestions whose rank changed (None = not in top 5):")
    for item, before, after in zip(questions, dense_ranks, reranked_ranks):
        if before != after:
            print(f"  Q{item['id']}: {before} -> {after}  {item['question']}")


if __name__ == "__main__":
    main()
