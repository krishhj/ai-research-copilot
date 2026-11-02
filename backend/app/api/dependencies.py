from groq import Groq
from functools import lru_cache

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.core.config import settings
from app.core.constants import PAPERS_DIR, PAPERS_DATABASE_PATH, VECTOR_STORE_DIR
from app.services.document_processor import DocumentProcessor
from app.services.chat_service import ChatService
from app.services.embedding_service import EmbeddingService
from app.services.indexing_service import IndexingService
from app.services.llm_service import LLMService
from app.services.metadata_extractor import MetadataExtractor
from app.services.paper_service import PaperService
from app.services.pdf_parser import PDFParser
from app.services.reranker_service import RerankerService
from app.services.retrieval_service import RetrievalService
from app.services.text_cleaner import TextCleaner
from app.services.text_chunker import TextChunker
from app.storage.chroma import ChromaVectorStore
from app.storage.file_storage import FileStorage
from app.storage.sqlite import SQLitePaperRepository


def get_paper_service() -> PaperService:
    """Provide a PaperService with local file storage"""
    file_storage = FileStorage(base_directory=PAPERS_DIR)
    paper_repository = SQLitePaperRepository(database_path= PAPERS_DATABASE_PATH)

    return PaperService(
        file_storage=file_storage, 
        paper_repository=paper_repository,
        document_processor=DocumentProcessor(
            pdf_parser=PDFParser(),
            text_cleaner=TextCleaner(),
            text_chunker=TextChunker(),
            metadata_extractor=MetadataExtractor()
        ),
        vector_store = get_vector_store()
    )

@lru_cache
def get_embedding_service() -> EmbeddingService:
    """Create the application's shared embedding model."""
    model = SentenceTransformer(settings.embedding_model)
    return EmbeddingService(model=model)

@lru_cache
def get_vector_store() -> ChromaVectorStore:
    """Create the application's shared ChromaDB Vector Store"""
    return ChromaVectorStore(persist_directory=VECTOR_STORE_DIR)

def get_indexing_service() -> IndexingService:
    """Provide the service that indexes processed paper chunks"""
    return IndexingService(
        paper_repository=SQLitePaperRepository(database_path=PAPERS_DATABASE_PATH),
        embedding_service=get_embedding_service(),
        vector_store= get_vector_store()
    )

@lru_cache
def get_reranker_service() -> RerankerService | None:
    """Create the shared reranker model, or None when reranking is turned off."""
    if not settings.rerank_enabled:
        return None

    # Sigmoid keeps scores between 0 and 1, max_length=256 keeps it fast on CPU
    model = CrossEncoder(settings.reranker_model, max_length=256, activation_fn=torch.nn.Sigmoid())
    return RerankerService(model=model)

def get_retrieval_service() -> RetrievalService:
    """Provide the semantic-retrieval service"""
    return RetrievalService(
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
        reranker_service=get_reranker_service(),
        rerank_candidates=settings.rerank_candidates
    )

@lru_cache
def get_llm_service() -> LLMService:
    """Create the shared Groq language-model service."""
    return LLMService(
        client=Groq(api_key=settings.groq_api_key),
        model=settings.groq_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens
    )

def get_chat_service() -> ChatService:
    """Provide the grounded RAG chat service"""
    return ChatService(
        retrieval_service=get_retrieval_service(),
        paper_repository=SQLitePaperRepository(database_path=PAPERS_DATABASE_PATH),
        llm_service=get_llm_service()
    )