from fastapi import APIRouter, Depends

from app.api.dependencies import get_retrieval_service
from app.models.query import Query
from app.models.search_schemas import SearchResponse, SearchResultItem
from app.services.retrieval_service import RetrievalService

router = APIRouter(prefix="/search", tags=["Search"])

@router.post("", response_model=SearchResponse)
def search_papers(
    query: Query,
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
) -> SearchResponse:
    """Search indexed paper chunks by meaning"""
    results = retrieval_service.search(query)

    return SearchResponse(
        results=[
            SearchResultItem(
                chunk_id= result.chunk_id,
                paper_id= result.paper_id,
                page_number=result.page_number,
                text=result.text,
                score=result.score
            )
            for result in results
        ]
    )