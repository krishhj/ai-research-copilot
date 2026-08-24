from uuid import UUID

from pydantic import BaseModel, Field

class SearchResultItem(BaseModel):
    """One chunk returned from semantic search"""

    chunk_id: UUID
    paper_id: UUID
    page_number: int = Field(ge=1)
    text: str
    score: float = Field(ge=0.0, le=1.0)

class SearchResponse(BaseModel):
    """Semantic-search Response"""
    results: list[SearchResultItem]