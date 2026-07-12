from uuid import UUID
from pydantic import BaseModel, Field

class Query(BaseModel):
    """Represents a user's retrieval request"""

    question: str = Field(
        min_length=1,
        description="User's Question"
    )

    top_k: int = Field(
        default=5,
        ge = 1,
        le=20,
        description="Number of chunks to retrieve"
    )

    paper_id: UUID | None = Field(
        default=None,
        description="Optional Paper to restrict the search"
    )

    conversation_id: UUID | None = Field(
        default= None,
        description="Optional conversation identifier"
    )