from uuid import UUID
from pydantic import BaseModel, Field

class Citation(BaseModel):
    """Represents a citation used in an answer"""
    paper_id: UUID
    paper_title: str
    chunk_id: UUID
    page_number: int = Field(ge=1)

class RetrievedChunk(BaseModel):
    """Represents a retrieved chunk and its relevance score"""
    chunk_id: UUID
    score: float = Field(ge=0.0, le=1.0)

class Answer(BaseModel):
    """Represents the AI-generated Answer"""
    answer_text: str = Field(
        min_length= 1,
        description= "Generated Answer"
    )
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    confidence: float = Field(
        default = 0.0,
        ge = 0.0,
        le = 1.0,
        description= "Estimated confidence score."
    )

    generation_time: float = Field(
        default= 0.0,
        ge= 0.0,
        description= "Generation time in seconds"
    )