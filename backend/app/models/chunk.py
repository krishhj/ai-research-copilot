from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """Represents a retrievable chunk of a research paper"""

    # Unique identifier for the chunk
    id: UUID = Field(default_factory=uuid4)

    # Reference to the parent paper
    paper_id: UUID

    # Position of the chunk within the paper (0-based)
    chunk_index: int = Field(ge=0, description="Zero-based index of the chunk within the paper.")

    # PDF page number (1-based)

    page_number: int = Field(ge=1, description="One-based page number in the original PDF.")

    # Actual text content
    chunk_text: str

    # Number of tokens in the chunk
    token_count: int = Field(ge=0)

    # Whether an embedding has been generated
    embedding_created: bool = False
