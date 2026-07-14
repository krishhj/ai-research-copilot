from datetime import UTC, datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field
from app.models.enums import ConversationRoles

class ConversationMessage(BaseModel):
    """Represents one message in a conversation."""

    id: UUID = Field(default_factory=uuid4)
    role: ConversationRoles
    content: str = Field(
        min_length=1,
        description="Message content."
    )
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

class Conversation(BaseModel):
    """Represents a conversation with the AI."""

    id: UUID = Field(default_factory=uuid4)
    messages: list[ConversationMessage] = Field(default_factory=list)