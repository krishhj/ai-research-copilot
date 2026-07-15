"""
Application enums
"""

from enum import Enum


# Paper Status
class PaperStatus(str, Enum):
    """Processing status of a research paper"""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


# Conversation Roles
class ConversationRoles(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
