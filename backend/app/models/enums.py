"""
Application enums
"""

from enum import Enum

class PaperStatus(str, Enum):
    """Processing status of a research paper"""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"