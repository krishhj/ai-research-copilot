"""
Custom Exceptions for the AI Research Copilot backend.
"""

class AppError(Exception):
    """Base exception for the application"""
    pass

# Configuration
class ConfigurationError(AppError):
    """Raised when application configuration is invalid"""
    pass

#Database
class DatabaseError(AppError):
    """Raised when database operations fails"""
    pass

#PDF Processing
class PDFProcessingError(AppError):
    """Raised when PDF extraction or parsing fails"""

#Embeddings
class EmbeddingError(AppError):
    """Raised when embedding generation fails"""
    pass

#Retrieval
class RetrievalError(AppError):
    """Raised when document retrieval fails"""
    pass

#Vector Store
class VectorStoreError(AppError):
    """Raised when vector store operations fails"""
    pass

#LLM
class LLMError(AppError):
    """Raised when the large language model fails"""
    pass

#Recommendations
class RecommendationError(AppError):
    """Raised when recommendations generations fails"""
    pass

#Analytics
class AnalyticsError(AppError):
    """Raised when analytics operations fail"""
    pass