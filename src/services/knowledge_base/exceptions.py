class KnowledgeBaseError(Exception):
    """Base exception for the Knowledge Base Retrieval Layer."""
    pass

class KnowledgeBaseNotFound(KnowledgeBaseError):
    """Raised when the knowledge base configuration or files cannot be found."""
    pass

class DiseaseNotFound(KnowledgeBaseError):
    """Raised when a specific disease cnn_class cannot be found in the knowledge base."""
    pass

class InvalidKnowledgeBase(KnowledgeBaseError):
    """Raised when the knowledge base data is malformed."""
    pass

class ValidationError(KnowledgeBaseError):
    """Raised when the knowledge base fails runtime schema validation."""
    pass
