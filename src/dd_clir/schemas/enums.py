from enum import StrEnum

class Device(StrEnum):
    """Device enum"""
    cpu = "cpu"
    cuda = "cuda"

class EmbeddingType(StrEnum):
    """Embedding type enum"""
    query = "query"
    passage = "passage"