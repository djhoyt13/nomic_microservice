from fastapi import HTTPException
from typing import Optional

class NomicServiceError(HTTPException):
    """Base exception for Nomic service errors"""
    def __init__(self, detail: str, status_code: int = 500):
        super().__init__(status_code=status_code, detail=detail)

class EmbeddingError(NomicServiceError):
    """Exception raised for embedding generation errors"""
    def __init__(self, detail: str):
        super().__init__(detail=detail, status_code=500)

class DatabaseServiceError(NomicServiceError):
    """Exception raised for database service errors"""
    def __init__(self, detail: str):
        super().__init__(detail=detail, status_code=503) 