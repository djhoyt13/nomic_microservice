from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from .database import get_db, DocumentEmbedding, init_db, check_db_connection
from .database import (
    DatabaseError,
    DatabaseConnectionError,
    DatabaseInitializationError,
    DatabaseSessionError
)
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from contextlib import asynccontextmanager
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Custom exceptions
class ServiceError(Exception):
    """Base exception for service errors"""
    pass

class ValidationError(ServiceError):
    """Exception raised for validation errors"""
    pass

class SearchError(ServiceError):
    """Exception raised for search operation errors"""
    pass

# Data models with enhanced validation
class Document(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None

    @validator('embedding')
    def validate_embedding(cls, v):
        if len(v) != 768:
            raise ValueError("Embedding must be 768-dimensional")
        return v

class Query(BaseModel):
    embedding: List[float]
    top_k: Optional[int] = Field(default=5, ge=1, le=100)

    @validator('embedding')
    def validate_embedding(cls, v):
        if len(v) != 768:
            raise ValueError("Embedding must be 768-dimensional")
        return v

class SearchResult(BaseModel):
    id: int
    text: str
    metadata: Optional[Dict[str, Any]]
    similarity: float
    created_at: str

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    timestamp: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting database service...")
    try:
        init_db()
        logger.info("Database initialized successfully")
    except DatabaseInitializationError as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during startup: {str(e)}")
        raise ServiceError(f"Unexpected error during startup: {str(e)}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down database service...")
    # Add any cleanup code here if needed

app = FastAPI(
    title="Vector Database Service",
    description="Service for storing and searching document embeddings",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error handlers
@app.exception_handler(ServiceError)
async def service_error_handler(request, exc: ServiceError):
    logger.error(f"Service error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=str(exc),
            timestamp=datetime.now(datetime.UTC).isoformat()
        ).dict()
    )

@app.exception_handler(DatabaseError)
async def database_error_handler(request, exc: DatabaseError):
    logger.error(f"Database error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error="Database service error",
            details=str(exc),
            timestamp=datetime.now(datetime.UTC).isoformat()
        ).dict()
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request, exc: ValidationError):
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Validation error",
            details=str(exc),
            timestamp=datetime.now(datetime.UTC).isoformat()
        ).dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    logger.error(f"HTTP error: {str(exc)}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.now(datetime.UTC).isoformat()
        ).dict()
    )

@app.post("/store", response_model=Dict[str, Any])
async def store_document(document: Document, db: Session = Depends(get_db)):
    try:
        # Validate embedding dimensions
        if len(document.embedding) != 768:
            raise ValidationError("Embedding must be 768-dimensional")

        # Store in database
        db_embedding = DocumentEmbedding(
            text=document.text,
            embedding=document.embedding,
            metadata=document.metadata
        )
        db.add(db_embedding)
        db.commit()
        db.refresh(db_embedding)
            
        return {
            "id": db_embedding.id,
            "status": "success",
            "created_at": db_embedding.created_at
        }
    except ValidationError as e:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error while storing document: {str(e)}")
        raise DatabaseError(f"Failed to store document: {str(e)}")
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error while storing document: {str(e)}")
        raise ServiceError(f"Unexpected error while storing document: {str(e)}")

@app.post("/search", response_model=List[SearchResult])
async def search_similar(query: Query, db: Session = Depends(get_db)):
    try:
        # Validate embedding dimensions
        if len(query.embedding) != 768:
            raise ValidationError("Embedding must be 768-dimensional")

        # Search using PGVector
        results = db.query(DocumentEmbedding).order_by(
            DocumentEmbedding.embedding.cosine_distance(query.embedding)
        ).limit(query.top_k).all()
        
        # Format results
        formatted_results = []
        for doc in results:
            similarity = 1 - doc.embedding.cosine_distance(query.embedding)
            formatted_results.append(SearchResult(
                id=doc.id,
                text=doc.text,
                metadata=doc.metadata,
                similarity=float(similarity),
                created_at=doc.created_at
            ))
        
        return formatted_results
    except ValidationError as e:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error while searching: {str(e)}")
        raise DatabaseError(f"Failed to search documents: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while searching: {str(e)}")
        raise ServiceError(f"Unexpected error while searching: {str(e)}")

@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    try:
        # Check database connection
        db_healthy = check_db_connection()
        
        return {
            "status": "healthy" if db_healthy else "unhealthy",
            "database": "connected" if db_healthy else "disconnected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "database": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        } 