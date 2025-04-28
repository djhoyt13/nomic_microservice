from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
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
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import time
from sqlalchemy.orm.attributes import instance_state
from sqlalchemy.orm import class_mapper

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

    @field_validator('embedding')
    @classmethod
    def validate_embedding(cls, v):
        if len(v) != 768:
            raise ValueError("Embedding must be 768-dimensional")
        return v

class Query(BaseModel):
    embedding: List[float]
    top_k: Optional[int] = Field(default=5, ge=1, le=100)

    @field_validator('embedding')
    @classmethod
    def validate_embedding(cls, v):
        if len(v) != 768:
            raise ValueError("Embedding must be 768-dimensional")
        return v

class SearchResult(BaseModel):
    id: int
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    similarity: float
    created_at: str

    @field_validator('metadata', mode='before')
    @classmethod
    def ensure_dict(cls, v):
        if v is None:
            return {}
        if hasattr(v, '_sa_instance_state'):
            # This is a SQLAlchemy object, convert it to dict
            return {}
        return dict(v)

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

# Create FastAPI app
app = FastAPI(
    title="Document Embedding Service",
    description="Service for storing and searching document embeddings",
    version="1.0.0",
    docs_url="/docs",  # Enable Swagger UI
    redoc_url="/redoc"  # Enable ReDoc
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic route for testing
@app.get("/")
async def root():
    return {"message": "Document Embedding Service is running"}

# Error handlers
@app.exception_handler(ServiceError)
async def service_error_handler(request, exc: ServiceError):
    logger.error(f"Service error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=str(exc),
            timestamp=datetime.now(timezone.utc).isoformat()
        ).model_dump()
    )

@app.exception_handler(DatabaseError)
async def database_error_handler(request, exc: DatabaseError):
    logger.error(f"Database error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(
            error="Database service error",
            details=str(exc),
            timestamp=datetime.now(timezone.utc).isoformat()
        ).model_dump()
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request, exc: ValidationError):
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Validation error",
            details=str(exc),
            timestamp=datetime.now(timezone.utc).isoformat()
        ).model_dump()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    logger.error(f"HTTP error: {str(exc)}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.now(timezone.utc).isoformat()
        ).model_dump()
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
            "created_at": db_embedding.created_at.isoformat()
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

def object_as_dict(obj):
    """Convert SQLAlchemy object to dictionary"""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, '_sa_instance_state'):
        # This is a SQLAlchemy object
        return {}
    try:
        return dict(obj)
    except:
        return {}

@app.post("/search", response_model=List[SearchResult])
async def search_similar(query: Query, db: Session = Depends(get_db)):
    try:
        # Validate embedding dimensions
        if len(query.embedding) != 768:
            raise ValidationError("Embedding must be 768-dimensional")

        # Convert query embedding to numpy array
        query_embedding = np.array(query.embedding)
        
        # Get all documents
        results = db.query(DocumentEmbedding).all()
        
        # Calculate similarities
        similarities = []
        for doc in results:
            doc_embedding = np.array(doc.embedding)
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
            )
            similarities.append((doc, similarity))
        
        # Sort by similarity and take top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_results = similarities[:query.top_k]
        
        # Format results
        formatted_results = []
        for doc, similarity in top_results:
            doc_dict = doc.to_dict()
            result = SearchResult(
                id=doc_dict['id'],
                text=doc_dict['text'],
                metadata=doc_dict['metadata'],
                similarity=float(similarity),
                created_at=doc_dict['created_at']
            )
            formatted_results.append(result)
        
        return formatted_results
    except ValidationError as e:
        raise
    except SQLAlchemyError as e:
        logger.error(f"Database error while searching: {str(e)}")
        raise DatabaseError(f"Failed to search documents: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while searching: {str(e)}")
        raise ServiceError(f"Unexpected error while searching: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        # Check database connection
        logger.info("Starting health check...")
        db_healthy = check_db_connection()
        logger.info(f"Database health check result: {db_healthy}")
        
        if not db_healthy:
            logger.warning("Database health check failed")
            return {"status": "error", "message": "Database connection failed"}
        
        logger.info("Health check successful")
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Health check failed with error: {str(e)}")
        return {"status": "error", "message": str(e)}

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
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