from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from .database import (
    get_db,
    DocumentEmbedding,
    init_db,
    check_db_connection,
    search_similar_embeddings,
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
import numpy as np

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

# Models
class Document(BaseModel):
    text: str
    embedding: List[float]
    metadata: Optional[Dict[str, Any]] = None

    @validator('embedding')
    def validate_embedding(cls, v):
        if len(v) != 768:
            raise ValueError("Embedding must be 768-dimensional")
        return v

class Query(BaseModel):
    embedding: List[float]
    top_k: Optional[int] = 5

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

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize database
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise
    yield
    # Cleanup
    logger.info("Shutting down application")

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
        status_code=500,
        content={
            "error": str(exc),
            "type": exc.__class__.__name__,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(DatabaseError)
async def database_error_handler(request, exc: DatabaseError):
    logger.error(f"Database error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": exc.__class__.__name__,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request, exc: ValidationError):
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={
            "error": str(exc),
            "type": exc.__class__.__name__,
            "timestamp": datetime.now().isoformat()
        }
    )

# Health check endpoint
@app.get("/health")
async def health_check():
    """Check service health"""
    try:
        if check_db_connection():
            return {"status": "healthy", "database": "connected"}
        return {"status": "unhealthy", "database": "disconnected"}
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {"status": "unhealthy", "error": str(e)}

@app.post("/store", response_model=Dict[str, Any])
async def store_document(document: Document, db: Session = Depends(get_db)):
    try:
        logger.info("Received document for storage")
        # Validate embedding dimensions
        if len(document.embedding) != 768:
            logger.error(f"Invalid embedding dimensions: {len(document.embedding)}")
            raise ValidationError("Embedding must be 768-dimensional")
        logger.info("Embedding validation successful")

        # Store in database
        logger.info("Creating database record")
        db_embedding = DocumentEmbedding(
            text=document.text,
            embedding=document.embedding,
            metadata=document.metadata
        )
        logger.info("Adding record to session")
        db.add(db_embedding)
        logger.info("Committing transaction")
        db.commit()
        logger.info("Refreshing record")
        db.refresh(db_embedding)
        logger.info("Document stored successfully")
            
        return {
            "id": db_embedding.id,
            "status": "success",
            "created_at": db_embedding.created_at
        }
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
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
async def search_similar(query: Query):
    try:
        # Search using SQLite implementation
        results = search_similar_embeddings(query.embedding, query.top_k)
        return results
    except ValidationError as e:
        raise
    except DatabaseError as e:
        logger.error(f"Database error while searching: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while searching: {str(e)}")
        raise ServiceError(f"Unexpected error while searching: {str(e)}") 