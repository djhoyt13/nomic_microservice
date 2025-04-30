from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import numpy as np
from langchain_nomic import NomicEmbeddings
import requests
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="Nomic Embedding Service")

# Database service URL
DB_SERVICE_URL = os.getenv("DB_SERVICE_URL", "http://db_service:8001")

# Initialize the embedding model
logger.info("Loading embedding model...")
embeddings = NomicEmbeddings(model='nomic-embed-text-v1.5')
logger.info("Model loaded successfully")

# Custom exceptions
class NomicServiceError(Exception):
    """Base exception for Nomic service errors"""
    pass

class EmbeddingError(NomicServiceError):
    """Exception raised for errors in embedding generation"""
    pass

class DatabaseServiceError(NomicServiceError):
    """Exception raised for errors in database service communication"""
    pass

class ValidationError(NomicServiceError):
    """Exception raised for input validation errors"""
    pass

# Data models with validation
class Document(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    metadata: Optional[dict] = None

class Query(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    top_k: Optional[int] = Field(default=5, ge=1, le=100)

# Error handlers
@app.exception_handler(NomicServiceError)
async def nomic_service_error_handler(request: Request, exc: NomicServiceError):
    logger.error(f"Nomic service error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": exc.__class__.__name__,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={
            "error": str(exc),
            "type": exc.__class__.__name__,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(requests.exceptions.RequestException)
async def request_exception_handler(request: Request, exc: requests.exceptions.RequestException):
    logger.error(f"Database service communication error: {str(exc)}")
    return JSONResponse(
        status_code=503,
        content={
            "error": "Database service is currently unavailable",
            "details": str(exc),
            "type": exc.__class__.__name__,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.post("/embed")
async def create_embedding(document: Document):
    try:
        logger.info("Received document for embedding")
        # Validate input
        if not document.text.strip():
            raise ValidationError("Document text cannot be empty or whitespace")
        logger.info("Input validation successful")

        # Generate embedding using local Nomic model
        try:
            logger.info("Generating embedding...")
            embedding = embeddings.embed_documents([document.text])[0]
            logger.info(f"Generated embedding of shape: {len(embedding)}")
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")
        
        # Store in database service
        try:
            logger.info("Storing embedding in database service...")
            response = requests.post(
                f"{DB_SERVICE_URL}/store",
                json={
                    "text": document.text,
                    "embedding": embedding,
                    "metadata": document.metadata
                }
            )
            response.raise_for_status()
            logger.info("Successfully stored embedding")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to store embedding: {str(e)}")
            raise DatabaseServiceError(f"Failed to store embedding: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise NomicServiceError(f"Unexpected error: {str(e)}")

@app.post("/search")
async def search_similar(query: Query):
    try:
        logger.info("Received search query")
        
        # Generate embedding for the query text
        try:
            logger.info("Generating query embedding...")
            query_embedding = embeddings.embed_documents([query.text])[0]
            logger.info(f"Generated query embedding of shape: {len(query_embedding)}")
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {str(e)}")
            raise EmbeddingError(f"Failed to generate query embedding: {str(e)}")

        # Search in database service
        try:
            response = requests.post(
                f"{DB_SERVICE_URL}/search",
                json={
                    "embedding": query_embedding.tolist(),  # Convert numpy array to list
                    "top_k": query.top_k
                }
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to search embeddings: {str(e)}")
            raise DatabaseServiceError(f"Failed to search embeddings: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise NomicServiceError(f"Unexpected error: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": "nomic-embed-text-v1.5",
        "embedding_dimension": 768,  # Nomic Embed v1.5 uses 768 dimensions
        "timestamp": datetime.now().isoformat()
    } 