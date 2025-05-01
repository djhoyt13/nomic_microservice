from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import requests
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timezone

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
model_name = "nomic-ai/nomic-embed-text-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
logger.info("Model loaded successfully")

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using the local model"""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy().tolist()
    return embeddings

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

# Data models with validation
class Document(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    metadata: Optional[dict] = None

class BatchDocument(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=1000)
    metadata: Optional[dict] = None

class Query(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    top_k: Optional[int] = Field(default=5, ge=1, le=100)

# Constants for batch processing
BATCH_SIZE = 32  # Number of texts to process in parallel
MAX_BATCH_SIZE = 1000  # Maximum number of texts in a single request
CHUNK_SIZE = 100  # Number of texts to store in database at once

def chunk_list(lst: List, size: int) -> List[List]:
    """Split a list into chunks of specified size"""
    return [lst[i:i + size] for i in range(0, len(lst), size)]

async def store_embeddings_batch(texts: List[str], embeddings: List[List[float]], metadata: Optional[dict] = None) -> List[dict]:
    """Store a batch of embeddings in the database service"""
    results = []
    for chunk_texts, chunk_embeddings in zip(chunk_list(texts, CHUNK_SIZE), chunk_list(embeddings, CHUNK_SIZE)):
        try:
            response = requests.post(
                f"{DB_SERVICE_URL}/store_batch",
                json={
                    "texts": chunk_texts,
                    "embeddings": chunk_embeddings,
                    "metadata": metadata
                },
                timeout=10  # Increased timeout for batch operations
            )
            
            if response.status_code != 200:
                error_detail = response.json().get('detail', 'Unknown error')
                logger.error(f"Database service error: {error_detail}")
                raise DatabaseServiceError(f"Database service error: {error_detail}")
                
            results.extend(response.json())
        except requests.exceptions.Timeout:
            logger.error("Database service request timed out")
            raise DatabaseServiceError("Database service request timed out")
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to database service")
            raise DatabaseServiceError("Failed to connect to database service")
    
    return results

# Error handlers
@app.exception_handler(NomicServiceError)
async def nomic_service_error_handler(request: Request, exc: NomicServiceError):
    logger.error(f"Nomic service error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "timestamp": datetime.now(timezone.utc).isoformat()}
    )

@app.exception_handler(ValidationError)
async def validation_error_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"error": str(exc), "timestamp": datetime.now(timezone.utc).isoformat()}
    )

@app.exception_handler(requests.exceptions.RequestException)
async def request_exception_handler(request: Request, exc: requests.exceptions.RequestException):
    logger.error(f"Database service communication error: {str(exc)}")
    return JSONResponse(
        status_code=503,
        content={
            "error": "Database service is currently unavailable",
            "details": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

@app.post("/embed")
async def create_embedding(document: Document):
    try:
        # Validate input
        if not document.text.strip():
            raise ValidationError("Document text cannot be empty or whitespace")

        # Generate embedding using Nomic
        try:
            embedding = get_embeddings([document.text])
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")
        
        # Store in database service
        try:
            response = requests.post(
                f"{DB_SERVICE_URL}/store",
                json={
                    "text": document.text,
                    "embedding": embedding[0],
                    "metadata": document.metadata
                },
                timeout=5
            )
            
            if response.status_code != 200:
                error_detail = response.json().get('detail', 'Unknown error')
                logger.error(f"Database service error: {error_detail}")
                raise DatabaseServiceError(f"Database service error: {error_detail}")
                
            return response.json()
        except requests.exceptions.Timeout:
            logger.error("Database service request timed out")
            raise DatabaseServiceError("Database service request timed out")
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to database service")
            raise DatabaseServiceError("Failed to connect to database service")
            
    except ValidationError as e:
        raise
    except NomicServiceError as e:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise NomicServiceError(f"Unexpected error: {str(e)}")

@app.post("/embed_batch")
async def create_embeddings_batch(batch: BatchDocument):
    try:
        # Validate input
        if not all(text.strip() for text in batch.texts):
            raise ValidationError("Document texts cannot be empty or whitespace")
        
        if len(batch.texts) > MAX_BATCH_SIZE:
            raise ValidationError(f"Batch size exceeds maximum of {MAX_BATCH_SIZE} texts")

        # Process texts in batches
        results = []
        for i in range(0, len(batch.texts), BATCH_SIZE):
            batch_texts = batch.texts[i:i + BATCH_SIZE]
            try:
                # Generate embeddings for the batch
                embeddings = get_embeddings(batch_texts)
                
                # Store embeddings in chunks
                batch_results = await store_embeddings_batch(batch_texts, embeddings, batch.metadata)
                results.extend(batch_results)
                
                logger.info(f"Processed batch {i//BATCH_SIZE + 1} of {(len(batch.texts) + BATCH_SIZE - 1)//BATCH_SIZE}")
                
            except Exception as e:
                logger.error(f"Failed to process batch {i//BATCH_SIZE + 1}: {str(e)}")
                raise EmbeddingError(f"Failed to process batch {i//BATCH_SIZE + 1}: {str(e)}")
        
        return {
            "status": "success",
            "processed_count": len(results),
            "results": results
        }
            
    except ValidationError as e:
        raise
    except NomicServiceError as e:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise NomicServiceError(f"Unexpected error: {str(e)}")

@app.post("/search")
async def search_similar(query: Query):
    try:
        # Validate input
        if not query.text.strip():
            raise ValidationError("Query text cannot be empty or whitespace")

        # Generate query embedding
        try:
            query_embedding = get_embeddings([query.text])
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {str(e)}")
            raise EmbeddingError(f"Failed to generate query embedding: {str(e)}")
        
        # Search in database service
        try:
            response = requests.post(
                f"{DB_SERVICE_URL}/search",
                json={
                    "embedding": query_embedding[0],
                    "top_k": query.top_k
                },
                timeout=5  # Add timeout for database service calls
            )
            
            if response.status_code != 200:
                error_detail = response.json().get('detail', 'Unknown error')
                logger.error(f"Database service error: {error_detail}")
                raise DatabaseServiceError(f"Database service error: {error_detail}")
                
            return response.json()
        except requests.exceptions.Timeout:
            logger.error("Database service request timed out")
            raise DatabaseServiceError("Database service request timed out")
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to database service")
            raise DatabaseServiceError("Failed to connect to database service")
            
    except ValidationError as e:
        raise
    except NomicServiceError as e:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise NomicServiceError(f"Unexpected error: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        # Check database service health
        response = requests.get(f"{DB_SERVICE_URL}/health", timeout=2)
        if response.status_code != 200:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "database_service": "unavailable",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
        
        return {
            "status": "healthy",
            "model": "nomic-embed-text-v1.5",
            "embedding_dimension": 768,
            "database_service": "available",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "database_service": "unavailable",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        )

@app.get("/embedding_info")
async def get_embedding_info():
    """Debug endpoint to check embedding dimensions"""
    test_text = "This is a test"
    embedding = get_embeddings([test_text])[0]
    return {
        "dimension": len(embedding),
        "sample_embedding": embedding[:5],  # Show first 5 values
        "model": "nomic-embed-text-v1.5"
    } 