from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from typing import List, Optional, Dict, Any
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import requests
import logging
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager

from .models import (
    NomicServiceError,
    EmbeddingError,
    DatabaseServiceError,
    Document,
    BatchDocument,
    Query,
    DocumentUpdate,
    BatchDocumentUpdate,
    BatchDocumentDelete
)

from .funcs import (
    init_model,
    validate_token_length,
    get_embeddings,
    store_embeddings_batch,
    search_similar,
    MODEL_NAME,
    MAX_LENGTH,
    BATCH_SIZE,
    DB_SERVICE_URL,
    CHUNK_SIZE,
    MAX_BATCH_SIZE,
    nomic_service_error_handler,
    validation_error_handler,
    request_exception_handler,
    DATABASE_URL,
    configure_logging,
    lifespan
)

# Create logger
logger = configure_logging()

# Create async engine and session
engine = create_async_engine(DATABASE_URL)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Initialize FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Error handlers
app.exception_handler(NomicServiceError)(nomic_service_error_handler)
app.exception_handler(ValidationError)(validation_error_handler)
app.exception_handler(requests.exceptions.RequestException)(request_exception_handler)

# Health and Monitoring Endpoints
@app.get("/health")
async def health_check():
    """Check the health of the service and its dependencies"""
    try:
        # Check database service health
        db_response = requests.get(f"{DB_SERVICE_URL}/health")
        db_response.raise_for_status()
        db_status = db_response.json()
        
        return {
            "status": "healthy",
            "model": MODEL_NAME,
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "db_service": db_status
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Database service health check failed: {str(e)}")
        raise DatabaseServiceError(detail=f"Database service health check failed: {str(e)}")
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise NomicServiceError(detail=f"Health check failed: {str(e)}")

# Embedding Endpoints   
@app.post("/embed")
async def create_embedding(document: Document):
    """Create an embedding for a single document"""
    try:
        # Validate token length
        if not validate_token_length(document.text):
            raise EmbeddingError(detail=f"Text exceeds maximum token length of {MAX_LENGTH}")
        
        # Generate embedding
        embeddings = get_embeddings([document.text])
        if not embeddings:
            raise EmbeddingError(detail="Failed to generate embedding")
        
        # Store embedding
        success, message = store_embeddings_batch(
            texts=[document.text],
            metadata=document.metadata
        )
        
        if not success:
            raise DatabaseServiceError(detail=message)
        
        return {"status": "success", "message": message}
    except (NomicServiceError, EmbeddingError, DatabaseServiceError) as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error while creating embedding: {str(e)}")
        raise NomicServiceError(detail=f"Unexpected error while creating embedding: {str(e)}")

@app.post("/embed_batch")
async def create_embeddings_batch(documents: BatchDocument):
    """Create embeddings for multiple documents"""
    try:
        # Validate token lengths
        for text in documents.texts:
            if not validate_token_length(text):
                raise EmbeddingError(detail=f"Text exceeds maximum token length of {MAX_LENGTH}")
        
        # Generate embeddings
        embeddings = get_embeddings(documents.texts)
        if not embeddings:
            raise EmbeddingError(detail="Failed to generate embeddings")
        
        # Store embeddings
        success, message = store_embeddings_batch(
            texts=documents.texts,
            metadata=documents.metadata
        )
        
        if not success:
            raise DatabaseServiceError(detail=message)
        
        return {"status": "success", "message": message}
    except (NomicServiceError, EmbeddingError, DatabaseServiceError) as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error while creating embeddings batch: {str(e)}")
        raise NomicServiceError(detail=f"Unexpected error while creating embeddings batch: {str(e)}")

@app.post("/search")
async def search_documents(query: Query):
    """Search for similar documents"""
    try:
        # Validate token length
        if not validate_token_length(query.text):
            raise EmbeddingError(detail=f"Query text exceeds maximum token length of {MAX_LENGTH}")
        
        # Search for similar documents
        success, results, message = search_similar(
            query_text=query.text,
            limit=query.limit
        )
        
        if not success:
            raise DatabaseServiceError(detail=message)
        
        return {"status": "success", "results": results}
    except (NomicServiceError, EmbeddingError, DatabaseServiceError) as e:
        raise e
    except Exception as e:
        logger.error(f"Unexpected error while searching documents: {str(e)}")
        raise NomicServiceError(detail=f"Unexpected error while searching documents: {str(e)}")

# Document Management Endpoints
@app.patch("/documents/batch")
async def update_documents_batch(batch: BatchDocumentUpdate):
    """Update multiple documents by IDs with enhanced validation and performance metrics"""
    start_time = datetime.now(timezone.utc)
    logger.info(f"Starting batch update for {len(batch.document_ids)} documents")
    
    try:
        # Update documents in database service
        try:
            response = requests.patch(
                f"{DB_SERVICE_URL}/documents/batch",
                json=batch.model_dump(),
                timeout=10
            )
            
            if response.status_code == 404:
                error_detail = response.json().get('detail', {})
                logger.warning(f"No documents found for IDs: {batch.document_ids}")
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "No documents found",
                        "requested_ids": batch.document_ids,
                        "found_ids": error_detail.get('found_ids', [])
                    }
                )
            elif response.status_code != 200:
                error_detail = response.json().get('detail', 'Unknown error')
                logger.error(f"Database service error: {error_detail}")
                raise DatabaseServiceError(f"Database service error: {error_detail}")
            
            # Calculate performance metrics
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            docs_per_second = len(batch.document_ids) / duration if duration > 0 else 0
            
            logger.info(f"Batch update completed in {duration:.2f} seconds ({docs_per_second:.2f} docs/sec)")
            
            result = response.json()
            result['performance'] = {
                "duration_seconds": duration,
                "documents_per_second": docs_per_second,
                "total_documents": len(batch.document_ids)
            }
            return result
            
        except requests.exceptions.Timeout:
            error_msg = "Database service request timed out"
            logger.error(error_msg)
            raise DatabaseServiceError(error_msg)
        except requests.exceptions.ConnectionError:
            error_msg = "Failed to connect to database service"
            logger.error(error_msg)
            raise DatabaseServiceError(error_msg)
            
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise
    except NomicServiceError as e:
        logger.error(f"Nomic service error: {str(e)}")
        raise
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        raise NomicServiceError(error_msg)

@app.delete("/documents/batch")
async def delete_documents_batch(batch: BatchDocumentDelete):
    """Delete multiple documents by IDs with enhanced validation and performance metrics"""
    start_time = datetime.now(timezone.utc)
    logger.info(f"Starting batch delete for {len(batch.document_ids)} documents")
    
    try:
        # Delete documents from database service
        try:
            response = requests.delete(
                f"{DB_SERVICE_URL}/documents/batch",
                json=batch.model_dump(),
                timeout=10
            )
            
            if response.status_code == 404:
                error_detail = response.json().get('detail', {})
                logger.warning(f"No documents found for IDs: {batch.document_ids}")
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "No documents found",
                        "requested_ids": batch.document_ids,
                        "found_ids": error_detail.get('found_ids', [])
                    }
                )
            elif response.status_code != 200:
                error_detail = response.json().get('detail', 'Unknown error')
                logger.error(f"Database service error: {error_detail}")
                raise DatabaseServiceError(f"Database service error: {error_detail}")
            
            # Calculate performance metrics
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()
            docs_per_second = len(batch.document_ids) / duration if duration > 0 else 0
            
            logger.info(f"Batch delete completed in {duration:.2f} seconds ({docs_per_second:.2f} docs/sec)")
            
            result = response.json()
            result['performance'] = {
                "duration_seconds": duration,
                "documents_per_second": docs_per_second,
                "total_documents": len(batch.document_ids)
            }
            return result
            
        except requests.exceptions.Timeout:
            error_msg = "Database service request timed out"
            logger.error(error_msg)
            raise DatabaseServiceError(error_msg)
        except requests.exceptions.ConnectionError:
            error_msg = "Failed to connect to database service"
            logger.error(error_msg)
            raise DatabaseServiceError(error_msg)
            
    except ValidationError as e:
        logger.error(f"Validation error: {str(e)}")
        raise
    except NomicServiceError as e:
        logger.error(f"Nomic service error: {str(e)}")
        raise
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        raise NomicServiceError(error_msg)

@app.get("/documents/{document_id}")
async def get_document(document_id: int):
    """Get a document by ID"""
    try:
        # Get document from database service
        try:
            response = requests.get(
                f"{DB_SERVICE_URL}/documents/{document_id}",
                timeout=5
            )
            
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Document not found")
            elif response.status_code != 200:
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

@app.patch("/documents/{document_id}")
async def update_document(document_id: int, update: DocumentUpdate):
    """Update a document by ID"""
    try:
        # Update document in database service
        try:
            response = requests.patch(
                f"{DB_SERVICE_URL}/documents/{document_id}",
                json=update.model_dump(),
                timeout=5
            )
            
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Document not found")
            elif response.status_code != 200:
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

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int):
    """Delete a document by ID"""
    try:
        # Delete document from database service
        try:
            response = requests.delete(
                f"{DB_SERVICE_URL}/documents/{document_id}",
                timeout=5
            )
            
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="Document not found")
            elif response.status_code != 200:
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

@app.get("/documents")
async def list_documents(skip: int = 0, limit: int = 10):
    """List documents with pagination"""
    try:
        response = requests.get(
            f"{DB_SERVICE_URL}/documents",
            params={"skip": skip, "limit": limit}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to list documents: {str(e)}")
        raise DatabaseServiceError(detail=f"Failed to list documents: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while listing documents: {str(e)}")
        raise NomicServiceError(detail=f"Unexpected error while listing documents: {str(e)}") 