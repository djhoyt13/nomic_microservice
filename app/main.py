from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from typing import List, Optional
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import requests
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timezone

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
    validate_token_length,
    get_embeddings,
    chunk_list,
    store_embeddings_batch,
    initialize_model,
    configure_logging,
    nomic_service_error_handler,
    validation_error_handler,
    request_exception_handler,
    tokenizer,
    model,
    MAX_LENGTH,
    BATCH_SIZE,
    CHUNK_SIZE,
    MAX_BATCH_SIZE,
    DB_SERVICE_URL
)

# Load environment variables
load_dotenv()

# Configure logging
logger = configure_logging()

app = FastAPI(title="Nomic Embedding Service")

# Initialize the model
tokenizer, model = initialize_model()

# Error handlers
app.exception_handler(NomicServiceError)(nomic_service_error_handler)
app.exception_handler(ValidationError)(validation_error_handler)
app.exception_handler(requests.exceptions.RequestException)(request_exception_handler)

# Health and Monitoring Endpoints
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

# Embedding Endpoints   
@app.post("/embed")
async def create_embedding(document: Document):
    try:
        # Validate input
        if not document.text.strip():
            raise ValidationError("Document text cannot be empty or whitespace")
        
        # Validate token length
        if not validate_token_length(document.text):
            raise ValidationError(f"Document text exceeds maximum token length of {MAX_LENGTH}")

        # Generate embedding using Nomic
        try:
            embedding = get_embeddings([document.text])
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")
        
        # Store in database service with enhanced metadata
        try:
            response = requests.post(
                f"{DB_SERVICE_URL}/store",
                json={
                    "text": document.text,
                    "embedding": embedding[0],
                    "metadata": document.get_enhanced_metadata()
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
        total_chunks = (len(batch.texts) + BATCH_SIZE - 1) // BATCH_SIZE
        
        for i in range(0, len(batch.texts), BATCH_SIZE):
            batch_texts = batch.texts[i:i + BATCH_SIZE]
            chunk_index = i // BATCH_SIZE
            
            try:
                # Generate embeddings for the batch
                embeddings = get_embeddings(batch_texts)
                
                # Store embeddings in chunks with enhanced metadata
                batch_results = await store_embeddings_batch(
                    batch_texts, 
                    embeddings, 
                    batch.get_enhanced_metadata(chunk_index, total_chunks)
                )
                results.extend(batch_results)
                
                logger.info(f"Processed batch {chunk_index + 1} of {total_chunks}")
                
            except Exception as e:
                logger.error(f"Failed to process batch {chunk_index + 1}: {str(e)}")
                raise EmbeddingError(f"Failed to process batch {chunk_index + 1}: {str(e)}")
        
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