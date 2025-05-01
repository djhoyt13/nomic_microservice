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
import logging
from datetime import datetime, timezone
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
    embedding: List[float] = Field(..., min_items=768, max_items=768)
    metadata: Optional[dict] = None

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
            return {}
        return dict(v)

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    timestamp: str

class BatchDocument(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=1000)
    embeddings: List[List[float]] = Field(..., min_items=1, max_items=1000)
    metadata: Optional[dict] = None

    @field_validator('embeddings')
    def validate_embeddings(cls, v, info):
        if 'texts' in info.data and len(v) != len(info.data['texts']):
            raise ValueError("Number of embeddings must match number of texts")
        for emb in v:
            if len(emb) != 768:
                raise ValueError("Each embedding must be 768-dimensional")
        return v

class DocumentUpdate(BaseModel):
    text: Optional[str] = None
    metadata: Optional[dict] = None

class BatchDocumentUpdate(BaseModel):
    document_ids: List[int] = Field(..., min_items=1, max_items=1000, description="List of document IDs to update")
    text: Optional[str] = Field(None, min_length=1, max_length=10000, description="New text content")
    metadata: Optional[dict] = Field(None, description="New metadata to update")

    @field_validator('document_ids')
    @classmethod
    def validate_document_ids(cls, v):
        if not v:
            raise ValueError("At least one document ID is required")
        if len(v) > 1000:
            raise ValueError("Cannot update more than 1000 documents at once")
        if not all(isinstance(x, int) and x > 0 for x in v):
            raise ValueError("All document IDs must be positive integers")
        return v

class BatchDocumentDelete(BaseModel):
    document_ids: List[int] = Field(..., min_items=1, max_items=1000, description="List of document IDs to delete")

    @field_validator('document_ids')
    @classmethod
    def validate_document_ids(cls, v):
        if not v:
            raise ValueError("At least one document ID is required")
        if len(v) > 1000:
            raise ValueError("Cannot delete more than 1000 documents at once")
        if not all(isinstance(x, int) and x > 0 for x in v):
            raise ValueError("All document IDs must be positive integers")
        return v

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown"""
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

# Create FastAPI app
app = FastAPI(
    title="Document Embedding Service",
    description="Service for storing and searching document embeddings",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
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

# Basic route for testing
@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint for service health check"""
    return {"message": "Document Embedding Service is running"}

# Error handlers
@app.exception_handler(ServiceError)
async def service_error_handler(request, exc: ServiceError) -> JSONResponse:
    """Handle service-level errors"""
    logger.error(f"Service error: {str(exc)}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error=str(exc),
            timestamp=datetime.now(timezone.utc).isoformat()
        ).model_dump()
    )

@app.exception_handler(DatabaseError)
async def database_error_handler(request, exc: DatabaseError) -> JSONResponse:
    """Handle database-related errors"""
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
async def validation_error_handler(request, exc: ValidationError) -> JSONResponse:
    """Handle validation errors"""
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
async def http_exception_handler(request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions"""
    logger.error(f"HTTP error: {str(exc)}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.now(timezone.utc).isoformat()
        ).model_dump()
    )

@app.post("/store", response_model=Dict[str, Any])
async def store_document(document: Document, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Store a document with its embedding"""
    try:
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
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error while storing document: {str(e)}")
        raise DatabaseError(f"Failed to store document: {str(e)}")
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error while storing document: {str(e)}")
        raise ServiceError(f"Unexpected error while storing document: {str(e)}")

@app.post("/search", response_model=List[SearchResult])
async def search_similar(query: Query, db: Session = Depends(get_db)) -> List[SearchResult]:
    """Search for similar documents based on embedding"""
    try:
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
            result = SearchResult(
                id=doc.id,
                text=doc.text,
                metadata=doc.doc_metadata if doc.doc_metadata is not None else {},
                similarity=float(similarity),
                created_at=doc.created_at.isoformat()
            )
            formatted_results.append(result)
        
        return formatted_results
    except SQLAlchemyError as e:
        logger.error(f"Database error while searching: {str(e)}")
        raise DatabaseError(f"Failed to search documents: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while searching: {str(e)}")
        raise ServiceError(f"Unexpected error while searching: {str(e)}")

@app.post("/store_batch", response_model=List[Dict[str, Any]])
async def store_documents_batch(batch: BatchDocument, db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """Store multiple documents with their embeddings"""
    try:
        results = []
        # Store in database
        for text, embedding in zip(batch.texts, batch.embeddings):
            db_embedding = DocumentEmbedding(
                text=text,
                embedding=embedding,
                metadata=batch.metadata
            )
            db.add(db_embedding)
            db.flush()  # Flush to get the ID and created_at
            results.append({
                "id": db_embedding.id,
                "status": "success",
                "created_at": db_embedding.created_at.isoformat()
            })
        
        db.commit()
        return results
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error while storing documents: {str(e)}")
        raise DatabaseError(f"Failed to store documents: {str(e)}")
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error while storing documents: {str(e)}")
        raise ServiceError(f"Unexpected error while storing documents: {str(e)}")

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Check the health of the service and database connection"""
    try:
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

@app.delete("/documents/batch")
async def delete_documents_batch(batch: BatchDocumentDelete, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Delete multiple documents by IDs with enhanced validation and performance metrics"""
    start_time = datetime.now(timezone.utc)
    logger.info(f"Starting batch delete for {len(batch.document_ids)} documents")
    
    try:
        # Get documents
        documents = db.query(DocumentEmbedding).filter(DocumentEmbedding.id.in_(batch.document_ids)).all()
        if not documents:
            logger.warning(f"No documents found for IDs: {batch.document_ids}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "No documents found",
                    "requested_ids": batch.document_ids,
                    "found_ids": []
                }
            )
        
        # Log found documents
        found_ids = [doc.id for doc in documents]
        logger.info(f"Found {len(found_ids)} documents: {found_ids}")
        
        # Delete documents
        for document in documents:
            db.delete(document)
        db.commit()
        
        # Calculate performance metrics
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        docs_per_second = len(documents) / duration if duration > 0 else 0
        
        logger.info(f"Batch delete completed in {duration:.2f} seconds ({docs_per_second:.2f} docs/sec)")
        
        return {
            "status": "success",
            "message": f"Successfully deleted {len(documents)} documents",
            "deleted_ids": [doc.id for doc in documents],
            "performance": {
                "duration_seconds": duration,
                "documents_per_second": docs_per_second,
                "total_documents": len(documents)
            }
        }
    except SQLAlchemyError as e:
        db.rollback()
        error_msg = f"Database error while deleting documents: {str(e)}"
        logger.error(error_msg)
        raise DatabaseError(error_msg)
    except Exception as e:
        db.rollback()
        error_msg = f"Unexpected error while deleting documents: {str(e)}"
        logger.error(error_msg)
        raise ServiceError(error_msg)

@app.patch("/documents/batch")
async def update_documents_batch(batch: BatchDocumentUpdate, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Update multiple documents by IDs with enhanced validation and performance metrics"""
    start_time = datetime.now(timezone.utc)
    logger.info(f"Starting batch update for {len(batch.document_ids)} documents")
    
    try:
        # Get documents
        documents = db.query(DocumentEmbedding).filter(DocumentEmbedding.id.in_(batch.document_ids)).all()
        if not documents:
            logger.warning(f"No documents found for IDs: {batch.document_ids}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "No documents found",
                    "requested_ids": batch.document_ids,
                    "found_ids": []
                }
            )
        
        # Log found documents
        found_ids = [doc.id for doc in documents]
        logger.info(f"Found {len(found_ids)} documents: {found_ids}")
        
        # Update documents
        updated_docs = []
        for document in documents:
            if batch.text is not None:
                document.text = batch.text
            if batch.metadata is not None:
                document.doc_metadata = batch.metadata
            updated_docs.append(document)
        
        db.commit()
        
        # Calculate performance metrics
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()
        docs_per_second = len(updated_docs) / duration if duration > 0 else 0
        
        logger.info(f"Batch update completed in {duration:.2f} seconds ({docs_per_second:.2f} docs/sec)")
        
        return {
            "status": "success",
            "message": f"Successfully updated {len(updated_docs)} documents",
            "updated_ids": [doc.id for doc in updated_docs],
            "performance": {
                "duration_seconds": duration,
                "documents_per_second": docs_per_second,
                "total_documents": len(updated_docs)
            }
        }
    except SQLAlchemyError as e:
        db.rollback()
        error_msg = f"Database error while updating documents: {str(e)}"
        logger.error(error_msg)
        raise DatabaseError(error_msg)
    except Exception as e:
        db.rollback()
        error_msg = f"Unexpected error while updating documents: {str(e)}"
        logger.error(error_msg)
        raise ServiceError(error_msg)

@app.get("/documents/{document_id}")
async def get_document(document_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Get a document by ID"""
    try:
        # Get document
        document = db.query(DocumentEmbedding).filter(DocumentEmbedding.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return {
            "id": document.id,
            "text": document.text,
            "metadata": document.doc_metadata if document.doc_metadata is not None else {},
            "created_at": document.created_at.isoformat()
        }
    except SQLAlchemyError as e:
        logger.error(f"Database error while retrieving document: {str(e)}")
        raise DatabaseError(f"Failed to retrieve document: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error while retrieving document: {str(e)}")
        raise ServiceError(f"Unexpected error while retrieving document: {str(e)}")

@app.patch("/documents/{document_id}")
async def update_document(document_id: int, update: DocumentUpdate, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Update a document by ID"""
    try:
        # Get document
        document = db.query(DocumentEmbedding).filter(DocumentEmbedding.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Update document
        if update.text is not None:
            document.text = update.text
        if update.metadata is not None:
            document.doc_metadata = update.metadata
        
        db.commit()
        db.refresh(document)
        
        return {
            "id": document.id,
            "text": document.text,
            "metadata": document.doc_metadata if document.doc_metadata is not None else {},
            "created_at": document.created_at.isoformat()
        }
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error while updating document: {str(e)}")
        raise DatabaseError(f"Failed to update document: {str(e)}")
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error while updating document: {str(e)}")
        raise ServiceError(f"Unexpected error while updating document: {str(e)}")

@app.delete("/documents/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Delete a document by ID"""
    try:
        # Get document
        document = db.query(DocumentEmbedding).filter(DocumentEmbedding.id == document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete document
        db.delete(document)
        db.commit()
        
        return {"status": "success", "message": f"Document {document_id} deleted successfully"}
    except SQLAlchemyError as e:
        db.rollback()
        logger.error(f"Database error while deleting document: {str(e)}")
        raise DatabaseError(f"Failed to delete document: {str(e)}")
    except Exception as e:
        db.rollback()
        logger.error(f"Unexpected error while deleting document: {str(e)}")
        raise ServiceError(f"Unexpected error while deleting document: {str(e)}") 