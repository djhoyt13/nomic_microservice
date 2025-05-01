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
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration from environment variables
MAX_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "2048"))  # Updated to 2048 tokens
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))  # Optimal batch size
MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "1000"))  # Max batch size per request
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))  # Optimal chunk size
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "51"))  # 10% of 512

app = FastAPI(title="Nomic Embedding Service")

# Database service URL
DB_SERVICE_URL = os.getenv("DB_SERVICE_URL", "http://db_service:8001")

# Initialize the embedding model
logger.info("Loading embedding model...")
model_name = "nomic-ai/nomic-embed-text-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
logger.info("Model loaded successfully")

def validate_token_length(text: str) -> bool:
    """Validate that text length is within model's token limits"""
    tokens = tokenizer.tokenize(text)
    token_count = len(tokens)
    logger.info(f"Token count: {token_count} (max: {MAX_LENGTH})")
    return token_count <= MAX_LENGTH

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using the local model"""
    # Validate token lengths
    for text in texts:
        if not validate_token_length(text):
            raise EmbeddingError(f"Text exceeds maximum token length of {MAX_LENGTH}")
    
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).numpy().tolist()
    return embeddings

def extract_highest_classification_full(text):
    """
    Extracts full classification markings from text, including portion and full markings,
    and returns the highest classification level found (including controls and caveats).
    If no classification markings are found, returns 'unknown'.
    """
    # Classification hierarchy
    hierarchy = {
        "TOP SECRET": 1,
        "TS": 1,
        "SECRET": 2,
        "S": 2,
        "CONFIDENTIAL": 3,
        "C": 3,
        "CUI": 4,
        "UNCLASSIFIED": 5,
        "U": 5,
    }

    # Patterns
    classification_levels = r"(TOP SECRET|SECRET|CONFIDENTIAL|UNCLASSIFIED|TS|S|C|U)"
    control_markings = r"(//(SCI|SAP|NATO|CUI))?"
    caveats = r"((//(NOFORN|ORCON|PROPIN|RSEN|REL TO [A-Z ,]+))*)"

    # Full marking pattern (e.g., TOP SECRET//SCI//NOFORN)
    full_pattern = re.compile(
        rf"\b{classification_levels}{control_markings}{caveats}\b",
        re.IGNORECASE
    )

    # Portion marking pattern (e.g., (TS), (CUI))
    portion_pattern = re.compile(r"\(([TCU]S?|CUI)\)", re.IGNORECASE)

    candidates = []

    # Full matches
    for match in full_pattern.finditer(text):
        full_marking = match.group(0).upper()
        base = match.group(1).upper()
        if base in hierarchy:
            candidates.append((hierarchy[base], full_marking))

    # Portion matches
    for match in portion_pattern.finditer(text):
        portion = match.group(0).upper()
        base = portion.strip("()")
        if base in hierarchy:
            candidates.append((hierarchy[base], portion))

    # Return the one with the highest classification (lowest rank number)
    if candidates:
        candidates.sort(key=lambda x: x[0])  # sort by hierarchy level
        return candidates[0][1]  # return full string (e.g., "TOP SECRET//SCI")
    
    return "unknown"

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
    text: str = Field(..., min_length=1, max_length=MAX_LENGTH)
    metadata: Optional[dict] = None

    def get_enhanced_metadata(self) -> dict:
        """Enhance metadata with processing information"""
        base_metadata = self.metadata or {}
        classification = extract_highest_classification_full(self.text)
        return {
            **base_metadata,
            "processing_info": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": "nomic-embed-text-v1.5",
                "embedding_dimension": 768,
                "source": base_metadata.get("source", "unknown"),
                "chunk_index": 0,
                "total_chunks": 1,
                "document_classification": classification
            }
        }

class BatchDocument(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=MAX_LENGTH)
    metadata: Optional[dict] = None

    def get_enhanced_metadata(self, chunk_index: int, total_chunks: int) -> dict:
        """Enhance metadata with batch processing information"""
        base_metadata = self.metadata or {}
        # Get the highest classification from all texts in the batch
        classifications = [extract_highest_classification_full(text) for text in self.texts]
        # Find the highest classification (lowest number in hierarchy)
        hierarchy = {
            "TOP SECRET": 1, "TS": 1,
            "SECRET": 2, "S": 2,
            "CONFIDENTIAL": 3, "C": 3,
            "CUI": 4,
            "UNCLASSIFIED": 5, "U": 5,
            "unknown": 6
        }
        highest_classification = min(
            classifications,
            key=lambda x: hierarchy.get(x.split("//")[0].strip(), 6)
        )
        
        return {
            **base_metadata,
            "processing_info": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": "nomic-embed-text-v1.5",
                "embedding_dimension": 768,
                "source": base_metadata.get("source", "unknown"),
                "batch_size": len(self.texts),
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "document_classification": highest_classification
            }
        }

class Query(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_LENGTH)
    top_k: Optional[int] = Field(default=5, ge=1, le=100)

class DocumentUpdate(BaseModel):
    text: Optional[str] = Field(None, min_length=1, max_length=MAX_LENGTH)
    metadata: Optional[dict] = None

class BatchDocumentUpdate(BaseModel):
    document_ids: List[int] = Field(..., min_items=1)
    text: Optional[str] = None
    metadata: Optional[dict] = None

class BatchDocumentDelete(BaseModel):
    document_ids: List[int] = Field(..., min_items=1)

def chunk_list(lst: List, size: int, overlap: int = CHUNK_OVERLAP) -> List[List]:
    """Split a list into chunks of specified size with overlap"""
    if not lst:
        return []
    
    # Calculate step size (chunk size minus overlap)
    step = size - overlap
    
    # Ensure step is at least 1 to prevent infinite loops
    step = max(1, step)
    
    # Generate chunks with overlap
    chunks = []
    for i in range(0, len(lst), step):
        chunk = lst[i:i + size]
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
    
    return chunks

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into chunks with token-based boundaries and natural breaks"""
    if not text.strip():
        return []
    
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    token_count = len(tokens)
    logger.info(f"Total tokens in document: {token_count}")
    
    # Find natural break points (paragraphs, sections)
    paragraphs = text.split('\n\n')
    paragraph_tokens = [tokenizer.tokenize(p) for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for para_tokens in paragraph_tokens:
        # If adding this paragraph would exceed chunk size, start a new chunk
        if current_length + len(para_tokens) > chunk_size and current_chunk:
            # Add the current chunk
            chunk_text = tokenizer.convert_tokens_to_string(current_chunk)
            chunks.append(chunk_text)
            logger.info(f"Created chunk with {len(current_chunk)} tokens")
            
            # Start new chunk with overlap
            overlap_tokens = current_chunk[-overlap:] if overlap > 0 else []
            current_chunk = overlap_tokens
            current_length = len(overlap_tokens)
            logger.info(f"Overlap tokens: {len(overlap_tokens)}")
        
        # Add paragraph tokens to current chunk
        current_chunk.extend(para_tokens)
        current_length += len(para_tokens)
    
    # Add the last chunk if it exists
    if current_chunk:
        chunk_text = tokenizer.convert_tokens_to_string(current_chunk)
        chunks.append(chunk_text)
        logger.info(f"Created final chunk with {len(current_chunk)} tokens")
    
    logger.info(f"Total chunks created: {len(chunks)}")
    return chunks

async def store_embeddings_batch(texts: List[str], embeddings: List[List[float]], metadata: Optional[dict] = None) -> List[dict]:
    """Store a batch of embeddings in the database service"""
    results = []
    # Get chunks for each text
    text_chunks = chunk_list(texts)
    
    # Process each text's chunks
    for chunks in text_chunks:
        # Generate embeddings for the chunks
        chunk_embeddings = get_embeddings(chunks)
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        
        # Store each chunk with its embedding
        for chunk_text, chunk_embedding in zip(chunks, chunk_embeddings):
            try:
                # Enhance metadata for the chunk
                chunk_metadata = {
                    **(metadata or {}),
                    "processing_info": {
                        **(metadata.get("processing_info", {}) if metadata else {}),
                        "chunk_size": len(tokenizer.tokenize(chunk_text)),
                        "total_chunks": len(chunks),
                        "chunk_index": chunks.index(chunk_text),
                        "document_classification": extract_highest_classification_full(chunk_text)
                    }
                }
                
                response = requests.post(
                    f"{DB_SERVICE_URL}/store",
                    json={
                        "text": chunk_text,
                        "embedding": chunk_embedding,
                        "metadata": chunk_metadata
                    },
                    timeout=10
                )
                
                if response.status_code != 200:
                    error_detail = response.json().get('detail', 'Unknown error')
                    logger.error(f"Database service error: {error_detail}")
                    raise DatabaseServiceError(f"Database service error: {error_detail}")
                
                results.append(response.json())
                logger.info(f"Stored chunk {chunks.index(chunk_text) + 1} of {len(chunks)}")
                
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

# Embed a single document
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

# Embed a batch of documents
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

@app.delete("/documents/batch")
async def delete_documents_batch(batch: BatchDocumentDelete):
    """Delete multiple documents by IDs"""
    try:
        # Delete documents from database service
        try:
            response = requests.delete(
                f"{DB_SERVICE_URL}/documents/batch",
                json=batch.model_dump(),
                timeout=5
            )
            
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="No documents found with the provided IDs")
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

@app.patch("/documents/batch")
async def update_documents_batch(batch: BatchDocumentUpdate):
    """Update multiple documents by IDs"""
    try:
        # Update documents in database service
        try:
            response = requests.patch(
                f"{DB_SERVICE_URL}/documents/batch",
                json=batch.model_dump(),
                timeout=5
            )
            
            if response.status_code == 404:
                raise HTTPException(status_code=404, detail="No documents found with the provided IDs")
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