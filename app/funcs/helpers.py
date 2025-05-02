import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import requests
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime, timezone
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from fastapi import FastAPI, HTTPException, Request
from contextlib import asynccontextmanager
from app.models import NomicServiceError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is not set")

# Logging configuration
def configure_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

# Create logger
logger = configure_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    try:
        init_model()
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise
    yield
    # Shutdown
    logger.info("Shutting down application...")

# Constants
MODEL_NAME = "nomic-ai/nomic-embed-text-v1"
MAX_LENGTH = 8192
BATCH_SIZE = 32
CHUNK_SIZE = 1024
MAX_BATCH_SIZE = 100
DB_SERVICE_URL = os.getenv("DB_SERVICE_URL", "http://db_service:8001")

# Global variables for model and tokenizer
tokenizer = None
model = None

def init_model():
    """Initialize the model and tokenizer"""
    global tokenizer, model
    try:
        # Initialize tokenizer and model with proper error handling
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, model_max_length=MAX_LENGTH)
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            rotary_scaling_factor=2,
            device_map="auto"  # Automatically handle device placement
        )
        
        # Set model to evaluation mode
        model.eval()
        
        # Verify model initialization
        if tokenizer is None or model is None:
            raise ValueError("Failed to initialize model or tokenizer")
            
        logger.info(f"Successfully initialized model and tokenizer: {MODEL_NAME}")
    except Exception as e:
        logger.error(f"Failed to initialize model and tokenizer: {str(e)}")
        raise NomicServiceError(detail=f"Model initialization failed: {str(e)}")

def validate_token_length(text: str) -> bool:
    """Check if text token length is within limits"""
    if not tokenizer:
        init_model()
    try:
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return len(tokens) <= MAX_LENGTH
    except Exception as e:
        logger.error(f"Token validation failed: {str(e)}")
        raise NomicServiceError(detail=f"Token validation failed: {str(e)}")

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts"""
    if not model or not tokenizer:
        init_model()
    
    # Add task instruction prefix
    texts = [f"search_document: {text}" for text in texts]
    
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i + BATCH_SIZE]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.expand(token_embeddings.size()).float()
            batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            # Normalize embeddings
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            batch_embeddings = batch_embeddings.numpy()
            embeddings.extend(batch_embeddings.tolist())
    
    return embeddings

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split a list into chunks of specified size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def store_embeddings_batch(texts: List[str], metadata: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """Store embeddings for a batch of texts"""
    try:
        # Generate embeddings
        embeddings = get_embeddings(texts)
        
        # Prepare request data
        data = {
            "texts": texts,
            "embeddings": embeddings,
            "metadata": metadata or {}
        }
        
        # Send request to database service
        response = requests.post(f"{DB_SERVICE_URL}/store_batch", json=data)
        response.raise_for_status()
        
        return True, "Successfully stored embeddings"
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to store embeddings: {str(e)}")
        return False, f"Failed to store embeddings: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error while storing embeddings: {str(e)}")
        return False, f"Unexpected error while storing embeddings: {str(e)}"

def search_similar(query_text: str, limit: int = 5) -> Tuple[bool, List[Dict[str, Any]], str]:
    """Search for similar documents"""
    try:
        # Add task instruction prefix
        query_text = f"search_query: {query_text}"
        
        # Generate query embedding
        query_embedding = get_embeddings([query_text])[0]
        
        # Prepare request data
        data = {
            "embedding": query_embedding,
            "limit": limit
        }
        
        # Send request to database service
        response = requests.post(f"{DB_SERVICE_URL}/search", json=data)
        response.raise_for_status()
        
        return True, response.json(), "Successfully retrieved similar documents"
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to search similar documents: {str(e)}")
        return False, [], f"Failed to search similar documents: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error while searching similar documents: {str(e)}")
        return False, [], f"Unexpected error while searching similar documents: {str(e)}"

async def nomic_service_error_handler(request, exc: NomicServiceError) -> JSONResponse:
    """Handle NomicServiceError exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": str(exc.detail),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

async def validation_error_handler(request, exc: ValidationError) -> JSONResponse:
    """Handle validation errors."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation error",
            "details": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

async def request_exception_handler(request, exc: requests.exceptions.RequestException) -> JSONResponse:
    """Handle request exceptions."""
    return JSONResponse(
        status_code=503,
        content={
            "error": "Service unavailable",
            "details": str(exc),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    ) 