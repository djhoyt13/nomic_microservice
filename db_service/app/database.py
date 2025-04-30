from sqlalchemy import create_engine, Column, Integer, String, JSON, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, OperationalError
import os
from dotenv import load_dotenv
import logging
from typing import Generator, Optional, List
from contextlib import contextmanager
import time
from datetime import datetime, timezone
import json
import numpy as np
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Custom exceptions
class DatabaseError(Exception):
    """Base exception for database errors"""
    pass

class DatabaseConnectionError(DatabaseError):
    """Exception raised for database connection issues"""
    pass

class DatabaseInitializationError(DatabaseError):
    """Exception raised for database initialization failures"""
    pass

class DatabaseSessionError(DatabaseError):
    """Exception raised for database session issues"""
    pass

# Database configuration
DB_DIR = os.getenv("DB_DIR", "data")
DB_NAME = os.getenv("DB_NAME", "embeddings.db")
DATABASE_URL = f"sqlite:///{Path(DB_DIR) / DB_NAME}"

# Ensure DB directory exists
os.makedirs(DB_DIR, exist_ok=True)

def create_db_engine():
    """Create SQLite database engine"""
    try:
        logger.info(f"Creating SQLite database at {DATABASE_URL}")
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False}  # Required for SQLite with FastAPI
        )
        
        # Test the connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            
        return engine
    except Exception as e:
        logger.error(f"Failed to create database engine: {str(e)}")
        raise DatabaseConnectionError(f"Failed to create database engine: {str(e)}")

# Create SQLAlchemy engine
engine = create_db_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class DocumentEmbedding(Base):
    __tablename__ = "document_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    embedding = Column(String, nullable=False)  # Store as JSON string
    document_metadata = Column(JSON)
    created_at = Column(String, default=lambda: datetime.now(timezone.utc).isoformat())

    def __init__(self, text: str, embedding: list, metadata: Optional[dict] = None):
        self.text = text
        self.embedding = json.dumps(embedding)  # Convert list to JSON string
        self.document_metadata = metadata

    def get_embedding(self) -> list:
        """Convert stored JSON string back to list"""
        return json.loads(self.embedding)

    def __repr__(self):
        return f"<DocumentEmbedding(id={self.id}, text='{self.text[:50]}...', created_at='{self.created_at}')>"

def init_db():
    """Initialize database"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except SQLAlchemyError as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise DatabaseInitializationError(f"Failed to create database tables: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error during database initialization: {str(e)}")
        raise DatabaseError(f"Unexpected error during database initialization: {str(e)}")

@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for database sessions with error handling"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        session.rollback()
        logger.error(f"Database session error: {str(e)}")
        raise DatabaseSessionError(f"Database session error: {str(e)}")
    except Exception as e:
        session.rollback()
        logger.error(f"Unexpected error in database session: {str(e)}")
        raise DatabaseError(f"Unexpected error in database session: {str(e)}")
    finally:
        session.close()

# Dependency to get DB session for FastAPI
def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions"""
    with get_db_session() as session:
        yield session

def check_db_connection() -> bool:
    """Check if database connection is available"""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.error(f"Database connection check failed: {str(e)}")
        return False

def cosine_similarity(a: list, b: list) -> float:
    """Calculate cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def search_similar_embeddings(query_embedding: list, top_k: int = 5) -> List[dict]:
    """Search for similar documents using cosine similarity"""
    try:
        with get_db_session() as session:
            # Get all documents
            documents = session.query(DocumentEmbedding).all()
            
            # Calculate similarities
            similarities = []
            for doc in documents:
                doc_embedding = doc.get_embedding()
                similarity = cosine_similarity(query_embedding, doc_embedding)
                similarities.append({
                    'id': doc.id,
                    'text': doc.text,
                    'metadata': doc.document_metadata,
                    'similarity': similarity,
                    'created_at': doc.created_at
                })
            
            # Sort by similarity and get top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
    except Exception as e:
        logger.error(f"Error searching similar embeddings: {str(e)}")
        raise DatabaseError(f"Error searching similar embeddings: {str(e)}") 