from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, Float, ARRAY, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os
from dotenv import load_dotenv
import logging
from typing import Generator
from contextlib import contextmanager
from datetime import datetime, timezone
import time

# Constants for retry logic
MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
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
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_HOST = os.getenv("DB_HOST", "db")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")

# Create database engine
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

# Define DocumentEmbedding model
class DocumentEmbedding(Base):
    __tablename__ = "document_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    embedding = Column(JSON, nullable=False)  # Store as JSON array
    doc_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), default=datetime.now(timezone.utc), nullable=False)

    def __init__(self, **kwargs):
        if 'metadata' in kwargs:
            metadata = kwargs.pop('metadata')
            if metadata is None:
                kwargs['doc_metadata'] = {}
            elif not isinstance(metadata, dict):
                kwargs['doc_metadata'] = {}
            else:
                kwargs['doc_metadata'] = metadata
        super().__init__(**kwargs)

    def to_dict(self):
        """Convert to dictionary"""
        return {
            'id': self.id,
            'text': self.text,
            'metadata': self.doc_metadata if self.doc_metadata is not None else {},
            'created_at': self.created_at.isoformat()
        }

def init_db():
    """Initialize the database by creating all tables"""
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {str(e)}")
        raise DatabaseError(f"Failed to create database tables: {str(e)}")

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
        logger.info("Starting database connection check...")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).fetchone()
            logger.info(f"Database connection test successful: {result}")
            return True
    except Exception as e:
        logger.error(f"Database connection check failed with error: {str(e)}")
        return False 