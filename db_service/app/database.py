from sqlalchemy import create_engine, Column, Integer, String, JSON, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, OperationalError
from pgvector.sqlalchemy import Vector
import os
from dotenv import load_dotenv
import logging
from typing import Generator, Optional
from contextlib import contextmanager
import time
from datetime import datetime

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
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@db:5432/embeddings")
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def create_db_engine():
    """Create database engine with retry logic and connection pooling"""
    try:
        logger.info(f"Attempting to connect to database at {DATABASE_URL}")
        engine = create_engine(
            DATABASE_URL,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,
            pool_pre_ping=True,
            connect_args={
                "connect_timeout": 10,
                "application_name": "db_service"
            }
        )
        
        # Test the connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            
        return engine
    except OperationalError as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        raise DatabaseConnectionError(f"Failed to connect to database: {str(e)}")
    except Exception as e:
        logger.error(f"Unexpected error creating database engine: {str(e)}")
        raise DatabaseError(f"Unexpected error creating database engine: {str(e)}")

# Create SQLAlchemy engine with retry logic
engine = None
for attempt in range(MAX_RETRIES):
    try:
        engine = create_db_engine()
        logger.info("Database engine created successfully")
        break
    except DatabaseConnectionError as e:
        if attempt == MAX_RETRIES - 1:
            logger.error("Failed to create database engine after all retry attempts")
            raise
        logger.warning(f"Connection attempt {attempt + 1} failed, retrying in {RETRY_DELAY} seconds...")
        time.sleep(RETRY_DELAY)

if not engine:
    raise DatabaseConnectionError("Failed to create database engine after multiple attempts")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class DocumentEmbedding(Base):
    __tablename__ = "document_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    text = Column(String, nullable=False)
    embedding = Column(Vector(768))  # Nomic embeddings are 768-dimensional
    document_metadata = Column(JSON)
    created_at = Column(String, default=lambda: datetime.now(datetime.UTC).isoformat())

# Event listeners for connection management
@event.listens_for(engine, "connect")
def connect(dbapi_connection, connection_record):
    logger.info("New database connection established")

@event.listens_for(engine, "checkout")
def checkout(dbapi_connection, connection_record, connection_proxy):
    logger.debug("Database connection checked out from pool")

@event.listens_for(engine, "checkin")
def checkin(dbapi_connection, connection_record):
    logger.debug("Database connection returned to pool")

def init_db():
    """Initialize database with error handling and retry logic"""
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
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Attempting database connection (attempt {attempt + 1}/{MAX_RETRIES})")
            with engine.connect() as conn:
                logger.info("Connection established, executing test query")
                conn.execute(text("SELECT 1"))
                logger.info("Test query executed successfully")
                return True
        except OperationalError as e:
            logger.error(f"Database operational error: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error("All connection attempts failed")
                return False
        except Exception as e:
            logger.error(f"Unexpected error during connection check: {str(e)}")
            if attempt < MAX_RETRIES - 1:
                logger.warning(f"Retrying in {RETRY_DELAY} seconds...")
                time.sleep(RETRY_DELAY)
            else:
                logger.error("All connection attempts failed")
                return False 