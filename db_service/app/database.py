from sqlalchemy import create_engine, Column, Integer, String, JSON, event, text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError, OperationalError
import os
from dotenv import load_dotenv
import logging
from typing import Generator, Optional
from contextlib import contextmanager
import time
from datetime import datetime, timezone

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
DB_DIR = os.path.join(os.getcwd(), "db_service", "data")
DB_FILE = os.path.join(DB_DIR, "embeddings.db")

# Create database directory if it doesn't exist
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

DATABASE_URL = f"sqlite:///{DB_FILE}"
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

def create_db_engine():
    """Create database engine with retry logic"""
    try:
        # Create SQLite engine directly
        engine = create_engine(
            DATABASE_URL,
            connect_args={"check_same_thread": False}  # Required for SQLite
        )
        
        # Test the connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            
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
        break
    except DatabaseConnectionError as e:
        if attempt == MAX_RETRIES - 1:
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
    embedding = Column(JSON)  # Store embeddings as JSON
    document_metadata = Column(JSON)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

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
    """Initialize database with error handling"""
    try:
        # Create database directory if it doesn't exist
        db_dir = os.path.dirname(DB_FILE)
        if not os.path.exists(db_dir):
            os.makedirs(db_dir)
            
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
        logger.info("Starting database connection check...")
        # Create database file if it doesn't exist
        if not os.path.exists(DB_FILE):
            logger.info(f"Database file does not exist at {DB_FILE}, creating it...")
            init_db()
            logger.info("Database file created successfully")
            
        logger.info("Testing database connection...")
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).fetchone()
            logger.info(f"Database connection test successful: {result}")
            return True
    except Exception as e:
        logger.error(f"Database connection check failed with error: {str(e)}")
        return False 