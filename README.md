# Nomic Microservices

This project consists of two microservices:
1. Database Service - A FastAPI service for storing and searching document embeddings
2. Embedding Service - A FastAPI service for generating embeddings using a local Nomic embedding model

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- PostgreSQL 13+ with pgvector extension

## Environment Setup

1. Create a `.env` file in the root directory with the following variables:

```env
# Database Service
DATABASE_URL=postgresql://postgres:postgres@db:5432/embeddings

# Embedding Service Configuration
MAX_TEXT_LENGTH=2048  # Maximum token length (model's context window)
BATCH_SIZE=32         # Optimal batch size for processing
MAX_BATCH_SIZE=1000   # Maximum number of texts per request
CHUNK_SIZE=512        # Optimal chunk size for long documents
CHUNK_OVERLAP=51      # 10% overlap between chunks (for context preservation)
```

## Optimal Configuration

The service is configured with optimal settings for the Nomic embedding model:

1. Text Processing:
   - Maximum text length: 2048 tokens (model's context window)
   - Optimal chunk size: 512 tokens
   - Chunk overlap: 51 tokens (10% of chunk size)
   - Token-based chunking with natural boundary respect

2. Batch Processing:
   - Optimal batch size: 32 texts
   - Maximum batch size: 1000 texts per request
   - Automatic chunking for long documents
   - Classification marking preservation

3. Performance Considerations:
   - Token-based length validation
   - Efficient memory usage
   - Context preservation through overlap
   - Natural document boundary respect

## PostgreSQL Setup

### Docker Prerequisites

Before running PostgreSQL with Docker, ensure Docker is properly installed and running:

1. **Check Docker Installation**:
   ```bash
   # Windows
   docker --version
   
   # If not installed, download and install Docker Desktop from:
   # https://www.docker.com/products/docker-desktop
   ```

2. **Start Docker Daemon**:
   - **Windows**: 
     - Open Docker Desktop
     - Wait for the whale icon in the system tray to stop animating
     - Verify Docker is running with `docker info`
   
   - **macOS/Linux**:
     ```bash
     # Check Docker service status
     sudo systemctl status docker
     
     # Start Docker if not running
     sudo systemctl start docker
     ```

3. **Verify Docker is Running**:
   ```bash
   docker info
   # Should show Docker system information without errors
   ```

### Option 1: Using Docker (Recommended)

1. Pull the PostgreSQL image with pgvector:
```bash
docker pull ankane/pgvector:latest
```

2. Run PostgreSQL with pgvector:
```bash
docker run --name postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=embeddings -p 5432:5432 -d ankane/pgvector:latest
```

3. Verify the installation:
```bash
docker exec -it postgres psql -U postgres -d embeddings -c "CREATE EXTENSION vector;"
```

### Option 2: Local Installation

1. Install PostgreSQL 13+:
   - **Windows**: Download and install from [PostgreSQL Downloads](https://www.postgresql.org/download/windows/)
   - **macOS**: `brew install postgresql@13`
   - **Linux**: 
     ```bash
     sudo apt update
     sudo apt install postgresql postgresql-contrib
     ```

2. Install pgvector extension:
   ```bash
   # Clone the repository
   git clone https://github.com/pgvector/pgvector.git
   cd pgvector
   
   # Build and install
   make
   make install
   ```

3. Create database and enable extension:
   ```bash
   # Create database
   createdb embeddings
   
   # Connect to database and enable extension
   psql embeddings
   CREATE EXTENSION vector;
   ```

4. Update your `.env` file:
   ```env
   DATABASE_URL=postgresql://postgres:postgres@localhost:5432/embeddings
   ```

### Verifying the Setup

1. Check PostgreSQL is running:
   ```bash
   # Using Docker
   docker ps | grep postgres
   
   # Local installation
   pg_isready
   ```

2. Test the connection:
   ```bash
   psql -U postgres -d embeddings -c "SELECT 1;"
   ```

3. Verify pgvector extension:
   ```bash
   psql -U postgres -d embeddings -c "SELECT * FROM pg_extension WHERE extname = 'vector';"
   ```

### Docker Troubleshooting

1. **Docker Daemon Not Running**:
   - Windows: Open Docker Desktop and wait for it to fully start
   - Check system tray for Docker icon
   - Restart Docker Desktop if needed
   - Run `docker info` to verify daemon is running

2. **Port Conflicts (5432 already in use)**:
   - **Check what's using port 5432**:
     ```bash
     # Windows
     netstat -ano | findstr :5432
     
     # Linux/macOS
     lsof -i :5432
     ```
   
   - **Solutions**:
     1. **Stop the conflicting process**:
        ```bash
        # If it's a PostgreSQL service
        net stop postgresql
        
        # Or if it's another process, use the PID from the previous command
        taskkill /PID <PID> /F
        ```
     
     2. **Use a different port**:
        - Modify the docker-compose.yml file:
          ```yaml
          services:
            db:
              ports:
                - "5433:5432"  # Change first number to an available port
          ```
        - Update DATABASE_URL in .env:
          ```env
          DATABASE_URL=postgresql://postgres:postgres@localhost:5433/embeddings
          ```
     
     3. **Clean up existing containers**:
        ```bash
        # Stop all containers
        docker-compose down
        
        # Remove the specific container causing issues
        docker rm -f nomic_microservice-db-1
        
        # Remove any lingering volumes
        docker volume prune
        ```

3. **Permission Issues**:
   - Windows: Ensure running as administrator
   - Linux: Add user to docker group:
     ```bash
     sudo usermod -aG docker $USER
     # Log out and back in for changes to take effect
     ```

4. **Container Issues**:
   - Check container status:
     ```bash
     docker ps -a
     ```
   - View container logs:
     ```bash
     docker logs postgres
     ```
   - Restart container if needed:
     ```bash
     docker restart postgres
     ```

### Common Port Conflicts and Solutions

1. **PostgreSQL Default Port (5432)**:
   - Common causes:
     - Another PostgreSQL instance running
     - Previous Docker container not properly stopped
     - System service using the port
   
   - Resolution steps:
     1. Identify the process:
        ```bash
        # Windows
        netstat -ano | findstr :5432
        # Note the PID
        tasklist | findstr <PID>
        ```
     
     2. Stop the process:
        ```bash
        # Windows
        taskkill /PID <PID> /F
        
        # Or stop PostgreSQL service
        net stop postgresql
        ```
     
     3. If using Docker, clean up:
        ```bash
        # Stop all containers
        docker-compose down
        
        # Remove the specific container causing issues
        docker rm -f nomic_microservice-db-1
        
        # Remove any lingering volumes
        docker volume prune
        ```

2. **Alternative Port Configuration**:
   If you can't free up port 5432, you can use a different port:
   
   ```yaml
   # docker-compose.yml
   services:
     db:
       ports:
         - "5433:5432"  # Map host port 5433 to container port 5432
   ```
   
   Then update your .env file:
   ```env
   DATABASE_URL=postgresql://postgres:postgres@localhost:5433/embeddings
   ```

## Database Service

The database service provides endpoints for storing and searching document embeddings using PostgreSQL with pgvector.

### Features
- Store document embeddings with metadata
- Search for similar documents using cosine similarity
- Health check endpoint
- Comprehensive error handling
- Connection pooling and retry mechanisms

### API Endpoints

- `POST /store` - Store a document embedding
  ```json
  {
    "text": "document text",
    "embedding": [0.1, 0.2, ...], // 768-dimensional vector
    "metadata": {"key": "value"} // optional
  }
  ```

- `POST /search` - Search for similar documents
  ```json
  {
    "embedding": [0.1, 0.2, ...], // 768-dimensional vector
    "top_k": 5 // optional, defaults to 5
  }
  ```

- `GET /health` - Check service health

### Running the Database Service

1. Using Docker Compose (recommended):
   ```bash
   docker-compose up db_service
   ```

2. Running locally:
   ```bash
   cd db_service
   pip install -r requirements.txt
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## Embedding Service

The embedding service generates embeddings using a local Nomic embedding model.

### Features
- Generate embeddings for text documents
- Batch processing support
- Error handling and retry mechanisms
- Rate limiting

### API Endpoints

- `POST /embed` - Generate embeddings for text
  ```json
  {
    "text": "document text"
  }
  ```

- `POST /embed_batch` - Generate embeddings for multiple texts
  ```json
  {
    "texts": ["text1", "text2", ...]
  }
  ```

### Running the Embedding Service

1. Using Docker Compose (recommended):
   ```bash
   docker-compose up embedding_service
   ```

2. Running locally:
   ```bash
   cd embedding_service
   pip install -r requirements.txt
   uvicorn app.main:app --host 0.0.0.0 --port 8001
   ```

## Docker Compose Setup

The project includes a `docker-compose.yml` file for easy deployment:

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up db_service
docker-compose up embedding_service

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## Development

### Setting up the Development Environment

1. Clone the repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Tests

```bash
# Database Service Tests
cd db_service
pytest

# Embedding Service Tests
cd embedding_service
pytest
```

### Code Style

The project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting

Run formatting and linting:
```bash
# Format code
black .
isort .

# Run linter
flake8
```

## Error Handling

Both services implement comprehensive error handling:

- Database Service:
  - Database connection errors
  - Validation errors
  - Service errors
  - HTTP exceptions

- Embedding Service:
  - API errors
  - Rate limiting
  - Validation errors
  - Service errors

All errors return consistent JSON responses with:
- Error message
- Details (when available)
- Timestamp

## Monitoring and Logging

Both services implement logging with:
- Timestamps
- Log levels
- Context information
- Error details

Logs are available through:
- Docker logs
- Local file system (when running locally)
- Standard output

## Security Considerations

1. API Keys:
   - Store API keys in environment variables
   - Never commit API keys to version control
   - Use .env file for local development

2. Database:
   - Use strong passwords
   - Limit database access
   - Use connection pooling
   - Implement proper error handling

3. API Security:
   - CORS configuration
   - Input validation
   - Rate limiting
   - Error handling

## Troubleshooting

### Common Issues

1. Environment Variables:
   - Check if `.env` file exists and is properly formatted
   - Verify DATABASE_URL is correctly set
   - Ensure environment variables are being loaded

2. Database Connection Issues:
   - Check if PostgreSQL is running
   - Verify port availability
   - Test database connection string
   - Ensure pgvector extension is installed

3. Model Loading Issues:
   - Check if required model dependencies are installed
   - Verify disk space for model downloads
   - Check network connectivity for initial model download

4. Security Best Practices:
   - Store sensitive information in environment variables
   - Never commit sensitive information to version control
   - Use appropriate file permissions
   - Follow security guidelines for production deployments

### Getting Help

For issues not covered in this README:
1. Check the service logs
2. Review the error responses
3. Consult the API documentation
4. Open an issue in the repository
