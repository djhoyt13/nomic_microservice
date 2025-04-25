# Nomic Microservices

This project consists of two microservices:
1. Database Service - A FastAPI service for storing and searching document embeddings
2. Embedding Service - A FastAPI service for generating embeddings using Nomic's embedding model

## Prerequisites

- Python 3.8+
- Docker and Docker Compose
- PostgreSQL 13+ with pgvector extension
- Nomic API key (for the embedding service)

## Environment Setup

1. Create a `.env` file in the root directory with the following variables:

```env
# Database Service
DATABASE_URL=postgresql://postgres:postgres@db:5432/embeddings

# Embedding Service
NOMIC_API_KEY=your_nomic_api_key
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

The embedding service generates embeddings using Nomic's embedding model.

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
docker-compose up

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

1. Database Connection Issues:
   - Check DATABASE_URL configuration
   - Verify PostgreSQL is running
   - Check pgvector extension is installed

2. API Key Issues:
   - Verify NOMIC_API_KEY is set
   - Check API key permissions
   - Monitor rate limits

3. Service Health:
   - Use health check endpoints
   - Check service logs
   - Monitor resource usage

### Getting Help

For issues not covered in this README:
1. Check the service logs
2. Review the error responses
3. Consult the API documentation
4. Open an issue in the repository
