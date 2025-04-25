# Nomic Embedding Microservice

A containerized microservice that provides document embedding and similarity search capabilities using the Nomic embedding model and PostgreSQL with PGVector for persistent storage.

## Features

- Document embedding generation using Nomic
- Similarity search using PGVector
- Persistent storage in PostgreSQL
- Metadata support
- RESTful API interface

## Prerequisites

- Docker and Docker Compose
- Nomic API key (set as environment variable `NOMIC_API_KEY`)

## Running the Service

1. Create a `.env` file in the project root with your Nomic API key:
```bash
NOMIC_API_KEY=your_api_key_here
```

2. Start the services using Docker Compose:
```bash
docker-compose up --build
```

The service will be available at `http://localhost:8000`

## API Endpoints

### Embed Documents
- **POST** `/embed`
- Request body:
```json
{
    "text": "Your document text here",
    "metadata": {
        "source": "optional metadata"
    }
}
```

### Search Similar Documents
- **POST** `/search`
- Request body:
```json
{
    "text": "Your search query",
    "top_k": 5
}
```

### Health Check
- **GET** `/health`

## Example Usage

```python
import requests

# Embed a document
response = requests.post(
    "http://localhost:8000/embed",
    json={
        "text": "This is a sample document",
        "metadata": {"source": "test"}
    }
)
print(response.json())

# Search for similar documents
response = requests.post(
    "http://localhost:8000/search",
    json={
        "text": "Find similar documents",
        "top_k": 3
    }
)
print(response.json())
```

## Database Access

The PostgreSQL database is accessible at:
- Host: localhost
- Port: 5432
- Database: embeddings
- Username: postgres
- Password: postgres

## Development

To stop the services:
```bash
docker-compose down
```

To remove all data (including the database volume):
```bash
docker-compose down -v
```

## Architecture

The service consists of:
1. FastAPI application for handling HTTP requests
2. Nomic embedding model for generating embeddings
3. PostgreSQL with PGVector extension for vector storage and similarity search
4. Docker containers for easy deployment

## Error Handling

The service includes proper error handling for:
- Database connection issues
- Invalid input data
- Nomic API errors
- Vector search failures
