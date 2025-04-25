from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from nomic import embed
from sqlalchemy.orm import Session
from .database import get_db, DocumentEmbedding, init_db
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Nomic Embedding Service")

# Initialize database
@app.on_event("startup")
async def startup_event():
    init_db()

# Data models
class Document(BaseModel):
    text: str
    metadata: Optional[dict] = None

class Query(BaseModel):
    text: str
    top_k: Optional[int] = 5

class SearchResult(BaseModel):
    id: int
    text: str
    metadata: Optional[dict]
    similarity: float

@app.post("/embed")
async def create_embedding(document: Document, db: Session = Depends(get_db)):
    try:
        # Generate embedding using Nomic
        embedding = embed.text(
            texts=[document.text],
            model='nomic-embed-text-v1'
        )
        
        # Store in database
        db_embedding = DocumentEmbedding(
            text=document.text,
            embedding=embedding.tolist()[0],  # Convert to list for PGVector
            metadata=document.metadata
        )
        db.add(db_embedding)
        db.commit()
        db.refresh(db_embedding)
            
        return {"id": db_embedding.id, "status": "success"}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search", response_model=List[SearchResult])
async def search_similar(query: Query, db: Session = Depends(get_db)):
    try:
        # Generate query embedding
        query_embedding = embed.text(
            texts=[query.text],
            model='nomic-embed-text-v1'
        )
        
        # Search using PGVector
        results = db.query(DocumentEmbedding).order_by(
            DocumentEmbedding.embedding.cosine_distance(query_embedding.tolist()[0])
        ).limit(query.top_k).all()
        
        # Format results
        formatted_results = []
        for doc in results:
            similarity = 1 - doc.embedding.cosine_distance(query_embedding.tolist()[0])
            formatted_results.append(SearchResult(
                id=doc.id,
                text=doc.text,
                metadata=doc.metadata,
                similarity=float(similarity)
            ))
        
        return formatted_results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 