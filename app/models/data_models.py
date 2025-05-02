from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

class Document(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    metadata: Optional[Dict[str, Any]] = None

    def get_enhanced_metadata(self) -> Dict[str, Any]:
        """Add timestamp and model info to metadata"""
        base_metadata = self.metadata or {}
        return {
            **base_metadata,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "nomic-bert-2048"
        }

class BatchDocument(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=1000)
    metadata: Optional[Dict[str, Any]] = None

    def get_enhanced_metadata(self, chunk_index: int, total_chunks: int) -> Dict[str, Any]:
        """Add chunk information and timestamp to metadata"""
        base_metadata = self.metadata or {}
        return {
            **base_metadata,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": "nomic-bert-2048"
        }

class Query(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    limit: Optional[int] = Field(default=5, ge=1, le=100)

class DocumentUpdate(BaseModel):
    text: Optional[str] = Field(None, min_length=1, max_length=10000)
    metadata: Optional[Dict[str, Any]] = None

class BatchDocumentUpdate(BaseModel):
    document_ids: List[int] = Field(..., min_items=1, max_items=1000)
    text: Optional[str] = Field(None, min_length=1, max_length=10000)
    metadata: Optional[Dict[str, Any]] = None

class BatchDocumentDelete(BaseModel):
    document_ids: List[int] = Field(..., min_items=1, max_items=1000) 