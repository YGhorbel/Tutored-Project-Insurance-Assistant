"""Pydantic models for API"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class QueryRequest(BaseModel):
    """Request model for RAG query"""
    question: str = Field(..., description="Natural language question")
    top_k: int = Field(5, description="Number of results to return", ge=1, le=20)
    include_metadata: bool = Field(True, description="Include chunk metadata")
    min_score: float = Field(0.0, description="Minimum similarity score", ge=0.0, le=1.0)


class ChunkResult(BaseModel):
    """Single chunk result"""
    chunk_id: str
    text: str
    summary: Optional[str] = None
    score: float
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    """Response model for RAG query"""
    question: str
    answer: str
    chunks: List[ChunkResult]
    total_chunks_searched: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    index_loaded: bool
    index_size: int
    model_loaded: bool