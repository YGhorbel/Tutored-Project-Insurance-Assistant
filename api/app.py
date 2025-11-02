"""FastAPI application for RAG queries"""
import os
import sys
from pathlib import Path
import yaml
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
from loguru import logger
import gradio as gr

sys.path.append(str(Path(__file__).parent.parent))

from api.models import QueryRequest, QueryResponse, HealthResponse, ChunkResult
from api.rag_engine import RAGEngine
from utils.logger import setup_logger

logger = setup_logger("api")

# Load configuration
config_path = Path("/app/config/config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# Expand environment variables
def expand_env(obj):
    if isinstance(obj, dict):
        return {k: expand_env(v) for k, v in obj.items()}
    elif isinstance(obj, str) and obj.startswith('${'):
        var_name = obj[2:-1].split(':')[0]
        default = obj[2:-1].split(':')[1] if ':' in obj else None
        return os.getenv(var_name, default)
    return obj

config = expand_env(config)

# Initialize FastAPI
app = FastAPI(
    title="Insurance Regulations RAG API",
    description="Query Tunisian and international insurance regulations",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config['api'].get('cors_origins', ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Metrics
query_counter = Counter('rag_queries_total', 'Total RAG queries')
query_duration = Histogram('rag_query_duration_seconds', 'Query duration')
error_counter = Counter('rag_errors_total', 'Total errors')

# Initialize RAG engine
rag_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize RAG engine on startup"""
    global rag_engine
    try:
        logger.info("Initializing RAG engine...")
        rag_engine = RAGEngine(config)
        logger.info("RAG engine ready")
    except Exception as e:
        logger.error(f"Failed to initialize RAG engine: {e}")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "Insurance Regulations RAG API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    stats = rag_engine.faiss_manager.get_stats()
    
    return HealthResponse(
        status="healthy" if rag_engine.is_ready() else "degraded",
        index_loaded=stats.get('total_vectors', 0) > 0,
        index_size=stats.get('total_vectors', 0),
        model_loaded=rag_engine.embedding_model is not None
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query endpoint"""
    if rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not initialized")
    
    query_counter.inc()
    start_time = time.time()
    
    try:
        # Execute query
        result = rag_engine.query(
            question=request.question,
            top_k=request.top_k,
            min_score=request.min_score
        )
        
        # Format response
        chunks = [
            ChunkResult(**chunk) if request.include_metadata else ChunkResult(
                chunk_id=chunk['chunk_id'],
                text=chunk['text'][:200] + "...",
                score=chunk['score'],
                metadata={}
            )
            for chunk in result['chunks']
        ]
        
        processing_time = (time.time() - start_time) * 1000
        query_duration.observe(processing_time / 1000)
        
        return QueryResponse(
            question=request.question,
            answer=result['answer'],
            chunks=chunks,
            total_chunks_searched=result['total_searched'],
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        error_counter.inc()
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")


# Gradio Interface
def gradio_query(question: str, top_k: int = 5):
    """Gradio query handler"""
    if rag_engine is None:
        return "RAG engine not initialized", []
    
    try:
        result = rag_engine.query(question, top_k=top_k, min_score=0.3)
        
        # Format chunks for display
        chunks_display = []
        for chunk in result['chunks']:
            source = chunk['metadata'].get('source', 'Unknown')
            page = chunk['metadata'].get('page', '?')
            score = chunk['score']
            
            chunks_display.append(
                f"**Source:** {source} (Page {page}) | **Score:** {score:.3f}\n\n"
                f"**Summary:** {chunk.get('summary', 'N/A')}\n\n"
                f"**Text:** {chunk['text'][:300]}...\n\n---\n"
            )
        
        return result['answer'], "\n".join(chunks_display)
    
    except Exception as e:
        return f"Error: {str(e)}", ""


# Create Gradio interface
demo = gr.Interface(
    fn=gradio_query,
    inputs=[
        gr.Textbox(label="Question", placeholder="What are the requirements for life insurance in Tunisia?"),
        gr.Slider(minimum=1, maximum=10, value=5, step=1, label="Number of results")
    ],
    outputs=[
        gr.Textbox(label="Answer", lines=5),
        gr.Markdown(label="Supporting Documents")
    ],
    title="Insurance Regulations RAG",
    description="Ask questions about Tunisian and international insurance regulations",
    examples=[
        ["What are the solvency requirements for insurance companies in Tunisia?", 5],
        ["What are the consumer protection regulations?", 5],
        ["What are the reporting obligations for insurers?", 5]
    ]
)

# Mount Gradio app
app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config['api'].get('host', '0.0.0.0'),
        port=config['api'].get('port', 8000),
        log_level=config['api'].get('log_level', 'info')
    )