#!/usr/bin/env python3
"""
Full RAG Backend Server for Property Analysis.
Integrates with the existing vector store and provides RAG capabilities.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Any, Dict

from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Property Analysis RAG API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
rag_agent = None
vector_store_manager = None
initialization_error = None

class QueryRequest(BaseModel):
    # Accept either `query` or `question` from clients
    query: Optional[str] = None
    question: Optional[str] = None
    conversation_history: Optional[list] = Field(default_factory=list)

class QueryResponse(BaseModel):
    response: str
    sources: Optional[list] = Field(default_factory=list)
    confidence: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    documents_loaded: int
    vector_store_active: bool
    backend_version: str
    rag_available: bool
    error_message: Optional[str] = None

def setup_aws_credentials():
    """Setup AWS credentials if they exist in environment."""
    credentials = {
        'AWS_ACCESS_KEY_ID': os.getenv('AWS_ACCESS_KEY_ID'),
        'AWS_SECRET_ACCESS_KEY': os.getenv('AWS_SECRET_ACCESS_KEY'),
        'AWS_SESSION_TOKEN': os.getenv('AWS_SESSION_TOKEN'),
        'AWS_DEFAULT_REGION': os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
    }
    
    missing = [k for k, v in credentials.items() if not v and k != 'AWS_DEFAULT_REGION']
    if missing:
        logger.warning(f"Missing AWS credentials: {missing}")
        return False
    
    logger.info("AWS credentials found in environment")
    return True

def initialize_rag_system():
    """Initialize the RAG system with vector store and models."""
    global rag_agent, vector_store_manager, initialization_error
    
    try:
        # Check AWS credentials
        if not setup_aws_credentials():
            initialization_error = "AWS credentials not found in environment"
            logger.error(initialization_error)
            return False
        
        # Import required modules
        from src.config import config
        from src.bedrock_client import create_bedrock_embeddings, create_bedrock_llm
        from src.vector_store import VectorStoreManager
        from src.agent import AgenticRAG
        
        logger.info("Loading configuration...")
        
        # Get configuration
        bedrock_config = config.get_bedrock_config()
        vector_config = config.get_vector_store_config()
        
        logger.info(f"Using vector store: {vector_config['store_type']}")
        logger.info(f"Collection: {vector_config['collection_name']}")
        
        # Create embeddings
        logger.info("Creating Bedrock embeddings...")
        embeddings = create_bedrock_embeddings(bedrock_config)
        
        # Create vector store manager
        logger.info("Loading vector store...")
        vector_store_manager = VectorStoreManager(
            store_type=vector_config["store_type"],
            collection_name=vector_config["collection_name"],
            embeddings=embeddings,
            milvus_uri=vector_config.get("milvus_uri"),
            embedding_dim=vector_config.get("embedding_dim", 1536)
        )
        
        # Check if vector store has documents
        count = vector_store_manager.get_count()
        logger.info(f"Vector store loaded with {count} documents")
        
        if count == 0:
            initialization_error = "Vector store is empty. No documents found."
            logger.warning(initialization_error)
            return False
        
        # Create LLM
        logger.info("Creating Bedrock LLM...")
        llm = create_bedrock_llm(bedrock_config)
        
        # Create RAG agent
        logger.info("Creating RAG agent...")
        rag_agent = AgenticRAG(
            vector_store_manager=vector_store_manager,
            llm=llm
        )
        
        logger.info("✅ RAG system initialized successfully!")
        return True
        
    except Exception as e:
        initialization_error = f"Failed to initialize RAG system: {str(e)}"
        logger.error(initialization_error, exc_info=True)
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup."""
    logger.info("🚀 Starting Property Analysis RAG API...")
    
    # Try to initialize RAG system
    success = initialize_rag_system()
    if success:
        logger.info("🎉 RAG system ready!")
    else:
        logger.warning("⚠️  RAG system initialization failed, running in limited mode")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global rag_agent, vector_store_manager, initialization_error
    
    documents_loaded = 0
    if vector_store_manager:
        documents_loaded = vector_store_manager.get_count()
    
    return HealthResponse(
        status="healthy" if rag_agent else "limited",
        documents_loaded=documents_loaded,
        vector_store_active=vector_store_manager is not None,
        backend_version="1.0.0",
        rag_available=rag_agent is not None,
        error_message=initialization_error
    )

@app.post("/query", response_model=QueryResponse)
async def process_query(request: Request, request_data: Optional[QueryRequest] = Body(default=None)):
    """Process a query using the RAG system."""
    global rag_agent, initialization_error
    
    if not rag_agent:
        raise HTTPException(
            status_code=503, 
            detail=f"RAG system not available. Error: {initialization_error or 'Unknown error'}"
        )
    
    try:
        # Determine the actual prompt text, supporting multiple payload shapes
        prompt = None
        if request_data is not None:
            prompt = (request_data.query or request_data.question)

        if not prompt:
            try:
                payload = await request.json()
            except Exception:
                payload = None

            if isinstance(payload, dict):
                prompt = payload.get("query") or payload.get("question")
                if not prompt and isinstance(payload.get("messages"), list):
                    # Try to extract last user message content
                    for msg in reversed(payload["messages"]):
                        if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                            prompt = msg["content"]
                            break

        # Fallback to query param or form data
        if not prompt:
            # URL query param ?q=...
            prompt = request.query_params.get("q")
        if not prompt and request.headers.get("content-type", "").startswith("application/x-www-form-urlencoded"):
            try:
                form = await request.form()
                prompt = form.get("query") or form.get("question")
            except Exception:
                pass

        if not prompt:
            # Gracefully handle missing prompt to avoid frontend 422/500 loops
            msg = "No query provided. Please send a 'query' or 'question' field."
            logger.warning(msg)
            return QueryResponse(response=msg, sources=[], confidence=None)

        logger.info(f"Processing query: {prompt[:100]}...")
        
        # Use the RAG agent to process the query
        response = rag_agent.run(prompt)
        
        # Extract response text and sources
        if isinstance(response, dict):
            response_text = response.get('response', str(response))
            # Handle images if available
            images = response.get('images', [])
        elif hasattr(response, 'response'):
            response_text = response.response
        elif hasattr(response, 'content'):
            response_text = response.content
        elif isinstance(response, str):
            response_text = response
        else:
            response_text = str(response)
        
        # Try to extract sources if available
        sources = []
        images = []
        if hasattr(response, 'source_documents'):
            sources = [
                {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in response.source_documents[:3]  # Limit to top 3 sources
            ]
        elif isinstance(response, dict):
            # If our Agent returns sources/images directly
            sources = response.get('sources', [])
            images = response.get('images', [])
        
        logger.info("✅ Query processed successfully")
        
        return QueryResponse(
            response=response_text,
            sources=sources,
            confidence=0.8  # Default confidence
        )
        
    except HTTPException as e:
        # Propagate intended HTTP errors (e.g., 422)
        raise e
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/documents/count")
async def get_document_count():
    """Get the number of documents in the vector store."""
    global vector_store_manager
    
    if not vector_store_manager:
        return {"count": 0, "status": "vector_store_not_available"}
    
    count = vector_store_manager.get_count()
    return {"count": count, "status": "active"}

@app.get("/documents/stats")
async def get_document_stats():
    """Return basic document statistics for the dashboard."""
    global vector_store_manager
    try:
        if not vector_store_manager:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "vector_store_type": "none",
                "status": "not_initialized",
            }

        count = vector_store_manager.get_count()
        # We don't track chunks separately here; treat count as chunks if using chunked ingest
        from src.config import config as _config
        vector_config = _config.get_vector_store_config()
        return {
            "total_documents": count,
            "total_chunks": count,
            "vector_store_type": vector_config.get("store_type", "unknown"),
            "status": "active",
        }
    except Exception as e:
        logger.error(f"Error getting document stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Property Analysis RAG API",
        "version": "1.0.0",
        "status": "running",
        "rag_available": rag_agent is not None,
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "document_count": "/documents/count"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)