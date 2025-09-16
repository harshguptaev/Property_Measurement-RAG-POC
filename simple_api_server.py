#!/usr/bin/env python3
"""
Simplified FastAPI backend for the Property Measurement RAG System.
This version provides basic API endpoints and will attempt to connect to RAG when available.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Property Measurement RAG API",
    description="REST API for property document analysis and question answering",
    version="1.0.0"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG agent instance (will be None until properly initialized)
rag_agent: Optional[Any] = None
rag_available = False

# Request/Response models
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class QueryRequest(BaseModel):
    question: str
    conversation_history: Optional[List[ChatMessage]] = []

class QueryResponse(BaseModel):
    response: str
    sources: Optional[List[Dict[str, Any]]] = []
    images: Optional[List[Dict[str, Any]]] = []
    metadata: Optional[Dict[str, Any]] = {}

class SystemStatusResponse(BaseModel):
    status: str
    documents_loaded: int
    vector_store_active: bool
    backend_version: str
    rag_available: bool

# Try to initialize RAG system
def try_initialize_rag():
    """Try to initialize the RAG system, but don't fail if dependencies are missing."""
    global rag_agent, rag_available
    
    try:
        # Add src to Python path
        src_path = Path(__file__).parent / "src"
        if src_path.exists():
            sys.path.insert(0, str(src_path))
        
        # Try to import RAG components
        from src.config import config
        from src.agent import AgenticRAG
        from src.bedrock_client import create_bedrock_llm, create_bedrock_embeddings
        from src.vector_store import VectorStoreManager
        
        logger.info("RAG dependencies found, attempting to initialize...")
        
        # Setup basic AWS environment if not already set
        if not os.getenv('AWS_DEFAULT_REGION'):
            os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
        
        # Check if vector store already exists
        vector_store_path = Path("vectorstore_faiss")
        has_existing_index = vector_store_path.exists() and any(vector_store_path.glob("*.faiss"))
        
        if has_existing_index:
            logger.info("Found existing vector store, attempting to load...")
            
            try:
                bedrock_config = config.get_bedrock_config()
                vector_config = config.get_vector_store_config()
                embeddings = create_bedrock_embeddings(bedrock_config)
                
                vector_store_manager = VectorStoreManager(
                    store_type=vector_config["store_type"],
                    collection_name=vector_config["collection_name"],
                    embeddings=embeddings
                )
                
                # Create RAG agent with existing vector store
                llm = create_bedrock_llm(bedrock_config)
                rag_agent = AgenticRAG(
                    vector_store_manager=vector_store_manager,
                    llm=llm
                )
                
                doc_count = vector_store_manager.get_count()
                logger.info(f"Successfully loaded RAG system with {doc_count} documents")
                rag_available = True
                
            except Exception as e:
                logger.warning(f"Could not initialize RAG system: {e}")
                rag_available = False
        else:
            logger.info("No existing vector store found. RAG system ready for document processing.")
            rag_available = False
            
    except ImportError as e:
        logger.warning(f"RAG dependencies not available: {e}")
        rag_available = False
    except Exception as e:
        logger.warning(f"Error initializing RAG system: {e}")
        rag_available = False

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Property Measurement RAG API",
        "version": "1.0.0",
        "status": "active",
        "rag_available": str(rag_available)
    }

@app.get("/health", response_model=SystemStatusResponse)
async def health_check():
    """Health check endpoint."""
    global rag_agent, rag_available
    
    doc_count = 0
    vector_store_active = False
    
    if rag_agent is not None and rag_available:
        try:
            doc_count = rag_agent.vector_store_manager.get_count()
            vector_store_active = True
        except Exception as e:
            logger.error(f"Error checking vector store: {e}")
    
    status = "healthy" if rag_available else "rag_unavailable"
    
    return SystemStatusResponse(
        status=status,
        documents_loaded=doc_count,
        vector_store_active=vector_store_active,
        backend_version="1.0.0",
        rag_available=rag_available
    )

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system with a question."""
    global rag_agent, rag_available
    
    if not rag_available or rag_agent is None:
        # Provide a demo response when RAG is not available
        demo_response = f"""
I understand you're asking: "{request.question}"

Currently, the RAG system is not fully initialized. This could be because:
1. Documents haven't been processed yet
2. AWS credentials need to be configured  
3. Vector store needs to be set up

**Demo Response:** This is a demonstration of the Property Measurement RAG system. In a fully configured system, I would analyze your property documents to provide detailed answers about:

• Roof condition and materials
• Structural assessments
• Measurement data and dimensions  
• Repair recommendations
• Cost estimates
• Risk assessments

To fully activate the system, please:
1. Configure AWS credentials
2. Process your property documents
3. Ensure all dependencies are installed

Would you like help setting up the system?
"""
        
        return QueryResponse(
            response=demo_response,
            sources=[],
            images=[],
            metadata={"demo_mode": True, "rag_available": False}
        )
    
    try:
        logger.info(f"Processing query: {request.question}")
        
        # Run the RAG agent
        result = rag_agent.run(request.question)
        
        # Handle response format
        if isinstance(result, dict):
            response_text = result.get("response", "")
            sources = result.get("sources", [])
            images = result.get("images", [])
            metadata = result.get("metadata", {})
        else:
            # Fallback for string response
            response_text = str(result)
            sources = []
            images = []
            metadata = {}
        
        logger.info(f"Query completed successfully")
        
        return QueryResponse(
            response=response_text,
            sources=sources,
            images=images,
            metadata=metadata
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/stats")
async def get_document_stats():
    """Get statistics about loaded documents."""
    global rag_agent, rag_available
    
    if not rag_available or rag_agent is None:
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "vector_store_type": "none",
            "status": "rag_not_available"
        }
    
    try:
        doc_count = rag_agent.vector_store_manager.get_count()
        
        return {
            "total_documents": doc_count,
            "total_chunks": doc_count,
            "vector_store_type": "faiss",
            "status": "active"
        }
        
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "vector_store_type": "error",
            "status": f"error: {str(e)}"
        }

@app.get("/documents/search")
async def search_documents(query: str, k: int = 5):
    """Search documents for relevant content."""
    global rag_agent, rag_available
    
    if not rag_available or rag_agent is None:
        return {
            "query": query,
            "results": [],
            "total_found": 0,
            "message": "RAG system not available. This is a demo endpoint."
        }
    
    try:
        # Use vector store manager to search
        docs = rag_agent.vector_store_manager.similarity_search(query, k=k)
        
        results = []
        for doc in docs:
            results.append({
                "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                "metadata": doc.metadata,
                "score": getattr(doc, 'score', None)
            })
        
        return {
            "query": query,
            "results": results,
            "total_found": len(results)
        }
        
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup."""
    logger.info("Starting Property Measurement RAG API...")
    try:
        try_initialize_rag()
        logger.info("API startup completed")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # Don't fail startup - allow API to run in demo mode

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "simple_api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )