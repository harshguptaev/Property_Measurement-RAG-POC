#!/usr/bin/env python3
"""
FastAPI backend for the Property Measurement RAG System.
This server provides REST API endpoints for the Assistant UI frontend.
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

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.config import config
from src.index import process_and_index_directory
from src.docling_index import process_and_index_directory_with_docling, DOCLING_AVAILABLE
from src.agent import AgenticRAG
from src.bedrock_client import create_bedrock_llm, create_bedrock_embeddings
from src.vector_store import VectorStoreManager

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

# Global RAG agent instance
rag_agent: Optional[AgenticRAG] = None

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

class DocumentUploadRequest(BaseModel):
    file_path: str
    process_images: bool = True

class SystemStatusResponse(BaseModel):
    status: str
    documents_loaded: int
    vector_store_active: bool
    backend_version: str

# Initialize RAG system
def initialize_rag_system():
    """Initialize the RAG system with existing documents."""
    global rag_agent
    
    try:
        logger.info("Initializing RAG system...")
        
        # Setup basic AWS environment if not already set
        if not os.getenv('AWS_DEFAULT_REGION'):
            os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'
        
        # Check if vector store already exists
        vector_store_path = Path("vectorstore_faiss")
        has_existing_index = vector_store_path.exists() and any(vector_store_path.glob("*.faiss"))
        
        if has_existing_index:
            logger.info("Loading existing vector store...")
            
            # Load existing vector store
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
            
            except Exception as e:
                logger.warning(f"Could not initialize RAG system with AWS: {e}")
                logger.info("RAG system will be available once AWS credentials are configured")
                
        else:
            logger.warning("No existing vector store found. Documents need to be processed first.")
            
    except Exception as e:
        logger.error(f"Error initializing RAG system: {e}")
        logger.info("System will continue without RAG functionality until properly configured")

# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Property Measurement RAG API",
        "version": "1.0.0",
        "status": "active"
    }

@app.get("/health", response_model=SystemStatusResponse)
async def health_check():
    """Health check endpoint."""
    global rag_agent
    
    doc_count = 0
    vector_store_active = False

    if rag_agent is not None:
        try:
            # Support new AgenticRAG API which maintains multiple vector stores
            if hasattr(rag_agent, "vector_stores") and rag_agent.vector_stores:
                vector_store_active = True
                for vs in rag_agent.vector_stores:
                    try:
                        doc_count += vs["store"].get_count()
                    except Exception as e:
                        logger.warning(f"Error getting count from a vector store: {e}")
            else:
                vector_store_active = False
        except Exception as e:
            logger.error(f"Error checking vector store: {e}")

    return SystemStatusResponse(
        status="healthy" if vector_store_active else ("no_documents" if rag_agent is not None else "not_initialized"),
        documents_loaded=doc_count,
        vector_store_active=vector_store_active,
        backend_version="1.0.0"
    )

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the RAG system with a question."""
    global rag_agent
    
    if rag_agent is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized. Please process documents first."
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

@app.post("/process-documents")
async def process_documents(background_tasks: BackgroundTasks):
    """Process documents in the input_files directory."""
    global rag_agent
    
    try:
        input_dir = Path("input_files")
        if not input_dir.exists():
            raise HTTPException(
                status_code=404,
                detail="Input files directory not found"
            )
        
        # Check for documents
        doc_files = [f for f in input_dir.glob("*") if f.is_file() and f.suffix.lower() in ['.pdf', '.txt', '.md', '.docx', '.doc']]
        
        if not doc_files:
            raise HTTPException(
                status_code=400,
                detail="No documents found in input_files directory"
            )
        
        logger.info(f"Processing {len(doc_files)} documents...")
        
        # Process documents in background
        background_tasks.add_task(process_documents_task, str(input_dir))
        
        return {
            "message": f"Started processing {len(doc_files)} documents",
            "status": "processing"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting document processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def process_documents_task(input_dir: str):
    """Background task to process documents."""
    global rag_agent
    
    try:
        logger.info("Starting document processing task...")
        
        # Use Docling processor if available, otherwise fallback to standard
        if DOCLING_AVAILABLE:
            logger.info("Using Docling for advanced PDF parsing...")
            vector_store_manager = process_and_index_directory_with_docling(
                directory_path=input_dir,
                drop_existing=True,
                extract_images=True
            )
        else:
            logger.info("Using standard PDF processor...")
            vector_store_manager = process_and_index_directory(
                directory_path=input_dir,
                drop_existing=True
            )
        
        # Create new RAG agent
        bedrock_config = config.get_bedrock_config()
        llm = create_bedrock_llm(bedrock_config)
        
        rag_agent = AgenticRAG(
            vector_store_manager=vector_store_manager,
            llm=llm
        )
        
        doc_count = vector_store_manager.get_count()
        logger.info(f"Document processing completed! Created {doc_count} document chunks.")
        
    except Exception as e:
        logger.error(f"Error in document processing task: {e}")

@app.get("/documents/stats")
async def get_document_stats():
    """Get statistics about loaded documents."""
    global rag_agent
    
    if rag_agent is None:
        return {
            "total_documents": 0,
            "total_chunks": 0,
            "vector_store_type": "none",
            "status": "not_initialized"
        }
    
    try:
        # Sum counts across all configured vector stores (new AgenticRAG API)
        doc_count = 0
        if hasattr(rag_agent, "vector_stores") and rag_agent.vector_stores:
            for vs in rag_agent.vector_stores:
                try:
                    doc_count += vs["store"].get_count()
                except Exception as e:
                    logger.warning(f"Error getting count from a vector store: {e}")
        vector_config = config.get_vector_store_config()
        
        return {
            "total_documents": doc_count,
            "total_chunks": doc_count,
            "vector_store_type": vector_config.get("store_type", "unknown"),
            "status": "active"
        }
        
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/search")
async def search_documents(query: str, k: int = 5):
    """Search documents for relevant content."""
    global rag_agent
    
    if rag_agent is None:
        raise HTTPException(
            status_code=503,
            detail="RAG system not initialized"
        )
    
    try:
        # Use the first available vector store to search (for simplicity)
        if not hasattr(rag_agent, "vector_stores") or not rag_agent.vector_stores:
            raise HTTPException(status_code=503, detail="No vector stores configured")

        store = rag_agent.vector_stores[0]["store"]
        docs = store.similarity_search(query, k=k)
        
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
        initialize_rag_system()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # Don't fail startup - allow API to run without RAG initially

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )