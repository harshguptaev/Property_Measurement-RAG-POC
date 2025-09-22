#!/usr/bin/env python3
"""
Hierarchical RAG Backend Server for Property Analysis.
Integrates with the hierarchical RAG system built from Final_Chunks data.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, Any, Dict

from fastapi import FastAPI, HTTPException, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add current directory to Python path to import our hierarchical RAG
sys.path.insert(0, str(Path(__file__).parent))

# Import our hierarchical RAG system
# Since the file has a space in the name, we need to import it differently
import importlib.util
spec = importlib.util.spec_from_file_location("hierarchical_rag_working", "hierarchical_rag_working 1.py")
hierarchical_rag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hierarchical_rag_module)

HierarchicalRAG = hierarchical_rag_module.HierarchicalRAG
load_agentic_rag_output = hierarchical_rag_module.load_agentic_rag_output

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Hierarchical RAG Property Analysis API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
hierarchical_rag = None
documents_loaded = 0
initialization_error = None

class QueryRequest(BaseModel):
    # Accept either `query` or `question` from clients
    query: Optional[str] = None
    question: Optional[str] = None
    conversation_history: Optional[list] = Field(default_factory=list)
    level1_limit: Optional[int] = Field(default=2, description="Number of documents to retrieve from Level 1")
    level2_limit: Optional[int] = Field(default=3, description="Number of chunks to retrieve from Level 2")
    show_raw_results: Optional[bool] = Field(default=False, description="Whether to show raw search results")

class QueryResponse(BaseModel):
    response: str
    sources: Optional[list] = Field(default_factory=list)
    confidence: Optional[float] = None
    level1_docs: Optional[list] = Field(default_factory=list, description="Documents found in Level 1 search")
    level2_chunks: Optional[list] = Field(default_factory=list, description="Chunks found in Level 2 search")

class HealthResponse(BaseModel):
    status: str
    documents_loaded: int
    level1_entities: int
    level2_entities: int
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

def initialize_hierarchical_rag():
    """Initialize the hierarchical RAG system."""
    global hierarchical_rag, documents_loaded, initialization_error
    
    try:
        # Check AWS credentials
        if not setup_aws_credentials():
            initialization_error = "AWS credentials not found in environment"
            logger.error(initialization_error)
            return False
        
        logger.info("Loading documents from Final_Chunks...")
        
        # Load documents from Final_Chunks
        documents = load_agentic_rag_output()
        
        if not documents:
            initialization_error = "No documents found in Final_Chunks folder"
            logger.error(initialization_error)
            return False
        
        documents_loaded = len(documents)
        logger.info(f"Loaded {documents_loaded} documents from Final_Chunks")
        
        # Initialize hierarchical RAG system
        logger.info("Initializing hierarchical RAG system...")
        hierarchical_rag = HierarchicalRAG()
        
        # Check if indices already exist
        hierarchical_rag.show_collection_status()
        
        # Check if we need to build indices
        level1_info = hierarchical_rag.milvus_manager.get_collection_info(hierarchical_rag.level1_collection_name)
        level2_info = hierarchical_rag.milvus_manager.get_collection_info(hierarchical_rag.level2_collection_name)
        
        level1_count = level1_info.get('entity_count', 0)
        level2_count = level2_info.get('entity_count', 0)
        
        if level1_count == 0 or level2_count == 0:
            logger.info("Building hierarchical indices...")
            hierarchical_rag.build_level1_index(documents, clear_existing=True)
            hierarchical_rag.build_level2_index(documents, clear_existing=False)
            logger.info("✅ Hierarchical indices built successfully!")
        else:
            logger.info(f"✅ Using existing indices: Level1={level1_count}, Level2={level2_count}")
        
        logger.info("✅ Hierarchical RAG system initialized successfully!")
        return True
        
    except Exception as e:
        initialization_error = f"Failed to initialize hierarchical RAG system: {str(e)}"
        logger.error(initialization_error, exc_info=True)
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize the hierarchical RAG system on startup."""
    logger.info("🚀 Starting Hierarchical RAG Property Analysis API...")
    
    # Try to initialize hierarchical RAG system
    success = initialize_hierarchical_rag()
    if success:
        logger.info("🎉 Hierarchical RAG system ready!")
    else:
        logger.warning("⚠️  Hierarchical RAG system initialization failed, running in limited mode")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    global hierarchical_rag, documents_loaded, initialization_error
    
    level1_entities = 0
    level2_entities = 0
    
    if hierarchical_rag:
        try:
            level1_info = hierarchical_rag.milvus_manager.get_collection_info(hierarchical_rag.level1_collection_name)
            level2_info = hierarchical_rag.milvus_manager.get_collection_info(hierarchical_rag.level2_collection_name)
            level1_entities = level1_info.get('entity_count', 0)
            level2_entities = level2_info.get('entity_count', 0)
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
    
    return HealthResponse(
        status="healthy" if hierarchical_rag else "limited",
        documents_loaded=documents_loaded,
        level1_entities=level1_entities,
        level2_entities=level2_entities,
        backend_version="1.0.0",
        rag_available=hierarchical_rag is not None,
        error_message=initialization_error
    )

@app.post("/query", response_model=QueryResponse)
async def process_query(request: Request, request_data: Optional[QueryRequest] = Body(default=None)):
    """Process a query using the hierarchical RAG system."""
    global hierarchical_rag, initialization_error
    
    if not hierarchical_rag:
        raise HTTPException(
            status_code=503, 
            detail=f"Hierarchical RAG system not available. Error: {initialization_error or 'Unknown error'}"
        )
    
    try:
        # Determine the actual prompt text, supporting multiple payload shapes
        prompt = None
        level1_limit = 2
        level2_limit = 3
        show_raw_results = False
        
        if request_data is not None:
            prompt = (request_data.query or request_data.question)
            level1_limit = request_data.level1_limit or 2
            level2_limit = request_data.level2_limit or 3
            show_raw_results = request_data.show_raw_results or False

        if not prompt:
            try:
                payload = await request.json()
            except Exception:
                payload = None

            if isinstance(payload, dict):
                prompt = payload.get("query") or payload.get("question")
                level1_limit = payload.get("level1_limit", 2)
                level2_limit = payload.get("level2_limit", 3)
                show_raw_results = payload.get("show_raw_results", False)
                
                if not prompt and isinstance(payload.get("messages"), list):
                    # Try to extract last user message content
                    for msg in reversed(payload["messages"]):
                        if isinstance(msg, dict) and msg.get("role") == "user" and msg.get("content"):
                            prompt = msg["content"]
                            break

        # Fallback to query param
        if not prompt:
            prompt = request.query_params.get("q")

        if not prompt:
            msg = "No query provided. Please send a 'query' or 'question' field."
            logger.warning(msg)
            return QueryResponse(response=msg, sources=[], confidence=None)

        logger.info(f"Processing hierarchical query: {prompt[:100]}...")
        
        # Use hierarchical search to get relevant chunks
        search_results = hierarchical_rag.search_hierarchical(prompt, level1_limit, level2_limit)
        
        # Generate LLM response
        llm_response = hierarchical_rag.generate_llm_response(prompt, search_results)
        
        # Format response
        try:
            formatted_response = hierarchical_rag.format_json_response(llm_response)
        except:
            formatted_response = llm_response
        
        # Extract sources from search results
        sources = []
        level1_docs = []
        level2_chunks = []
        
        for result in search_results[:3]:  # Limit to top 3 sources
            # Level 2 chunk info
            chunk_info = {
                "chunk_id": result.get("chunk_id"),
                "section": result.get("section"),
                "chunk_type": result.get("chunk_type"),
                "content": result.get("chunk_text", "")[:200] + "..." if len(result.get("chunk_text", "")) > 200 else result.get("chunk_text", ""),
                "distance": result.get("distance", 0)
            }
            level2_chunks.append(chunk_info)
            
            # Source info for compatibility
            source_info = {
                "content": chunk_info["content"],
                "metadata": {
                    "doc_id": result.get("doc_id"),
                    "address": result.get("doc_address"),
                    "section": result.get("section"),
                    "chunk_type": result.get("chunk_type"),
                    "distance": result.get("distance", 0)
                }
            }
            sources.append(source_info)
            
            # Level 1 doc info (unique docs only)
            doc_info = {
                "doc_id": result.get("doc_id"),
                "address": result.get("doc_address"),
                "summary": result.get("doc_summary")
            }
            if doc_info not in level1_docs:
                level1_docs.append(doc_info)
        
        logger.info("✅ Hierarchical query processed successfully")
        
        return QueryResponse(
            response=formatted_response,
            sources=sources,
            confidence=0.9,  # Higher confidence due to hierarchical approach
            level1_docs=level1_docs,
            level2_chunks=level2_chunks
        )
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error processing hierarchical query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.get("/documents/count")
async def get_document_count():
    """Get the number of documents and chunks in the hierarchical indices."""
    global hierarchical_rag, documents_loaded
    
    if not hierarchical_rag:
        return {"documents": 0, "level1_entities": 0, "level2_entities": 0, "status": "not_available"}
    
    try:
        level1_info = hierarchical_rag.milvus_manager.get_collection_info(hierarchical_rag.level1_collection_name)
        level2_info = hierarchical_rag.milvus_manager.get_collection_info(hierarchical_rag.level2_collection_name)
        
        return {
            "documents": documents_loaded,
            "level1_entities": level1_info.get('entity_count', 0),
            "level2_entities": level2_info.get('entity_count', 0),
            "status": "active"
        }
    except Exception as e:
        logger.error(f"Error getting document count: {e}")
        return {"documents": 0, "level1_entities": 0, "level2_entities": 0, "status": "error"}

@app.get("/documents/stats")
async def get_document_stats():
    """Return basic document statistics for the dashboard."""
    global hierarchical_rag, documents_loaded
    
    try:
        if not hierarchical_rag:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "level1_entities": 0,
                "level2_entities": 0,
                "vector_store_type": "hierarchical_milvus",
                "status": "not_initialized",
            }

        level1_info = hierarchical_rag.milvus_manager.get_collection_info(hierarchical_rag.level1_collection_name)
        level2_info = hierarchical_rag.milvus_manager.get_collection_info(hierarchical_rag.level2_collection_name)
        
        return {
            "total_documents": documents_loaded,
            "total_chunks": level2_info.get('entity_count', 0),
            "level1_entities": level1_info.get('entity_count', 0),
            "level2_entities": level2_info.get('entity_count', 0),
            "vector_store_type": "hierarchical_milvus",
            "status": "active",
        }
    except Exception as e:
        logger.error(f"Error getting document stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections/status")
async def get_collections_status():
    """Get the status of hierarchical collections."""
    global hierarchical_rag
    
    if not hierarchical_rag:
        return {"status": "not_available", "error": "Hierarchical RAG not initialized"}
    
    try:
        collections = hierarchical_rag.milvus_manager.list_collections()
        
        level1_info = hierarchical_rag.milvus_manager.get_collection_info(hierarchical_rag.level1_collection_name)
        level2_info = hierarchical_rag.milvus_manager.get_collection_info(hierarchical_rag.level2_collection_name)
        
        return {
            "status": "active",
            "collections": collections,
            "level1": {
                "name": hierarchical_rag.level1_collection_name,
                "entities": level1_info.get('entity_count', 0),
                "exists": hierarchical_rag.level1_collection_name in collections
            },
            "level2": {
                "name": hierarchical_rag.level2_collection_name,
                "entities": level2_info.get('entity_count', 0),
                "exists": hierarchical_rag.level2_collection_name in collections
            }
        }
    except Exception as e:
        logger.error(f"Error getting collections status: {e}")
        return {"status": "error", "error": str(e)}

@app.post("/rebuild")
async def rebuild_indices():
    """Rebuild the hierarchical indices from Final_Chunks data."""
    global hierarchical_rag, documents_loaded, initialization_error
    
    if not hierarchical_rag:
        raise HTTPException(status_code=503, detail="Hierarchical RAG system not available")
    
    try:
        logger.info("Rebuilding hierarchical indices...")
        
        # Load fresh documents
        documents = load_agentic_rag_output()
        if not documents:
            raise HTTPException(status_code=404, detail="No documents found in Final_Chunks folder")
        
        documents_loaded = len(documents)
        
        # Rebuild indices
        hierarchical_rag.build_level1_index(documents, clear_existing=True)
        hierarchical_rag.build_level2_index(documents, clear_existing=False)
        
        # Get new counts
        level1_info = hierarchical_rag.milvus_manager.get_collection_info(hierarchical_rag.level1_collection_name)
        level2_info = hierarchical_rag.milvus_manager.get_collection_info(hierarchical_rag.level2_collection_name)
        
        logger.info("✅ Hierarchical indices rebuilt successfully!")
        
        return {
            "status": "success",
            "message": "Hierarchical indices rebuilt successfully",
            "documents_loaded": documents_loaded,
            "level1_entities": level1_info.get('entity_count', 0),
            "level2_entities": level2_info.get('entity_count', 0)
        }
        
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Error rebuilding indices: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error rebuilding indices: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information."""
    global hierarchical_rag, documents_loaded
    
    level1_entities = 0
    level2_entities = 0
    
    if hierarchical_rag:
        try:
            level1_info = hierarchical_rag.milvus_manager.get_collection_info(hierarchical_rag.level1_collection_name)
            level2_info = hierarchical_rag.milvus_manager.get_collection_info(hierarchical_rag.level2_collection_name)
            level1_entities = level1_info.get('entity_count', 0)
            level2_entities = level2_info.get('entity_count', 0)
        except:
            pass
    
    return {
        "message": "Hierarchical RAG Property Analysis API",
        "version": "1.0.0",
        "status": "running",
        "rag_available": hierarchical_rag is not None,
        "documents_loaded": documents_loaded,
        "level1_entities": level1_entities,
        "level2_entities": level2_entities,
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "document_count": "/documents/count",
            "document_stats": "/documents/stats",
            "collections_status": "/collections/status",
            "rebuild": "/rebuild"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Different port to avoid conflicts