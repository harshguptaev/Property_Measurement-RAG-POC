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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Add current directory to Python path to import our hierarchical RAG
sys.path.insert(0, str(Path(__file__).parent))

# Ensure we're using the venv_rag virtual environment
venv_path = Path("venv_rag")
if venv_path.exists():
    # Add venv bin to PATH
    venv_bin = venv_path / "bin"
    if str(venv_bin) not in os.environ.get("PATH", ""):
        os.environ["PATH"] = str(venv_bin) + ":" + os.environ.get("PATH", "")
    print(f"✅ Using virtual environment: {venv_bin}")
else:
    print("⚠️  venv_rag not found, using system Python")

# Import our hierarchical RAG system
# Since the file has a space in the name, we need to import it differently
import importlib.util
spec = importlib.util.spec_from_file_location("hierarchical_rag_working", "hierarchical_rag_working 1.py")
hierarchical_rag_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hierarchical_rag_module)

HierarchicalRAG = hierarchical_rag_module.HierarchicalRAG
load_agentic_rag_output = hierarchical_rag_module.load_agentic_rag_output
QueryAnalysis = hierarchical_rag_module.QueryAnalysis

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

# Mount static files for serving images
extracted_images_path = Path("extracted_images")
if extracted_images_path.exists():
    app.mount("/images", StaticFiles(directory=str(extracted_images_path)), name="images")

# Mount input_data folder for DDD diagrams and other input images
input_data_path = Path("input_data")
if input_data_path.exists():
    app.mount("/input_data_images", StaticFiles(directory=str(input_data_path)), name="input_data_images")

# Mount final_data folder for generated roof overlays and analysis results
final_data_path = Path("final_data")
if final_data_path.exists():
    app.mount("/final_data_images", StaticFiles(directory=str(final_data_path)), name="final_data_images")

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
    level2_limit: Optional[int] = Field(default=5, description="Number of chunks to retrieve from Level 2")
    show_raw_results: Optional[bool] = Field(default=False, description="Whether to show raw search results")

class QueryResponse(BaseModel):
    response: str
    sources: Optional[list] = Field(default_factory=list)
    confidence: Optional[float] = None
    level1_docs: Optional[list] = Field(default_factory=list, description="Documents found in Level 1 search")
    level2_chunks: Optional[list] = Field(default_factory=list, description="Chunks found in Level 2 search")
    roof_pitch_data: Optional[list] = Field(default_factory=list, description="Structured roof pitch data for rich display")
    measurement_data: Optional[list] = Field(default_factory=list, description="Structured measurement data for rich display")
    measurement_type: Optional[str] = Field(default=None, description="Type of measurement data (lengths, rafters, area, azimuth)")

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

@app.get("/pdf/{file_path:path}")
async def serve_pdf(file_path: str):
    """Serve PDF files from final_data directory."""
    try:
        # Construct the full path
        full_path = Path("final_data") / file_path
        
        # Security check: ensure the path is within final_data directory
        if not str(full_path.resolve()).startswith(str(Path("final_data").resolve())):
            raise HTTPException(status_code=403, detail="Access denied")
        
        if not full_path.exists():
            raise HTTPException(status_code=404, detail="PDF file not found")
        
        if not full_path.suffix.lower() == '.pdf':
            raise HTTPException(status_code=400, detail="Not a PDF file")
        
        return FileResponse(
            path=str(full_path),
            media_type="application/pdf",
            filename=full_path.name
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving PDF {file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving PDF: {str(e)}")

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
        level2_limit = 5
        show_raw_results = False
        
        if request_data is not None:
            prompt = (request_data.query or request_data.question)
            level1_limit = request_data.level1_limit or 2
            level2_limit = request_data.level2_limit or 7
            show_raw_results = request_data.show_raw_results or False

        if not prompt:
            try:
                payload = await request.json()
            except Exception:
                payload = None

            if isinstance(payload, dict):
                prompt = payload.get("query") or payload.get("question")
                level1_limit = payload.get("level1_limit", 2)
                level2_limit = payload.get("level2_limit", 7)
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

        # Use the intelligent query routing system (detects property-specific vs general queries)
        # This handles both search and LLM response generation, and returns raw results for source extraction
        llm_response, search_results, query_analysis = hierarchical_rag.answer_query_with_raw_results(prompt, level1_limit, level2_limit)
        
        # Format response
        try:
            formatted_response = hierarchical_rag.format_json_response(llm_response)
        except:
            formatted_response = llm_response
        
       
        sources = []
        level1_docs = []
        level2_chunks = []

        # Separate text and image chunks for better processing
        text_chunks = [r for r in search_results if r.get("chunk_type") != "image"]
        image_chunks = [r for r in search_results if r.get("chunk_type") == "image"]

        # Process text chunks (limit to top results for sources)
        for result in text_chunks[:3]:
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
                    "property_id": result.get("property_id"),
                    "address": result.get("doc_address"),
                    "section": result.get("section"),
                    "chunk_type": result.get("chunk_type"),
                    "distance": result.get("distance", 0)
                }
            }
            sources.append(source_info)

        # Process all PDF chunks
        pdf_chunks = [r for r in search_results if r.get("chunk_type") == "pdf" or r.get("type") == "pdf"]
        for result in pdf_chunks:
            chunk_text = result.get("chunk_text", "")
            pdf_path = result.get("pdf_path", "")
            pdf_filename = result.get("pdf_filename", "")
            
            # Extract PDF path from chunk_text if not directly available
            if not pdf_path and 'pdf_file:' in chunk_text:
                lines = chunk_text.split('\n')
                for line in lines:
                    if line.startswith('pdf_file:'):
                        pdf_path = line.split('pdf_file:')[1].strip()
                        break
            
            if pdf_path:
                # Level 2 chunk info for PDF
                chunk_info = {
                    "chunk_id": result.get("chunk_id"),
                    "section": result.get("section", "PDF Report"),
                    "chunk_type": "pdf",
                    "content": result.get("chunk_text", "PDF Report available"),
                    "distance": result.get("distance", 1.0),
                    "pdf_path": pdf_path,
                    "pdf_filename": pdf_filename or pdf_path.split('/')[-1],
                    "doc_address": result.get("doc_address", "Unknown Address"),
                    "description": result.get("data", {}).get("description", "Property analysis report")
                }
                level2_chunks.append(chunk_info)
                sources.append(chunk_info)
                logger.info(f"Added PDF chunk: {pdf_path}")

        # Process all image chunks (important for frontend display)
        for result in image_chunks:
            # Extract image information from chunk text
            chunk_text = result.get("chunk_text", "")
            image_paths = []
            description = ""

            # Debug log for Property Imagery chunks
            if result.get("section") == "Property Imagery":
                logger.info(f"Processing Property Imagery chunk: {result.get('chunk_id')}")
                logger.info(f"Property address: {result.get('doc_address')}")
                logger.info(f"Chunk text: {chunk_text[:200]}...")

            # Parse image information from chunk text - handle multiple formats
            lines = chunk_text.split('\n')
            for line in lines:
                # Look for image paths in different formats
                if 'extracted_images/' in line:
                    # Extract path using regex to handle different formats
                    import re
                    matches = re.findall(r'extracted_images/[^,\s\]]+\.png', line)
                    image_paths.extend(matches)
                
                # Also look for input_data/ paths (for DDD diagrams)
                if 'input_data/' in line:
                    import re
                    matches = re.findall(r'input_data/[^,\s\]]+\.png', line)
                    image_paths.extend(matches)
                
                # Also look for final_data/ paths (for generated roof overlays)
                if 'final_data/' in line:
                    import re
                    matches = re.findall(r'final_data/[^,\s\]]+\.png', line)
                    image_paths.extend(matches)

                if line.startswith('description:'):
                    description = line.split('description:')[1].strip()

            # If we found image paths, create entries for each
            if image_paths:
                for image_path in image_paths:
                    # Create user-friendly image title
                    filename = image_path.split('/')[-1].replace('.png', '') if image_path else ""
                    title_mappings = {
                        'Lengthsimage': '📏 Length Measurements',
                        'Pitch_Degrees': '📐 Roof Pitch (Degrees)',
                        'Pitch_on_12': '📐 Roof Pitch (Rise over 12)',
                        'Rafters': '🏗️ Rafter Structure',
                        'Azimuth': '🧭 Roof Azimuth/Direction',
                        'Area': '📊 Roof Area Measurements',
                        'Roof_Penetrations': '🔍 Roof Penetrations',
                        'Top_View': '🛰️ Aerial/Top View',
                        'North_Side': '⬆️ North Side View',
                        'South_Side': '⬇️ South Side View',
                        'East_Side': '➡️ East Side View',
                        'West_Side': '⬅️ West Side View',
                        'Cover_Image': '🏠 Cover/Overview Image',
                        'Structure_Summary': '📋 Structure Summary',
                        'processed_DDD': '📐 DDD Diagram (Similar Property)',
                        'roof_overlay_without_lengths': '🏠 Generated Roof Overlay'
                    }

                    display_title = title_mappings.get(filename, result.get("section", "Unknown Image"))

                    # Filter images based on important_imagery if specified
                    if hasattr(query_analysis, 'important_imagery') and query_analysis.important_imagery:
                        # Map filename to the image type names used by the LLM
                        filename_to_type = {
                            'Lengthsimage': 'Lengths',
                            'Pitch_Degrees': 'Pitch_Degrees',
                            'Pitch_on_12': 'Pitch_on_12',
                            'Rafters': 'Rafters',
                            'Azimuth': 'Azimuth',
                            'Area': 'Area',
                            'Roof_Penetrations': 'Roof_Penetrations',
                            'Top_View': 'Top_View',
                            'North_Side': 'North_Side',
                            'South_Side': 'South_Side',
                            'East_Side': 'East_Side',
                            'West_Side': 'West_Side',
                            'Cover_Image': 'Cover_Image',
                            'Structure_Summary': 'Structure_Summary'
                        }

                        # Get the image type from filename
                        image_type = filename_to_type.get(filename, filename)
                        if image_type not in query_analysis.important_imagery:
                            # Skip this image if it's not in the important imagery list
                            continue

                    # Level 2 chunk info for images
                    chunk_info = {
                        "chunk_id": result.get("chunk_id") + f"_{filename}" if len(image_paths) > 1 else result.get("chunk_id"),
                        "section": display_title,
                        "chunk_type": result.get("chunk_type"),
                        "content": f"Image: {display_title}",
                        "distance": result.get("distance", 0),
                        "image_path": image_path,
                        "image_title": display_title,
                        "image_description": description or f"{display_title} image"
                    }
                    level2_chunks.append(chunk_info)
            else:
                # Fallback for chunks without extractable image paths
                display_title = result.get("section", "Unknown Image")
                chunk_info = {
                    "chunk_id": result.get("chunk_id"),
                    "section": display_title,
                    "chunk_type": result.get("chunk_type"),
                    "content": f"Image: {display_title}",
                    "distance": result.get("distance", 0),
                    "image_path": "",
                    "image_title": display_title,
                    "image_description": description or "Property image"
                }
                level2_chunks.append(chunk_info)

            # Collect unique documents from all results
            for result in search_results:
                doc_info = {
                    "property_id": result.get("property_id"),
                    "address": result.get("doc_address"),
                    "report_id": result.get("report_id"),
                    "pdf_filename": result.get("pdf_filename")
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

@app.get("/roof-pitch-data")
async def get_roof_pitch_data():
    """Get detailed roof pitch information with actual values and images."""
    try:
        import json
        
        final_chunks_dir = Path("Final_Chunks")
        if not final_chunks_dir.exists():
            raise HTTPException(status_code=404, detail="Final_Chunks directory not found")
        
        roof_pitch_data = []
        
        # Process each JSON file in Final_Chunks
        for json_file in final_chunks_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                property_info = {}
                
                # Extract basic property information
                for chunk in data.get('text', []):
                    if chunk.get('section') == 'Report Header':
                        property_info.update(chunk.get('data', {}))
                    elif chunk.get('section') == 'Measurements - Structure 1':
                        structure_data = chunk.get('data', {})
                        property_info['predominant_pitch'] = structure_data.get('predominant_pitch')
                        property_info['total_roof_facets'] = structure_data.get('total_roof_facets')
                        property_info['total_roof_area'] = structure_data.get('total_area_all_pitches')
                
                # Extract pitch images
                pitch_images = {}
                for chunk in data.get('text', []):
                    if chunk.get('section') == 'Pitch (on 12) Diagram':
                        chunk_data = chunk.get('data', {})
                        pitch_images['pitch_on_12'] = {
                            'image_path': chunk_data.get('image_file'),
                            'title': '📐 Roof Pitch (Rise over 12)',
                            'description': chunk_data.get('description', 'Roof pitch image in x/12 format')
                        }
                    elif chunk.get('section') == 'Pitch (Degrees) Diagram':
                        chunk_data = chunk.get('data', {})
                        pitch_images['pitch_degrees'] = {
                            'image_path': chunk_data.get('image_file'),
                            'title': '📐 Roof Pitch (Degrees)',
                            'description': chunk_data.get('description', 'Roof pitch image in degrees')
                        }
                
                # Extract pitch table data (Areas per Pitch)
                pitch_breakdown = []
                for table in data.get('table', []):
                    if 'Areas_per_Pitch' in table.get('section', ''):
                        raw_text = table.get('raw_text', [])
                        if isinstance(raw_text, list):
                            for row in raw_text:
                                if isinstance(row, dict) and 'Roof Pitches' in row:
                                    pitch_breakdown.append({
                                        'pitch': row.get('Roof Pitches'),
                                        'area': row.get('Area (sq ft)', row.get('Area (m²)')),
                                        'percentage': row.get('%of Roof')
                                    })
                        break
                
                if property_info.get('report_id'):
                    roof_pitch_data.append({
                        'property_id': property_info.get('report_id'),
                        'property_address': property_info.get('property_address'),
                        'predominant_pitch': property_info.get('predominant_pitch'),
                        'total_roof_facets': property_info.get('total_roof_facets'),
                        'total_roof_area': property_info.get('total_roof_area'),
                        'pitch_breakdown': pitch_breakdown,
                        'images': pitch_images
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing {json_file}: {e}")
                continue
        
        return {
            "status": "success",
            "properties_found": len(roof_pitch_data),
            "roof_pitch_data": roof_pitch_data
        }
        
    except Exception as e:
        logger.error(f"Error extracting roof pitch data: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error extracting roof pitch data: {str(e)}")

async def get_roof_pitch_data_internal():
    """Internal function to get roof pitch data (same as endpoint but returns data directly)."""
    try:
        import json
        
        final_chunks_dir = Path("Final_Chunks")
        if not final_chunks_dir.exists():
            return None
        
        roof_pitch_data = []
        
        # Process each JSON file in Final_Chunks
        for json_file in final_chunks_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                property_info = {}
                
                # Extract basic property information
                for chunk in data.get('text', []):
                    if chunk.get('section') == 'Report Header':
                        property_info.update(chunk.get('data', {}))
                    elif chunk.get('section') == 'Measurements - Structure 1':
                        structure_data = chunk.get('data', {})
                        property_info['predominant_pitch'] = structure_data.get('predominant_pitch')
                        property_info['total_roof_facets'] = structure_data.get('total_roof_facets')
                        property_info['total_roof_area'] = structure_data.get('total_area_all_pitches')
                
                # Extract pitch images
                pitch_images = {}
                for chunk in data.get('text', []):
                    if chunk.get('section') == 'Pitch (on 12) Diagram':
                        chunk_data = chunk.get('data', {})
                        pitch_images['pitch_on_12'] = {
                            'image_path': chunk_data.get('image_file'),
                            'title': '📐 Roof Pitch (Rise over 12)',
                            'description': chunk_data.get('description', 'Roof pitch image in x/12 format')
                        }
                    elif chunk.get('section') == 'Pitch (Degrees) Diagram':
                        chunk_data = chunk.get('data', {})
                        pitch_images['pitch_degrees'] = {
                            'image_path': chunk_data.get('image_file'),
                            'title': '📐 Roof Pitch (Degrees)',
                            'description': chunk_data.get('description', 'Roof pitch image in degrees')
                        }
                
                # Extract pitch table data (Areas per Pitch)
                pitch_breakdown = []
                for table in data.get('table', []):
                    if 'Areas_per_Pitch' in table.get('section', ''):
                        raw_text = table.get('raw_text', [])
                        if isinstance(raw_text, list):
                            for row in raw_text:
                                if isinstance(row, dict) and 'Roof Pitches' in row:
                                    pitch_breakdown.append({
                                        'pitch': row.get('Roof Pitches'),
                                        'area': row.get('Area (sq ft)', row.get('Area (m²)')),
                                        'percentage': row.get('%of Roof')
                                    })
                        break
                
                if property_info.get('report_id'):
                    roof_pitch_data.append({
                        'property_id': property_info.get('report_id'),
                        'property_address': property_info.get('property_address'),
                        'predominant_pitch': property_info.get('predominant_pitch'),
                        'total_roof_facets': property_info.get('total_roof_facets'),
                        'total_roof_area': property_info.get('total_roof_area'),
                        'pitch_breakdown': pitch_breakdown,
                        'images': pitch_images
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing {json_file}: {e}")
                continue
        
        return {
            "status": "success",
            "properties_found": len(roof_pitch_data),
            "roof_pitch_data": roof_pitch_data
        }
        
    except Exception as e:
        logger.error(f"Error extracting roof pitch data: {e}")
        return None

def format_roof_pitch_response(roof_data):
    """Format roof pitch data as a clean text response for chat with result console integration."""
    properties = roof_data.get("roof_pitch_data", [])
    
    if not properties:
        return "No roof pitch data found in the database."
    
    # Create a clean text response that will trigger the result console
    response = f"""# 🏠 Roof Pitch Information

Found **{len(properties)}** properties with detailed roof pitch data including:

📊 **Analysis Overview:**
- Predominant pitch values for each property
- Complete roof facet breakdowns  
- Detailed area measurements and percentages
- Visual pitch diagrams (degrees and X/12 format)

📋 **Properties Analyzed:**
"""
    
    # Add a summary list of properties
    for i, prop in enumerate(properties, 1):
        property_id = prop.get('property_id', 'Unknown')
        address = prop.get('property_address', 'Unknown Address')
        pitch = prop.get('predominant_pitch', 'N/A')
        
        response += f"{i}. **Property {property_id}** - {pitch} pitch\n   {address}\n\n"
    
    response += """
🔍 **What you can view:**
- Interactive property cards with all measurements
- Detailed pitch breakdown tables
- High-resolution roof pitch diagrams
- Comprehensive area calculations

*Click the "View Results" button below to open the detailed analysis in the result console.*
"""
    
    return response

async def get_measurement_data_internal(measurement_type):
    """Generic function to extract measurement data by type."""
    try:
        import json
        
        final_chunks_dir = Path("Final_Chunks")
        if not final_chunks_dir.exists():
            return None
        
        measurement_data = []
        
        # Process each JSON file in Final_Chunks
        for json_file in final_chunks_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                property_info = {}
                measurement_images = {}
                
                # Extract basic property information and measurements
                for chunk in data.get('text', []):
                    chunk_data = chunk.get('data', {})
                    
                    if 'report_id' in chunk_data:
                        property_info.update(chunk_data)
                    
                    if 'structure_name' in chunk_data:
                        property_info.update(chunk_data)
                
                # Extract relevant images based on measurement type
                for chunk in data.get('text', []):
                    if chunk.get('type') == 'image':
                        section = chunk.get('section', '').lower()
                        chunk_data = chunk.get('data', {})
                        image_file = chunk_data.get('image_file', '')
                        
                        # Extract filename for better matching
                        filename = image_file.split('/')[-1].lower() if image_file else ''
                        
                        if measurement_type == 'lengths' and ('lengths' in section or 'lengthsimage' in filename):
                            measurement_images['lengths_image'] = {
                                'image_path': image_file,
                                'title': '📏 Length Measurements',
                                'description': chunk_data.get('description', 'Roof length measurements image')
                            }
                        elif measurement_type == 'rafters' and ('rafters' in section or 'rafters' in filename):
                            measurement_images['rafters_image'] = {
                                'image_path': image_file,
                                'title': '🏗️ Rafter Structure',
                                'description': chunk_data.get('description', 'Rafter measurements image')
                            }
                        elif measurement_type == 'area' and ('area' in section or 'area' in filename):
                            measurement_images['area_image'] = {
                                'image_path': image_file,
                                'title': '📊 Roof Area Measurements',
                                'description': chunk_data.get('description', 'Roof area measurements image')
                            }
                        elif measurement_type == 'azimuth' and ('azimuth' in section or 'azimuth' in filename):
                            measurement_images['azimuth_image'] = {
                                'image_path': image_file,
                                'title': '🧭 Roof Azimuth/Direction',
                                'description': chunk_data.get('description', 'Roof orientation image')
                            }
                
                if property_info.get('report_id'):
                    measurement_data.append({
                        'property_id': property_info.get('report_id'),
                        'property_address': property_info.get('property_address'),
                        'measurements': property_info,
                        'images': measurement_images
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing {json_file}: {e}")
                continue
        
        return {
            "status": "success",
            "measurement_type": measurement_type,
            "properties_found": len(measurement_data),
            "measurement_data": measurement_data
        }
        
    except Exception as e:
        logger.error(f"Error extracting {measurement_type} data: {e}")
        return None

@app.get("/lengths-data")
async def get_lengths_data():
    """Get detailed roof length measurements (ridges, hips, valleys, etc.)."""
    result = await get_measurement_data_internal('lengths')
    if result:
        return result
    else:
        raise HTTPException(status_code=500, detail="Error extracting length measurement data")

@app.get("/rafters-data")
async def get_rafters_data():
    """Get detailed rafter measurement data."""
    result = await get_measurement_data_internal('rafters')
    if result:
        return result
    else:
        raise HTTPException(status_code=500, detail="Error extracting rafter measurement data")

@app.get("/area-data")
async def get_area_data():
    """Get detailed roof area measurement data."""
    result = await get_measurement_data_internal('area')
    if result:
        return result
    else:
        raise HTTPException(status_code=500, detail="Error extracting area measurement data")

@app.get("/azimuth-data")
async def get_azimuth_data():
    """Get detailed roof azimuth/orientation data."""
    result = await get_measurement_data_internal('azimuth')
    if result:
        return result
    else:
        raise HTTPException(status_code=500, detail="Error extracting azimuth measurement data")

@app.get("/all-images-data")
async def get_all_images_data():
    """Get comprehensive image data for all properties including all diagram types and property views."""
    try:
        import json
        
        final_chunks_dir = Path("Final_Chunks")
        if not final_chunks_dir.exists():
            raise HTTPException(status_code=404, detail="Final_Chunks directory not found")
        
        image_data = []
        
        # Process each JSON file in Final_Chunks
        for json_file in final_chunks_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                property_info = {}
                measurement_images = []
                property_views = []
                roof_analysis = []
                
                # Extract basic property information
                for chunk in data.get('text', []):
                    chunk_data = chunk.get('data', {})
                    
                    if 'report_id' in chunk_data:
                        property_info.update(chunk_data)
                
                # Extract all images and categorize them
                for chunk in data.get('text', []):
                    if chunk.get('type') == 'image':
                        section = chunk.get('section', '')
                        section_lower = section.lower()
                        chunk_data = chunk.get('data', {})
                        
                        # Handle single image
                        if 'image_file' in chunk_data:
                            # Extract filename for better titles
                            image_path = chunk_data['image_file']
                            filename = image_path.split('/')[-1].replace('.png', '')
                            
                            # Create user-friendly title based on filename
                            title_mappings = {
                                'Lengthsimage': '📏 Length Measurements',
                                'Pitch_Degrees': '📐 Roof Pitch (Degrees)',
                                'Pitch_on_12': '📐 Roof Pitch (Rise over 12)',
                                'Rafters': '🏗️ Rafter Structure',
                                'Azimuth': '🧭 Roof Azimuth/Direction',
                                'Area': '📊 Roof Area Measurements',
                                'Roof_Penetrations': '🔍 Roof Penetrations',
                                'Top_View': '🛰️ Aerial/Top View',
                                'North_Side': '⬆️ North Side View',
                                'South_Side': '⬇️ South Side View',
                                'East_Side': '➡️ East Side View',
                                'West_Side': '⬅️ West Side View',
                                'Cover_Image': '🏠 Cover/Overview Image',
                                'Structure_Summary': '📋 Structure Summary'
                            }
                            
                            display_title = title_mappings.get(filename, section or 'Unknown Image')
                            
                            image_info = {
                                'type': section_lower.replace(' ', '_'),
                                'title': display_title,
                                'description': chunk_data.get('description', ''),
                                'image_path': image_path
                            }
                            
                            # Categorize images based on section and filename
                            if any(x in section_lower for x in ['length', 'pitch', 'rafter', 'azimuth', 'area']) or \
                               any(x in filename.lower() for x in ['lengthsimage', 'pitch_degrees', 'pitch_on_12', 'rafters', 'azimuth', 'area']):
                                measurement_images.append(image_info)
                            elif any(x in section_lower for x in ['obstruction', 'penetration', 'roof penetrations']) or \
                                 'roof_penetrations' in filename.lower():
                                roof_analysis.append(image_info)
                            elif any(x in filename.lower() for x in ['top_view', 'north_side', 'south_side', 'east_side', 'west_side', 'cover_image']):
                                property_views.append(image_info)
                            else:
                                # Default to roof analysis for other images
                                roof_analysis.append(image_info)
                        
                        # Handle multiple images (property views) - if this structure exists
                        if 'images' in chunk_data and isinstance(chunk_data['images'], list):
                            for i, image_path in enumerate(chunk_data['images']):
                                # Extract view type from filename
                                filename = image_path.split('/')[-1].replace('.png', '').replace('_', ' ')
                                
                                image_info = {
                                    'type': f'property_view_{i}',
                                    'title': filename.title(),
                                    'description': f'Property view: {filename}',
                                    'image_path': image_path
                                }
                                property_views.append(image_info)
                
                if property_info.get('report_id'):
                    image_data.append({
                        'property_id': property_info.get('report_id'),
                        'property_address': property_info.get('property_address'),
                        'images': {
                            'measurement_images': measurement_images,
                            'property_views': property_views,
                            'roof_analysis': roof_analysis
                        }
                    })
                    
            except Exception as e:
                logger.warning(f"Error processing {json_file}: {e}")
                continue
        
        return {
            "status": "success",
            "properties_found": len(image_data),
            "image_data": image_data
        }
        
    except Exception as e:
        logger.error(f"Error extracting all images data: {e}")
        raise HTTPException(status_code=500, detail="Error extracting all images data")

def format_all_images_response(image_data):
    """Format all images data as a clean text response for chat."""
    properties = image_data.get("image_data", [])
    
    if not properties:
        return "No image data found in the database."
    
    # Count total images
    total_images = 0
    for prop in properties:
        total_images += len(prop['images']['measurement_images'])
        total_images += len(prop['images']['property_views'])
        total_images += len(prop['images']['roof_analysis'])
    
    response = f"""# 🏠 Complete Property Image Gallery

Found **{len(properties)}** properties with comprehensive image data including **{total_images}** total images:

📊 **Image Categories Available:**
- 📐 **Measurement Images**: Length measurements, pitch analysis, rafter calculations, area breakdowns, azimuth orientations
- 🏠 **Property Views**: Aerial imagery from top, north, south, east, and west perspectives  
- 🔍 **Roof Analysis**: Roof penetrations, obstructions, and structural analysis images

💡 **Visual Analysis Features:**
- High-resolution property imagery from multiple angles
- Technical measurement images with precise calculations
- Roof obstruction and penetration mapping
- Comprehensive structural analysis visuals

*Click the "View Results" button below to open the complete image gallery in the side panel.*
"""
    
    return response

def format_measurement_response(measurement_data, measurement_type):
    """Format measurement data as a clean text response for chat."""
    properties = measurement_data.get("measurement_data", [])
    
    if not properties:
        return f"No {measurement_type} measurement data found in the database."
    
    # Create measurement-specific response
    measurement_titles = {
        'lengths': '📏 Roof Length Measurements',
        'rafters': '🏗️ Rafter Analysis', 
        'area': '📐 Roof Area Analysis',
        'azimuth': '🧭 Roof Orientation Analysis'
    }
    
    title = measurement_titles.get(measurement_type, f'{measurement_type.title()} Analysis')
    
    response = f"""# {title}

Found **{len(properties)}** properties with detailed {measurement_type} measurement data including:

📊 **Analysis Overview:**
- Comprehensive {measurement_type} measurements for each property
- Detailed breakdowns with precise values
- Visual measurement diagrams and technical drawings
- Property-specific measurement data

💡 **Available Data Types:**"""

    if measurement_type == 'lengths':
        response += """
- Ridge lengths and counts
- Hip measurements  
- Valley dimensions
- Rake measurements
- Eaves and starter measurements
- Drip edge lengths
- Flashing and step flashing details"""
    elif measurement_type == 'rafters':
        response += """
- Rafter length calculations
- Section-specific measurements
- Structural analysis data"""
    elif measurement_type == 'area':
        response += """
- Total roof area calculations
- Facet-specific area measurements  
- Area breakdowns by section"""
    elif measurement_type == 'azimuth':
        response += """
- Roof facet orientations
- Compass bearing measurements
- True north references"""

    response += f"""

*Click the "View Results" button below to open the detailed {measurement_type} analysis in the result console.*
"""
    
    return response

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
            "rebuild": "/rebuild",
            "roof_pitch_data": "/roof-pitch-data"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Different port to avoid conflicts