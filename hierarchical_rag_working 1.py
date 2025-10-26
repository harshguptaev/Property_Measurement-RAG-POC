"""
Working Hierarchical RAG System with Two-Level Indices
Simplified version that works with MilvusClient without complex index creation
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time
import argparse
import sys

import boto3
from pymilvus import MilvusClient
from tqdm import tqdm

# Import the general query handler module
from query_router import GeneralQueryHandler

# Simple MilvusCollectionManager class
class MilvusCollectionManager:
    """Simple collection manager for Milvus operations"""
    
    def __init__(self, uri: str):
        self.client = MilvusClient(uri=uri)
    
    def create_collection_safely(self, collection_name: str, embedding_dim: int, metric_type: str = "COSINE",
                                clear_existing: bool = True, index_params: Dict = None):
        """Create a collection safely, optionally clearing existing data"""
        if clear_existing and self.client.has_collection(collection_name):
            logger.info(f"Dropping existing collection: {collection_name}")
            self.client.drop_collection(collection_name)

        if not self.client.has_collection(collection_name):
            logger.info(f"Creating collection: {collection_name} with dimension {embedding_dim}, metric {metric_type}")
            self.client.create_collection(
                collection_name=collection_name,
                dimension=embedding_dim,
                metric_type=metric_type,
                index_params=index_params
            )
    
    def get_collection_info(self, collection_name: str) -> Dict:
        """Get collection information"""
        if not self.client.has_collection(collection_name):
            return {"entity_count": 0}
        
        try:
            stats = self.client.get_collection_stats(collection_name)
            return {"entity_count": stats.get("row_count", 0)}
        except:
            return {"entity_count": "unknown"}
    
    def list_collections(self) -> List[str]:
        """List all collections"""
        return self.client.list_collections()
    
    def clear_collection(self, collection_name: str):
        """Clear a collection"""
        if self.client.has_collection(collection_name):
            self.client.drop_collection(collection_name)
            logger.info(f"Cleared collection: {collection_name}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HierarchicalRAG:
    """
    Hierarchical RAG System with Two-Level Indices:
    - Level 1: Document Summary/Metadata Index
    - Level 2: Chunk Index
    """
    
    def __init__(self, region_name: str = "us-east-1", model_id: str = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"):
        """
        Initialize the Hierarchical RAG System
        What was the % of Roof and Area covered where roof pitch is about 6/12 for address  2455 New Holland Cir, Murfreesboro, TN 37128
        Args:
            region_name: AWS region for Bedrock
            model_id: Bedrock model ID for text processing
        """
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=region_name)
        self.model_id = model_id
        self.region_name = region_name

        # Initialize Milvus client (using full Milvus via Docker)
        logger.info("🔗 Connecting to Milvus database at http://localhost:19530")
        try:
            self.milvus_client = MilvusClient(uri="http://localhost:19530")
            # Test connection
            collections = self.milvus_client.list_collections()
            logger.info(f"✅ Connected to Milvus successfully. Found {len(collections)} existing collections.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Milvus at http://localhost:19530: {str(e)}")
            logger.error("Please ensure Milvus is running with: ./start_milvus.sh")
            raise

        # Initialize Milvus collection manager for safe collection handling
        self.milvus_manager = MilvusCollectionManager(uri="http://localhost:19530")

        # Collection names for the two levels
        self.level1_collection_name = "hierarchical_level1"
        self.level2_collection_name = "hierarchical_level2"

        # Embedding dimensions: L1 uses 384 (address/geometry), L2 uses 1536 (full semantic)
        # Default Titan embedding dimension is 1536, reduced to 384 for L1 collections
        self.embedding_dim = 1536  # Default Titan embedding dimension

        # Initialize General Query Handler for cross-property queries
        self.general_query_handler = GeneralQueryHandler(self.milvus_client)
        # Inject the required dependencies into the handler
        self.general_query_handler._get_query_embedding = self.titan_embed_text
        self.general_query_handler._bedrock_client = self.bedrock_client

        logger.info(f"🔧 Initialized Hierarchical RAG with model: {model_id}")
        logger.info(f"🔧 Level 1 collection: {self.level1_collection_name}")
        logger.info(f"🔧 Level 2 collection: {self.level2_collection_name}")
        logger.info(f"🔧 General Query Handler: Initialized")
    
    def titan_embed_text(self, text: str, target_dim: int = 1536) -> List[float]:
        """
        Get text embeddings from Amazon Titan Embed Text v1

        Args:
            text: Text to embed
            target_dim: Target dimension for the embedding (384 for L1, 1536 for L2)

        Returns:
            List of embedding values
        """
        try:
            body = json.dumps({
                "inputText": text
            })

            response = self.bedrock_client.invoke_model(
                modelId="amazon.titan-embed-text-v1",
                body=body,
                accept="application/json",
                contentType="application/json"
            )

            result = json.loads(response["body"].read())
            embedding = result["embedding"]

            # Titan returns 1536 dimensions by default
            # For L1 collections, we need to reduce to 384 dimensions
            if target_dim == 384 and len(embedding) == 1536:
                # Simple dimension reduction by taking every 4th element (1536 / 4 = 384)
                embedding = embedding[::4]
                logger.debug(f"Reduced embedding from 1536 to {len(embedding)} dimensions for L1")

            return embedding

        except Exception as e:
            logger.error(f"Error generating text embedding: {str(e)}")
            raise
    
    def create_document_summary(self, chunks_data: List[Dict]) -> str:
        """
        Create a document summary from chunks using LLM
        
        Args:
            chunks_data: List of chunks from a document
            
        Returns:
            Document summary string
        """
        # Combine all text content from chunks
        combined_content = ""
        for chunk in chunks_data:
            if chunk.get("type") in ["text", "table"]:
                content = chunk.get("data", {})
                if isinstance(content, dict):
                    # Flatten the data dictionary into readable text
                    for key, value in content.items():
                        combined_content += f"{key}: {value}\n"
                else:
                    combined_content += str(content) + "\n"
        
        # Create summary prompt
        prompt = f"""Create a concise summary of this roofing report document. Focus on key information like:
- Property address and location
- Roof measurements (area, facets, pitch)
- Key structural details
- Important findings or observations

Document Content:
{combined_content[:3000]}

Provide a clear, structured summary in 2-3 sentences:"""
        
        try:
            body = {
                "anthropic_version": "anthropic.claude-3-7-sonnet-20250219-v1:0",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            response_body = json.loads(response['body'].read())
            return response_body['content'][0]['text'].strip()
            
        except Exception as e:
            logger.error(f"Error creating document summary: {str(e)}")
            # Fallback to simple concatenation
            return f"Roofing report with {len(chunks_data)} sections covering measurements, images, and property details."
    
    def create_milvus_collections(self, clear_existing: bool = True):
        """
        Create Milvus collections using the collection manager for safe handling
        
        Args:
            clear_existing: Whether to clear existing collections to prevent duplicates
        """
        logger.info("🔧 Creating Milvus collections for hierarchical indexing")
        
        if clear_existing:
            logger.info("🗑️  Clearing existing collections to prevent duplicates...")
        
        # Level 1 Collection Configuration (Addresses, Geometry, Filters)
        # - Fast lookup by address/coordinates + limited semantic search
        # - ~200K vectors, 384 dimensions, COSINE metric
        level1_index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {
                "M": 24,  # Smaller neighborhood graph for L1
                "efConstruction": 100  # Lower construction effort for L1
            }
        }

        # Level 2 Collection Configuration (Semantic Chunks, Embeddings)
        # - Heavy semantic similarity search across all property chunks
        # - ~1M vectors, 1536 dimensions, COSINE metric
        level2_index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {
                "M": 32,  # Higher degree for better recall in L2
                "efConstruction": 200  # Higher construction effort for L2
            }
        }

        # Create Level 1 collection (Document summaries) - 384 dimensions
        self.milvus_manager.create_collection_safely(
            collection_name=self.level1_collection_name,
            embedding_dim=384,  # L1 uses smaller embeddings for address/geometry lookup
            metric_type="COSINE",
            clear_existing=clear_existing,
            index_params=level1_index_params
        )

        # Create Level 2 collection (Chunks) - 1536 dimensions
        self.milvus_manager.create_collection_safely(
            collection_name=self.level2_collection_name,
            embedding_dim=1536,  # L2 uses full Titan embeddings for semantic search
            metric_type="COSINE",
            clear_existing=clear_existing,
            index_params=level2_index_params
        )
        
        logger.info(f"✅ Level 1 collection ready: {self.level1_collection_name}")
        logger.info(f"✅ Level 2 collection ready: {self.level2_collection_name}")
        
        # Show collection status
        level1_info = self.milvus_manager.get_collection_info(self.level1_collection_name)
        level2_info = self.milvus_manager.get_collection_info(self.level2_collection_name)
        logger.info(f"📊 Level 1 entities: {level1_info.get('entity_count', 0)}")
        logger.info(f"📊 Level 2 entities: {level2_info.get('entity_count', 0)}")
    
    def build_level1_index(self, documents_data: List[Dict], clear_existing: bool = True) -> None:
        """
        Build Level 1 Index (Parent Chunk Index)

        Args:
            documents_data: List of documents with their chunks
            clear_existing: Whether to clear existing collections to prevent duplicates
        """
        logger.info("🏗️ Building Level 1 Index (Parent Chunk Index)")

        # Create collections first (with duplicate prevention)
        self.create_milvus_collections(clear_existing=clear_existing)

        level1_data = []

        for i, doc in enumerate(tqdm(documents_data, desc="Building Level 1 Index")):
            try:
                # Extract document metadata
                doc_id = doc.get("doc_id", f"doc_{i}")
                source_file = doc.get("source_file", "unknown.pdf")
                chunks = doc.get("chunks", [])

                # The first chunk contains property metadata
                if not chunks or len(chunks) == 0:
                    logger.warning(f"No chunks found for document {doc_id}")
                    continue

                property_chunk = chunks[0]  # First chunk is property metadata

                # Extract property information from the first chunk
                property_id = property_chunk.get("property_id", f"PROP_{doc_id}")
                address = property_chunk.get("address", "Unknown Address")
                latitude = property_chunk.get("latitude", 0.0)
                longitude = property_chunk.get("longitude", 0.0)

                # Extract report_id from property_id (remove PROP_ prefix)
                report_id = property_id.replace("PROP_", "")

                # Generate PDF filename
                # Format: {report_id}_{address_cleaned}.pdf
                address_cleaned = address.replace(" ", "_").replace(",", "").replace(".", "")
                pdf_filename = f"{report_id}_{address_cleaned}.pdf"

                # Collect child chunk IDs (all chunks except the first property metadata chunk)
                child_chunk_ids = []
                for chunk in chunks[1:]:  # Skip the first chunk (property metadata)
                    chunk_id = chunk.get("chunk_id", "")
                    if chunk_id:
                        child_chunk_ids.append(chunk_id)

                print("=" * 80)
                print(f"property_id ::: {property_id}")
                print(f"report_id ::: {report_id}")
                print(f"address ::: {address}")
                print(f"latitude ::: {latitude}")
                print(f"longitude ::: {longitude}")
                print(f"pdf_filename ::: {pdf_filename}")
                print(f"child_chunk_ids ::: {child_chunk_ids}")
                print("=" * 80)

                # Generate embedding for the address (for vector search) - L1 uses 384 dimensions
                address_embedding = self.titan_embed_text(address, target_dim=384)

                level1_data.append({
                    "id": i,
                    "vector": address_embedding,
                    "property_id": property_id,
                    "report_id": report_id,
                    "data": {
                        "address": address,
                        "latitude": latitude,
                        "longitude": longitude
                    },
                    "pdf_filename": pdf_filename,
                    "child_chunk_ids": child_chunk_ids
                })

                logger.info(f"Created Level 1 entry for {property_id}: {address}")
                print()
                print()

            except Exception as e:
                logger.error(f"Error processing document {i}: {str(e)}")
                continue

        # Insert data into Level 1 collection
        if level1_data:
            self.milvus_client.insert(
                collection_name=self.level1_collection_name,
                data=level1_data
            )
            logger.info(f"✅ Level 1 Index built with {len(level1_data)} parent chunks")
        else:
            logger.warning("No data to insert into Level 1 Index")
    
    def build_level2_index(self, documents_data: List[Dict], clear_existing: bool = False) -> None:
        """
        Build Level 2 Index (Children Chunks Index)

        Args:
            documents_data: List of documents with their chunks
            clear_existing: Whether to clear existing collections (usually False since Level 1 already did this)
        """
        logger.info("🏗️ Building Level 2 Index (Children Chunks Index)")

        # Only create collections if they don't exist (Level 1 should have created them)
        if not self.milvus_client.has_collection(self.level2_collection_name):
            logger.warning("Level 2 collection doesn't exist, creating it...")
            self.create_milvus_collections(clear_existing=clear_existing)

        level2_data = []
        chunk_counter = 0

        for doc in tqdm(documents_data, desc="Building Level 2 Index"):
            try:
                doc_id = doc.get("doc_id", "unknown")
                chunks = doc.get("chunks", [])

                # Skip the first chunk as it contains property metadata (stored in Level 1)
                for chunk in chunks[1:]:  # Start from index 1 to skip property metadata
                    try:
                        # Extract chunk information in the specified format
                        chunk_id = chunk.get("chunk_id", f"chunk_{chunk_counter}")
                        section = chunk.get("section", "Unknown Section")
                        chunk_type = chunk.get("type", "text")

                        # Get property_id from the chunk (should be set)
                        property_id = chunk.get("property_id", f"PROP_{doc_id}")

                        # Get the data field directly (contains the structured data)
                        data = chunk.get("data", {})

                        # Create chunk text for embedding (flatten the data for semantic search)
                        chunk_text = ""
                        if isinstance(data, dict):
                            # Flatten the data dictionary into readable text for embedding
                            for key, value in data.items():
                                chunk_text += f"{key}: {value}\n"
                        else:
                            chunk_text = str(data)

                        # Add section and type information to the chunk text
                        chunk_text = f"Section: {section}\nType: {chunk_type}\nContent: {chunk_text}"

                        # Generate embedding for chunk text - L2 uses full 1536 dimensions
                        chunk_embedding = self.titan_embed_text(chunk_text, target_dim=1536)

                        level2_data.append({
                            "id": chunk_counter,
                            "vector": chunk_embedding,
                            "chunk_id": chunk_id,
                            "property_id": property_id,
                            "section": section,
                            "type": chunk_type,
                            "data": data,
                            "chunk_text": chunk_text[:5000]  # Truncate if too long, for debugging
                        })

                        print(f"Created Level 2 chunk: {chunk_id} ({section}) for property {property_id}")
                        chunk_counter += 1

                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk.get('chunk_id', 'unknown')}: {str(e)}")
                        continue

            except Exception as e:
                logger.error(f"Error processing document {doc_id}: {str(e)}")
                continue

        # Insert data into Level 2 collection
        if level2_data:
            self.milvus_client.insert(
                collection_name=self.level2_collection_name,
                data=level2_data
            )
            logger.info(f"✅ Level 2 Index built with {len(level2_data)} children chunks")
        else:
            logger.warning("No data to insert into Level 2 Index")
    
    def detect_query_type(self, query: str) -> str:
        """
        Detect if a query is property-specific or general.

        Args:
            query: The search query

        Returns:
            "property_specific" if query mentions a specific property/address
            "general" if query is asking general questions across all properties
        """
        # First check if it's a general query using the GeneralQueryHandler
        if self.general_query_handler.is_general_query(query):
            return "general"

        # If not general, it's property-specific
        return "property_specific"

    def search_hierarchical(self, query: str, level1_limit: int = 1, level2_limit: int = 5) -> List[Dict]:
        """
        Perform hierarchical search: Level 1 → Level 2

        Args:
            query: Search query
            level1_limit: Number of documents to retrieve from Level 1
            level2_limit: Number of chunks to retrieve from Level 2

        Returns:
            List of relevant chunks with metadata
        """
        logger.info(f"🔍 Starting hierarchical search for: '{query}'")
        
        try:
            # Step 1: Embed query for Level 1 (384 dimensions for address/geometry lookup)
            query_vec_l1 = self.titan_embed_text(query, target_dim=384)

            # Step 2: Embed query for Level 2 (1536 dimensions for semantic search)
            query_vec_l2 = self.titan_embed_text(query, target_dim=1536)

            # Level 1 search parameters (efSearch = 64)
            level1_search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 64}  # efSearch for L1
            }

            # Level 2 search parameters (efSearch = 96-128, using 96 as default)
            level2_search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 96}  # efSearch for L2
            }
            
            # Step 3: Search Level 1 (Parent Chunks)
            logger.info("📊 Searching Level 1 (Parent Chunk Index)...")
            res1 = self.milvus_client.search(
                collection_name=self.level1_collection_name,
                data=[query_vec_l1],
                limit=1,
                output_fields=["property_id", "child_chunk_ids", "data", "report_id", "pdf_filename"],
                search_params=level1_search_params
            )
            
            if not res1 or not res1[0]:
                logger.warning("No results found in Level 1 Index")
                return []


            print(f"Fetching Chunks from Level 1 Index - Address matching")
            print(f"res1 ::: {res1}")
            # Get relevant chunk_ids
            retrieved_chunk_ids = []
            level1_docs = []

            for hit in res1[0]:
                # Extract data from the nested structure
                data = hit.get("data", {})
                address = data.get("address", "Unknown Address") if isinstance(data, dict) else "Unknown Address"

                doc_info = {
                    "property_id": hit.get("property_id"),
                    "address": address,
                    "report_id": hit.get("report_id"),
                    "pdf_filename": hit.get("pdf_filename"),
                    "distance": hit.get("distance", 0)
                }
                level1_docs.append(doc_info)

                print(f"\n{'='*80}")
                print(f"property_id: {doc_info['property_id']}")
                print(f"address: {doc_info['address']}")
                print(f"report_id: {doc_info['report_id']}")
                print(f"pdf_filename: {doc_info['pdf_filename']}")
                print(f"distance: {doc_info['distance']}")

                # Get child chunk IDs
                child_chunk_ids = hit.get("child_chunk_ids", [])
                if isinstance(child_chunk_ids, list):
                    retrieved_chunk_ids.extend(child_chunk_ids)

                print(f"retrieved_chunk_ids ::: {retrieved_chunk_ids}")
                print("=" * 80)
            
            logger.info(f"📋 Found {len(level1_docs)} relevant documents with {len(retrieved_chunk_ids)} total chunks")

            # Get property IDs from level 1 results for filtering level 2 search
            relevant_property_ids = [doc["property_id"] for doc in level1_docs]

            # Step 3: Search Level 2 (Chunks within retrieved docs)
            logger.info("📊 Searching Level 2 (Chunk Index)...")

            if not retrieved_chunk_ids:
                logger.warning("No chunk IDs found from Level 1 search")
                return []

            # Safer way: use repr() to auto-quote strings, then replace single quotes with double quotes
            property_ids_quoted = [f'"{pid}"' for pid in relevant_property_ids]

            # Search Level 2 chunks filtered by property_id from Level 1 results
            res2 = self.milvus_client.search(
                collection_name=self.level2_collection_name,
                data=[query_vec_l2],
                limit=5,  # Use the level2_limit parameter
                filter=f'property_id in [{",".join(property_ids_quoted)}]',
                output_fields=["chunk_text", "section", "property_id", "type", "chunk_id", "data"],
                search_params=level2_search_params
            )
            
            if not res2 or not res2[0]:
                logger.warning("No results found in Level 2 Index")
                return []
            
            # Step 4: Filter results to only include chunks from relevant documents
            final_results = []

            print(f"Fetching Chunks from Level 2 Index")

            print(f"\n{'='*80}")
            print(f"relevant_property_ids :: {relevant_property_ids}")
            print(f"Length :: {len(res2[0])}")
            print("=" * 80)

            # Collect all chunks (text, image, table) from relevant documents
            all_chunks = []

            for result in res2[0]:
                chunk_property_id = result.get("property_id")
                if chunk_property_id in relevant_property_ids:
                    chunk_info = {
                        "chunk_id": result.get("chunk_id"),
                        "property_id": chunk_property_id,
                        "section": result.get("section"),
                        "chunk_type": result.get("type"),  # Changed from chunk_type to type
                        "chunk_text": result.get("chunk_text"),
                        "data": result.get("data"),  # Include the structured data
                        "distance": result.get("distance", 0)
                    }

                    # Add document-level info from Level 1
                    for doc_info in level1_docs:
                        if doc_info["property_id"] == chunk_property_id:
                            chunk_info["doc_address"] = doc_info["address"]
                            chunk_info["report_id"] = doc_info["report_id"]
                            chunk_info["pdf_filename"] = doc_info["pdf_filename"]
                            break

                    all_chunks.append(chunk_info)

            # Sort all chunks by distance (relevance) and take top level2_limit
            all_chunks.sort(key=lambda x: x["distance"])
            final_results = all_chunks[:level2_limit]
            
            # Count chunk types for logging
            text_count = sum(1 for chunk in all_chunks if chunk["chunk_type"] != "image")
            image_count = sum(1 for chunk in all_chunks if chunk["chunk_type"] == "image")

            print(f"Total chunks found: {len(all_chunks)} (Text: {text_count}, Images: {image_count})")
            print(f"Final results: {len(final_results)}")
            print(f"level2_limit :: {level2_limit}")
            print(f"len(final_results) :: {len(final_results)}")

            print("\n" + "="*80)
            print("FINAL RESULTS - DISTINGUISHABLE CHUNKS")
            print("="*80)

            for i, chunk in enumerate(final_results, 1):
                print(f"\n{'='*60} CHUNK #{i} {'='*60}")
                print(f"📄 Type: {chunk.get('chunk_type', 'N/A').upper()}")
                print(f"🆔 Chunk ID: {chunk.get('chunk_id', 'N/A')}")
                print(f"🏠 Property ID: {chunk.get('property_id', 'N/A')}")
                print(f"📋 Section: {chunk.get('section', 'N/A')}")
                print(f"📍 Address: {chunk.get('doc_address', 'N/A')}")

                print("\n📖 Content:")
                content = chunk.get('chunk_text', 'N/A')
                if len(content) > 500:
                    print(f"   {content[:500]}...")
                else:
                    print(f"   {content}")

                # Show structured data if available
                data = chunk.get('data', {})
                if data and isinstance(data, dict):
                    print("\n📊 Structured Data:")
                    for key, value in data.items():
                        print(f"   {key}: {value}")

                print(f"{'='*60} END CHUNK #{i} {'='*60}")

            print("\n" + "="*80)
            print("END OF FINAL RESULTS")
            print("="*80)
            
            logger.info(f"✅ Hierarchical search completed. Found {len(final_results)} relevant chunks")
            return final_results
            
        except Exception as e:
            logger.error(f"Error during hierarchical search: {str(e)}")
            return []

    def generate_semantic_text(self, section: str, chunk_type: str, data: Dict) -> str:
        """
        Generate detailed, searchable semantic text for a chunk based on its section, type, and data.
        The text includes key measurements and features to enable better semantic search and retrieval.

        Args:
            section: The section name (e.g., "House Measurements", "Roof Measurements")
            chunk_type: The chunk type ("text" or "image")
            data: The chunk data dictionary

        Returns:
            str: Detailed human-readable semantic text describing the chunk
        """
        if chunk_type == "image":
            if section == "Imagery":
                views = list(data.keys())
                return f"Roof imagery section with visual perspectives: {', '.join(views).lower()}. Contains aerial and ground-level photos showing north, south, east, west, and top views of the roof structure for inspection and assessment."
            elif section == "Diagrams":
                diagrams = list(data.keys())
                return f"Technical diagrams section featuring: {', '.join(diagrams).lower()}. Includes detailed structural drawings showing lengths, pitch measurements, rafter configurations, azimuth angles, area calculations, and roof penetration locations."
            else:
                return f"Visual documentation section for {section.lower()} containing reference images and diagrams."

        elif chunk_type == "text":
            if section == "House Measurements":
                stories = data.get("number_of_stories", "unknown")
                attic = data.get("estimated_attic", data.get("estimated_attic_area_sqft", "unknown"))
                facets = data.get("total_roof_facets", "unknown")
                obstructions = data.get("total_roof_obstructions", "unknown")
                complexity = data.get("structure_complexity", "unknown")

                # Clean up attic value (remove 'sq ft' if present)
                if isinstance(attic, str) and "sq ft" in attic:
                    attic = attic.replace(" sq ft", "")

                return f"House structural measurements and characteristics: {stories} story building with estimated attic area of {attic} square feet. Roof consists of {facets} separate facets with {obstructions} obstructions. Building complexity rated as {complexity.lower()}. Key metrics include number of stories, attic space, roof complexity, and obstruction count for material estimation and structural assessment."

            elif "Roof Measurements" in section:
                area = data.get("total_area", data.get("total_area_sqft", "unknown"))
                facets = data.get("total_roof_facets", "unknown")
                pitch = data.get("predominant_pitch", "unknown")
                ridges = data.get("ridges", "unknown")
                hips = data.get("hips", data.get("hips_ft", "unknown"))
                valleys = data.get("valleys", data.get("valleys_ft", "unknown"))
                rakes = data.get("rakes", data.get("rakes_ft", "unknown"))
                eaves = data.get("eaves_starter", data.get("eaves_ft", "unknown"))
                drip_edge = data.get("drip_edge", data.get("drip_edge_ft", "unknown"))
                parapet = data.get("parapet_walls", data.get("parapet_walls_ft", "unknown"))
                flashing = data.get("flashing", data.get("flashing_ft", "unknown"))
                step_flashing = data.get("step_flashing", data.get("step_flashing_ft", "unknown"))
                roof_obs_perimeter = data.get("roof_obstructions_perimeter", data.get("roof_obstructions_perimeter_ft", "unknown"))
                roof_obs_area = data.get("roof_obstructions_area", data.get("roof_obstructions_area_sqft", "unknown"))
                net_area = data.get("net_roof_area", data.get("net_roof_area_sqft", "unknown"))

                # Clean up values (extract numbers and convert units)
                def clean_measurement(value, unit="ft"):
                    if isinstance(value, str):
                        if f" {unit}" in value:
                            return value.split(f" {unit}")[0]
                        elif f" sq {unit}" in value:
                            return value.split(f" sq {unit}")[0]
                    return value

                area = clean_measurement(area, "ft")
                ridges = clean_measurement(ridges, "ft")
                hips = clean_measurement(hips, "ft")
                valleys = clean_measurement(valleys, "ft")
                rakes = clean_measurement(rakes, "ft")
                eaves = clean_measurement(eaves, "ft")
                drip_edge = clean_measurement(drip_edge, "ft")
                parapet = clean_measurement(parapet, "ft")
                flashing = clean_measurement(flashing, "ft")
                step_flashing = clean_measurement(step_flashing, "ft")
                roof_obs_perimeter = clean_measurement(roof_obs_perimeter, "ft")
                roof_obs_area = clean_measurement(roof_obs_area, "ft")
                net_area = clean_measurement(net_area, "ft")

                measurements = []
                if area: measurements.append(f"total roof area {area} sq ft")
                if facets: measurements.append(f"{facets} roof facets")
                if pitch: measurements.append(f"predominant pitch {pitch}")
                if ridges and ridges not in ["0", "0.0"]: measurements.append(f"ridge length {ridges} ft")
                if hips and hips not in ["0", "0.0"]: measurements.append(f"hip length {hips} ft")
                if valleys and valleys not in ["0", "0.0"]: measurements.append(f"valley length {valleys} ft")
                if rakes and rakes not in ["0", "0.0"]: measurements.append(f"rake length {rakes} ft")
                if eaves and eaves not in ["0", "0.0"]: measurements.append(f"eave length {eaves} ft")
                if drip_edge and drip_edge not in ["0", "0.0"]: measurements.append(f"drip edge {drip_edge} ft")
                if parapet and parapet not in ["0", "0.0"]: measurements.append(f"parapet wall {parapet} ft")
                if flashing and flashing not in ["0", "0.0"]: measurements.append(f"flashing {flashing} ft")
                if step_flashing and step_flashing not in ["0", "0.0"]: measurements.append(f"step flashing {step_flashing} ft")
                if roof_obs_perimeter and roof_obs_perimeter not in ["0", "0.0"]: measurements.append(f"roof obstruction perimeter {roof_obs_perimeter} ft")
                if roof_obs_area and roof_obs_area not in ["0", "0.0"]: measurements.append(f"roof obstruction area {roof_obs_area} sq ft")
                if net_area: measurements.append(f"net roof area {net_area} sq ft")

                measurement_text = ". ".join(measurements) if measurements else "various roof measurements"

                return f"Comprehensive roof measurements and specifications: {measurement_text}. Detailed breakdown includes total area, pitch information, ridge/hip/valley/rake measurements, eaves and drip edges, flashing details, obstruction calculations, and net area computations for accurate material estimation and roofing quotes."

            elif section == "Pitch Breakdown":
                return f"Roof pitch analysis and waste calculation breakdown. Contains detailed percentage waste factors for different roof pitches including 4%, 9%, 14%, 17%, 19%, 21%, 24%, 29% waste calculations. Essential for accurate material quantity estimation based on roof slope and pitch requirements."

            else:
                # Generic fallback for other text sections
                return f"Detailed {section.lower()} information and specifications for property assessment and measurement calculations."

        # Fallback for unknown types
        return f"Property measurement data for {section.lower()} containing detailed specifications and calculations."

        
    def generate_llm_response(self, query: str, results: List[Dict]) -> str:
        """
        Use LLM to generate a comprehensive response based on retrieved chunks
        
        Args:
            query: Original user query
            results: Retrieved chunks from hierarchical search
            
        Returns:
            LLM-generated response string
        """
        if not results:
            return "I couldn't find any relevant information to answer your question."
        
        # Prepare context from retrieved chunks
        context_parts = []

        # Check if we have image chunks
        images_found = [result for result in results if result.get('chunk_type') == 'image']

        for i, result in enumerate(results, 1):
            doc_address = result.get('doc_address', 'Unknown Address')
            section = result.get('section', 'Unknown Section')
            chunk_type = result.get('chunk_type', 'text')
            chunk_text = result.get('chunk_text', '')
            
            
            # Clean up chunk text for context
            if chunk_text.startswith('Section:'):
                # Remove the redundant section/type prefixes
                lines = chunk_text.split('\n')
                content_lines = []
                for line in lines:
                    if line.startswith('Content:'):
                        content_lines.append(line[8:].strip())  # Remove "Content:" prefix
                    elif not line.startswith(('Section:', 'Type:')):
                        content_lines.append(line)
                chunk_text = '\n'.join(content_lines).strip()
            
            context_parts.append(f"""
Document {i}: {doc_address}
Section: {section} ({chunk_type})
Content: {chunk_text}
""")
        
        # Add image chunks directly to context as JSON
        image_chunks = [result for result in results if result.get('chunk_type') == 'image']
        if image_chunks:
            context_parts.append(f"\n\nIMAGE CHUNKS FROM LEVEL 2 SEARCH:")
            for img_chunk in image_chunks:
                context_parts.append(f"""
Image Chunk ID: {img_chunk.get('chunk_id', 'N/A')}
Document: {img_chunk.get('doc_address', 'Unknown Address')}
Section: {img_chunk.get('section', 'Unknown Section')}
Description: {img_chunk.get('chunk_text', 'No description available')}
""")

        context = "\n".join(context_parts)

        prompt = f"""You are a professional EagleView assistant specializing in roofing analysis and property information.

Your task is to provide accurate, relevant information to customer questions based on retrieved property data.

INFORMATION PROVIDED:
- Level 1 chunks: General property information and overviews
- Level 2 chunks: Specific technical details (roof area, facets, pitch, measurements, etc.)

INSTRUCTIONS:
1. Answer ONLY using the information from the provided chunks
2. Provide complete, accurate measurements and technical details when available
3. If exact information is not available, infer reasonable estimates from related chunk data
4. Be concise but comprehensive - include all relevant measurements and specifications
5. Use professional, clear language appropriate for roofing industry customers
6. Include specific numbers, units, and technical terms as they appear in the chunks
7. Reference image data when relevant to the question
8. You will receive text chunks, image chunks, and table chunks containing comprehensive property data

QUESTION: {query}

RETRIEVED INFORMATION:
{context}

Provide a clear, professional answer that directly addresses the customer's question with specific details from the data."""

        try:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 2000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body)
            )
            
            response_body = json.loads(response['body'].read())
            llm_text = response_body['content'][0]['text'].strip()
            
            # Try to parse as JSON
            try:
                json_response = json.loads(llm_text)
                # Add images_found to the response if not already included
                if images_found and 'images_available' not in json_response:
                    json_response['images_available'] = images_found
                return json.dumps(json_response, indent=2)
            except json.JSONDecodeError:
                # If not valid JSON, return as-is
                logger.warning("LLM response is not valid JSON, returning as text")
                return llm_text
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            # Fallback to simple summary
            return self._create_fallback_response(query, results)
    
    def _create_fallback_response(self, query: str, results: List[Dict]) -> str:
        """
        Create a fallback response if LLM fails
        
        Args:
            query: Original user query
            results: Retrieved chunks
            
        Returns:
            Simple formatted response
        """
        response_parts = [f"Based on the roofing reports, here's what I found for your query: '{query}'\n"]
        
        for i, result in enumerate(results, 1):
            doc_address = result.get('doc_address', 'Unknown Address')
            section = result.get('section', 'Unknown Section')
            chunk_type = result.get('chunk_type', 'text')
            
            response_parts.append(f"{i}. Property: {doc_address}")
            response_parts.append(f"   Section: {section} ({chunk_type})")
            
            # Extract key information from chunk text
            chunk_text = result.get('chunk_text', '')
            if 'area:' in chunk_text.lower():
                # Extract area information
                lines = chunk_text.split('\n')
                for line in lines:
                    if 'area:' in line.lower():
                        response_parts.append(f"   {line.strip()}")
            
            response_parts.append("")
        
        return "\n".join(response_parts)
    
    def format_json_response(self, json_response: str) -> str:
        """
        Format JSON response with markdown tables for better frontend display
        
        Args:
            json_response: JSON string response from LLM
            
        Returns:
            Formatted markdown string with proper tables
        """
        try:
            data = json.loads(json_response)
            
            formatted_parts = []
            
            # Header with property info
            if "properties" in data and data["properties"]:
                prop = data["properties"][0]  # Assume single property for now
                address = prop.get('address', 'Unknown Address')
                formatted_parts.append(f"# 🏠 Property Analysis Report")
                formatted_parts.append(f"**Address:** {address}")
                formatted_parts.append("")
            
            # Executive Summary
            if "summary" in data:
                formatted_parts.append("## 📊 Executive Summary")
                formatted_parts.append(f"> {data['summary']}")
                formatted_parts.append("")
            
            # Main Measurements Table
            if "properties" in data and data["properties"]:
                prop = data["properties"][0]
                if "key_measurements" in prop:
                    measurements = prop["key_measurements"]
                    
                    formatted_parts.append("## 📏 Core Measurements")
                    formatted_parts.append("| Measurement Type | Value |")
                    formatted_parts.append("|------------------|-------|")
                    
                    # Basic measurements first
                    basic_measurements = ['roof_facets', 'predominant_pitch', 'total_area', 'obstructions']
                    for key in basic_measurements:
                        if key in measurements and measurements[key] and measurements[key] != "N/A":
                            display_name = key.replace('_', ' ').title()
                            value = str(measurements[key])
                            formatted_parts.append(f"| {display_name} | {value} |")
                    
                    formatted_parts.append("")
                    
                    # Roof Components Table
                    roof_components = {}
                    for key, value in measurements.items():
                        if value and value != "N/A" and key not in basic_measurements:
                            if any(term in key.lower() for term in ['ridge', 'hip', 'valley', 'rake', 'eave', 'drip', 'flash']):
                                roof_components[key] = value
                    
                    if roof_components:
                        formatted_parts.append("## 🔧 Roof Components")
                        formatted_parts.append("| Component | Measurement | Details |")
                        formatted_parts.append("|-----------|-------------|---------|")
                        
                        for key, value in roof_components.items():
                            component = key.replace('_', ' ').title()
                            # Split value into measurement and details if possible
                            value_parts = str(value).split('(')
                            measurement = value_parts[0].strip()
                            details = '(' + value_parts[1] if len(value_parts) > 1 else ''
                            
                            formatted_parts.append(f"| {component} | {measurement} | {details} |")
                        
                        formatted_parts.append("")
                    
                    # Location & Coordinates
                    if 'coordinates' in measurements:
                        formatted_parts.append("## 📍 Location Information")
                        formatted_parts.append("| Property | Details |")
                        formatted_parts.append("|----------|---------|")
                        formatted_parts.append(f"| Address | {address} |")
                        formatted_parts.append(f"| Coordinates | {measurements['coordinates']} |")
                        formatted_parts.append("")
            
            # Additional Details Section
            if "properties" in data and data["properties"]:
                prop = data["properties"][0]
                if "additional_details" in prop and prop["additional_details"]:
                    formatted_parts.append("## 📝 Additional Details")
                    for detail in prop["additional_details"]:
                        formatted_parts.append(f"- {detail}")
                    formatted_parts.append("")
            
            # Images Section
            if "images_available" in data and data["images_available"]:
                formatted_parts.append("## 🖼️ Available Documentation")
                for img in data["images_available"]:
                    title = img.get('title', 'Unknown Image')
                    formatted_parts.append(f"- 📸 {title}")
                formatted_parts.append("")
            
            # Analysis Summary
            if "properties" in data and data["properties"]:
                formatted_parts.append("## 📈 Analysis Summary")
                formatted_parts.append("| Metric | Value |")
                formatted_parts.append("|--------|-------|")
                formatted_parts.append(f"| Properties Analyzed | {len(data['properties'])} |")
                
                # Extract key stats
                total_facets = 0
                pitch_info = []
                for prop in data["properties"]:
                    if "key_measurements" in prop:
                        measurements = prop["key_measurements"]
                        if "roof_facets" in measurements:
                            try:
                                facets = int(str(measurements["roof_facets"]).split()[0])
                                total_facets += facets
                            except:
                                pass
                        if "predominant_pitch" in measurements:
                            pitch_info.append(measurements["predominant_pitch"])
                
                if total_facets > 0:
                    formatted_parts.append(f"| Total Roof Facets | {total_facets} |")
                if pitch_info:
                    formatted_parts.append(f"| Pitch Variations | {', '.join(set(pitch_info))} |")
                
                formatted_parts.append("")
            
            # Footer with confidence
            if "confidence" in data:
                confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(data["confidence"], "⚪")
                formatted_parts.append(f"**{confidence_emoji} Analysis Confidence:** {data['confidence'].upper()}")
            
            if "notes" in data and data["notes"]:
                formatted_parts.append(f"**💡 Notes:** {data['notes']}")
            
            return "\n".join(formatted_parts)
            
        except json.JSONDecodeError:
            return json_response

    def print_search_results(self, query: str, results: List[Dict]) -> None:
        """
        Print hierarchical search results in a formatted way
        
        Args:
            query: Original search query
            results: Search results from hierarchical_search
        """
        print(f"\n🔍 Hierarchical Search Results for: '{query}'")
        print("=" * 80)
        
        if not results:
            print("❌ No results found.")
            return
        
        print(f"📊 Found {len(results)} relevant chunks from hierarchical search:\n")
        
        for i, result in enumerate(results, 1):
            print(f"--- Result {i} ---")
            print(f"📄 Document: {result.get('doc_id', 'N/A')}")
            print(f"🏠 Address: {result.get('doc_address', 'N/A')}")
            print(f"📋 Section: {result.get('section', 'N/A')}")
            print(f"🏷️  Type: {result.get('chunk_type', 'N/A')}")
            print(f"🆔 Chunk ID: {result.get('chunk_id', 'N/A')}")
            print(f"📏 Distance: {result.get('distance', 'N/A'):.4f}")
            
            # Display chunk content
            chunk_text = result.get('chunk_text', '')
            if len(chunk_text) > 300:
                chunk_text = chunk_text[:300] + "..."
            print(f"📝 Content: {chunk_text}")
            
            print()
        
        print("=" * 80)

    def answer_query(self, query: str, level1_limit: int = 1, level2_limit: int = 5, show_raw_results: bool = False) -> str:
        """
        Complete query answering pipeline: detect query type and route to appropriate search flow

        Note: For server/API use, use answer_query_with_raw_results() instead to get both LLM response and raw search results.

        Args:
            query: User's question
            level1_limit: Number of documents to retrieve from Level 1 (for hierarchical search)
            level2_limit: Number of chunks to retrieve from Level 2 (for hierarchical search)
            show_raw_results: Whether to print raw search results

        Returns:
            LLM-generated response string
        """
        # Detect query type and route to appropriate search strategy
        query_type = self.detect_query_type(query)

        if query_type == "property_specific":
            logger.info("🏠 Using hierarchical search flow (property-specific query)")
            results = self.search_hierarchical(query, level1_limit, level2_limit)
        else:
            logger.info("🌍 Using general search flow (cross-property query)")
            results = self.general_query_handler.search_general(query, limit=max(level2_limit * 2, 20))

        # Optionally show raw results
        if show_raw_results:
            if query_type == "property_specific":
                self.print_search_results(query, results)
            else:
                # For general search, use GeneralQueryHandler's print method
                self.general_query_handler._print_general_search_results(query, results)

        # Generate LLM response using the appropriate handler
        if query_type == "property_specific":
            llm_response = self.generate_llm_response(query, results)
        else:
            # For general queries, use the GeneralQueryHandler's response generation
            llm_response = self.general_query_handler.generate_general_response(query, results)

        return llm_response

    def answer_query_with_raw_results(self, query: str, level1_limit: int = 1, level2_limit: int = 5, show_raw_results: bool = False) -> Tuple[str, List[Dict]]:
        """
        Complete query answering pipeline that returns both LLM response and raw search results

        Args:
            query: User's question
            level1_limit: Number of documents to retrieve from Level 1 (for hierarchical search)
            level2_limit: Number of chunks to retrieve from Level 2 (for hierarchical search)
            show_raw_results: Whether to print raw search results

        Returns:
            Tuple of (LLM response string, raw search results)
        """
        # Detect query type and route to appropriate search strategy
        query_type = self.detect_query_type(query)

        if query_type == "property_specific":
            logger.info("🏠 Using hierarchical search flow (property-specific query)")
            results = self.search_hierarchical(query, level1_limit, level2_limit)
        else:
            logger.info("🌍 Using general search flow (cross-property query)")
            results = self.general_query_handler.search_general(query, limit=max(level2_limit * 2, 20))

        # Optionally show raw results
        if show_raw_results:
            if query_type == "property_specific":
                self.print_search_results(query, results)
            else:
                # For general search, use GeneralQueryHandler's print method
                self.general_query_handler._print_general_search_results(query, results)

        # Generate LLM response using the appropriate handler
        if query_type == "property_specific":
            llm_response = self.generate_llm_response(query, results)
        else:
            # For general queries, use the GeneralQueryHandler's response generation
            llm_response = self.general_query_handler.generate_general_response(query, results)

        return llm_response, results, query_type

    def show_collection_status(self):
        """Show the current status of both hierarchical collections"""
        logger.info("📊 Checking collection status...")
        
        collections = self.milvus_manager.list_collections()
        print("\n" + "="*60)
        print("🗂️  HIERARCHICAL RAG COLLECTION STATUS")
        print("="*60)
        
        # Check Level 1 collection
        if self.level1_collection_name in collections:
            level1_info = self.milvus_manager.get_collection_info(self.level1_collection_name)
            print(f"✅ Level 1 ({self.level1_collection_name}):")
            print(f"   📄 Documents: {level1_info.get('entity_count', 'unknown')}")
        else:
            print(f"❌ Level 1 ({self.level1_collection_name}): Not found")
        
        # Check Level 2 collection
        if self.level2_collection_name in collections:
            level2_info = self.milvus_manager.get_collection_info(self.level2_collection_name)
            print(f"✅ Level 2 ({self.level2_collection_name}):")
            print(f"   📝 Chunks: {level2_info.get('entity_count', 'unknown')}")
        else:
            print(f"❌ Level 2 ({self.level2_collection_name}): Not found")
        
        # Show other collections
        other_collections = [c for c in collections if c not in [self.level1_collection_name, self.level2_collection_name]]
        if other_collections:
            print(f"\n🔍 Other collections: {other_collections}")
        
        print("="*60)
    
    def clear_all_hierarchical_collections(self, confirm: bool = False):
        """
        Clear all hierarchical collections
        
        Args:
            confirm: Must be True to actually clear collections
        """
        if not confirm:
            logger.warning("⚠️  clear_all_hierarchical_collections called without confirmation")
            return
        
        logger.info("🧹 Clearing all hierarchical collections...")
        self.milvus_manager.clear_collection(self.level1_collection_name)
        self.milvus_manager.clear_collection(self.level2_collection_name)
        logger.info("✅ All hierarchical collections cleared")


def load_agentic_rag_output() -> List[Dict]:
    """
    Load the output from the Final_Chunks folder and convert to hierarchical format
    
    Returns:
        List of documents in hierarchical format
    """
    logger.info("📂 Loading chunks from Final_Chunks folder...")
    
    # Load the Final_Chunks directory
    final_chunks_dir = Path("Final_Chunks")
    if not final_chunks_dir.exists():
        logger.error("Final_Chunks directory not found")
        return []
    
    documents = []
    
    # Process each JSON file in Final_Chunks
    for chunk_file in final_chunks_dir.glob("*.json"):
        try:
            logger.info(f"Processing {chunk_file.name}")
            
            with open(chunk_file, 'r') as f:
                file_data = json.load(f)
            
            # Extract document ID from filename (e.g., RoofReport-44995431.json -> 44995431)
            doc_id = chunk_file.stem.replace("RoofReport-", "")
            source_file = chunk_file.name
            
            # Extract chunks from the file data
            # Handle both formats: direct array or {"text": [...], "table": [...], "image": [...]}
            chunks = []
            if isinstance(file_data, list):
                # Direct array format (current Final_Chunks structure)
                chunks.extend(file_data)
            else:
                # Wrapped format {"text": [...], ...}
                if "text" in file_data:
                    chunks.extend(file_data["text"])
                if "table" in file_data:
                    chunks.extend(file_data["table"])
                if "image" in file_data:
                    chunks.extend(file_data["image"])
            
            # Create document entry
            doc_entry = {
                "doc_id": doc_id,
                "source_file": source_file,
                "chunks": chunks
            }
            
            documents.append(doc_entry)
            logger.info(f"Loaded {len(chunks)} chunks from {source_file}")
            
        except Exception as e:
            logger.error(f"Error loading {chunk_file}: {str(e)}")
            continue
    
    logger.info(f"📊 Loaded {len(documents)} documents with total chunks")
    return documents


def main(query: str = None, build_index: bool = True, show_raw: bool = False, raw_only: bool = False, json_only: bool = False):
    """
    Main function to build hierarchical indices and demonstrate search
    
    Args:
        query: Optional query string to search for
        build_index: Whether to build the indices (set to False if already built)
        show_raw: Whether to show raw search results along with LLM response
        raw_only: Whether to show only raw search results (no LLM response)
        json_only: Whether to show only raw JSON response from LLM (no formatting)
    """
    logger.info("🚀 Starting Hierarchical RAG System")
    
    # Load agentic RAG output
    documents = load_agentic_rag_output()
    
    if not documents:
        logger.error("No documents found. Please ensure Final_Chunks folder contains JSON files.")
        return
    
    # Initialize hierarchical RAG system
    hierarchical_rag = HierarchicalRAG()
    
    # Show initial collection status
    hierarchical_rag.show_collection_status()
    
    if build_index:
        # Build hierarchical indices (clear existing to prevent duplicates)
        logger.info("🏗️ Building hierarchical indices with duplicate prevention...")
        hierarchical_rag.build_level1_index(documents, clear_existing=True)
        hierarchical_rag.build_level2_index(documents, clear_existing=False)  # Level 1 already cleared
        
        # Show final collection status
        hierarchical_rag.show_collection_status()
    else:
        logger.info("⏭️ Skipping index building (using existing indices)")
    
    # Handle query input
    if not query:
        logger.info("No query provided. Use --query to specify a search query.")
        return

    # Single query provided as parameter
    test_queries = [query]
    
    if raw_only:
        logger.info("🔍 Testing hierarchical search (raw results only)...")
        for test_query in test_queries:
            print(f"\n{'='*80}")
            results = hierarchical_rag.search_hierarchical(test_query, level1_limit=1, level2_limit=5)
            hierarchical_rag.print_search_results(test_query, results)
            time.sleep(1)  # Small delay between queries
    else:
        logger.info("🔍 Testing hierarchical search with LLM response generation...")
        for test_query in test_queries:
            print(f"\n{'='*80}")
            print(f"🤔 Question: {test_query}")
            print("=" * 80)
            
            # Get LLM-generated response
            llm_response = hierarchical_rag.answer_query(test_query, level1_limit=2, level2_limit=5, show_raw_results=show_raw)
            
            print("🤖 AI Assistant Response:")
            print("-" * 40)
            
            if json_only:
                # Show raw JSON response
                print(llm_response)
            else:
                # Try to format as JSON if possible, otherwise display as-is
                try:
                    formatted_response = hierarchical_rag.format_json_response(llm_response)
                    print(formatted_response)
                except:
                    print(llm_response)
            
            print("=" * 80)
            
            time.sleep(2)  # Small delay between queries
    
    logger.info("✅ Hierarchical RAG system demonstration completed!")


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Hierarchical RAG System with Two-Level Indices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Build indices and run default queries (clears existing collections)
        python src/hierarchical_rag_working.py
        
        # Search with a specific query (will build indices first with duplicate prevention)
        python src/hierarchical_rag_working.py --query "What is the roof area?"
        
        # Search without rebuilding indices (faster if indices already exist)
        python src/hierarchical_rag_working.py --query "What are the obstructions?" --no-build
        
        # Show current collection status
        python src/hierarchical_rag_working.py --show-status
        
        # Clear all collections (use with caution!)
        python src/hierarchical_rag_working.py --clear-collections
        
        # Interactive mode - just build indices, then you can call functions directly
        python src/hierarchical_rag_working.py --build-only
        """
    )
    
    parser.add_argument(
        "--query", "-q",
        type=str,
        help="Query to search for in the hierarchical RAG system"
    )
    
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip building indices (use existing indices)"
    )
    
    parser.add_argument(
        "--build-only",
        action="store_true", 
        help="Only build indices, don't run any queries"
    )
    
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Show raw search results along with LLM response"
    )
    
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Show only raw search results (no LLM response)"
    )
    
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Show only raw JSON response from LLM (no formatting)"
    )
    
    parser.add_argument(
        "--clear-collections",
        action="store_true",
        help="Clear all hierarchical collections and exit (use with caution!)"
    )
    
    parser.add_argument(
        "--show-status",
        action="store_true",
        help="Show collection status and exit"
    )
    
    args = parser.parse_args()
    
    # Handle clear collections mode
    if args.clear_collections:
        hierarchical_rag = HierarchicalRAG()
        hierarchical_rag.show_collection_status()
        print("\n⚠️  WARNING: This will clear all hierarchical collections!")
        confirm = input("Type 'yes' to confirm: ").strip().lower()
        if confirm == 'yes':
            hierarchical_rag.clear_all_hierarchical_collections(confirm=True)
            hierarchical_rag.show_collection_status()
        else:
            print("❌ Operation cancelled.")
        sys.exit(0)
    
    # Handle show status mode
    if args.show_status:
        hierarchical_rag = HierarchicalRAG()
        hierarchical_rag.show_collection_status()
        sys.exit(0)
    
    # Handle build-only mode
    if args.build_only:
        main(query=None, build_index=True)
        print("\n✅ Indices built successfully! You can now run queries with --no-build flag.")
        sys.exit(0)
    
    # Run main function with parsed arguments
    build_index = not args.no_build
    main(query=args.query, build_index=build_index, show_raw=args.show_raw, raw_only=args.raw_only, json_only=args.json_only)
