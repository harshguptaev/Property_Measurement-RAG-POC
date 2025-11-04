"""
Working Hierarchical RAG System with Two-Level Indices
Simplified version that works with MilvusClient without complex index creation
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import time
import argparse
import sys

import boto3
from pymilvus import MilvusClient
from tqdm import tqdm

# Import query router
from src.query_router import QueryRouter, QueryAnalysis
from src.property_rag_status_dao import PropertyRAGStatusDAO

# Import LLM response handlers
from llm_response_handlers import LLMResponseHandlers


# Simple MilvusCollectionManager class
class MilvusCollectionManager:
    """Simple collection manager for Milvus operations"""

    def __init__(self, uri: str):
        self.client = MilvusClient(uri=uri)

    def create_collection_safely(self, collection_name: str, embedding_dim: int, metric_type: str = "COSINE",
                                clear_existing: bool = False, index_params: Dict = None):
        """Create a collection safely, optionally clearing existing data"""
        from pymilvus import CollectionSchema, FieldSchema, DataType

        if clear_existing and self.client.has_collection(collection_name):
            logger.info(f"Dropping existing collection: {collection_name}")
            self.client.drop_collection(collection_name)

        if not self.client.has_collection(collection_name):
            logger.info(f"Creating collection: {collection_name} with dimension {embedding_dim}, metric {metric_type}")

            # Define schema based on collection type
            if "level1" in collection_name.lower():
                # Level 1 Schema - Property level data
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
                    FieldSchema(name="property_id", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="address", dtype=DataType.VARCHAR, max_length=512),
                    FieldSchema(name="geometry", dtype=DataType.JSON),
                    FieldSchema(name="metadata", dtype=DataType.JSON)
                ]
            elif "level2" in collection_name.lower():
                # Level 2 Schema - Chunk level data
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=1536),
                    FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="property_id", dtype=DataType.VARCHAR, max_length=64),
                    FieldSchema(name="section", dtype=DataType.VARCHAR, max_length=128),
                    FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=32),
                    FieldSchema(name="chunk_text", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="metadata", dtype=DataType.JSON)
                ]
            else:
                # Fallback to simple schema
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim)
                ]

            schema = CollectionSchema(
                fields=fields,
                description=f"Schema for {collection_name} collection"
            )

            # Create collection with schema (without index_params)
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema
            )

            # Note: Index will be created after data insertion for better performance
            # This is done in the build methods
    
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
    
    def __init__(self, region_name: str = "us-east-1", model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"):
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

        # Initialize query router
        self.query_router = QueryRouter(region_name=region_name, model_id="anthropic.claude-3-haiku-20240307-v1:0")

        # Initialize LLM response handlers
        self.llm_handlers = LLMResponseHandlers(self.bedrock_client, self.model_id)

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

        # Embedding dimensions: Both L1 and L2 use 1536 (full semantic embeddings)
        # Default Titan embedding dimension is 1536
        self.embedding_dim = 1536  # Default Titan embedding dimension

        logger.info(f"🔧 Initialized Hierarchical RAG with model: {model_id}")
        logger.info(f"🔧 Level 1 collection: {self.level1_collection_name}")
        logger.info(f"🔧 Level 2 collection: {self.level2_collection_name}")
    
    def titan_embed_text(self, text: str, target_dim: int = 1536) -> List[float]:
        """
        Get text embeddings from Amazon Titan Embed Text v1

        Args:
            text: Text to embed
            target_dim: Target dimension for the embedding (always 1536)

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
            # All collections now use full 1536 dimensions
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
                "anthropic_version": "anthropic.claude-3-haiku-20240307-v1:0",
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
    
    def create_milvus_collections(self, clear_existing: bool = False):
        """
        Create Milvus collections using the collection manager for safe handling
        
        Args:
            clear_existing: Whether to clear existing collections to prevent duplicates
        """
        logger.info("🔧 Creating Milvus collections for hierarchical indexing")
        
        if clear_existing:
            logger.info("🗑️  Clearing existing collections to prevent duplicates...")
        
        # Level 1 Collection Configuration (Addresses, Geometry, Filters)
        # - Fast lookup by address/coordinates + full semantic search
        # - ~200K vectors, 1536 dimensions, COSINE metric
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

        # Create Level 1 collection (Document summaries) - 1536 dimensions
        self.milvus_manager.create_collection_safely(
            collection_name=self.level1_collection_name,
            embedding_dim=1536,  # L1 uses full embeddings for address/geometry lookup
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

        # Note: Collections will be loaded after indexes are created in build methods

        logger.info(f"✅ Level 1 collection ready: {self.level1_collection_name}")
        logger.info(f"✅ Level 2 collection ready: {self.level2_collection_name}")
        
        # Show collection status
        level1_info = self.milvus_manager.get_collection_info(self.level1_collection_name)
        level2_info = self.milvus_manager.get_collection_info(self.level2_collection_name)
        logger.info(f"📊 Level 1 entities: {level1_info.get('entity_count', 0)}")
        logger.info(f"📊 Level 2 entities: {level2_info.get('entity_count', 0)}")
    
    def build_level1_index(self, documents_data: List[Dict], clear_existing: bool = False) -> None:
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

                # Generate embedding for the address (for vector search) - L1 uses 1536 dimensions
                address_embedding = self.titan_embed_text(address, target_dim=1536)

                level1_data.append({
                    "id": i,
                    "vector": address_embedding,
                    "property_id": property_id,
                    "address": address,
                    "geometry": {
                        "type": "Point",
                        "coordinates": [longitude, latitude]
                    },
                    "metadata": {
                        "report_id": report_id,
                        "pdf_filename": pdf_filename,
                        "child_chunk_ids": child_chunk_ids
                    }
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

            # Create index after data insertion
            try:
                from pymilvus.milvus_client.index import IndexParams
                index_params = IndexParams()
                index_params.add_index(
                    field_name="vector",
                    index_type="HNSW",
                    metric_type="COSINE",
                    params={"M": 24, "efConstruction": 100}
                )
                self.milvus_client.create_index(
                    collection_name=self.level1_collection_name,
                    index_params=index_params
                )
                logger.info("✅ Created index for Level 1 collection")
            except Exception as e:
                logger.warning(f"⚠️ Failed to create index for Level 1: {str(e)}")

            # Load collection after index creation
            try:
                self.milvus_client.load_collection(collection_name=self.level1_collection_name)
                logger.info("✅ Level 1 collection loaded")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load Level 1 collection: {str(e)}")

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
                        semantic_text = self.generate_semantic_text(section, chunk_type, data)
                        # Generate embedding for chunk text - all collections use 1536 dimensions
                        chunk_embedding = self.titan_embed_text(semantic_text, target_dim=1536)

                        level2_data.append({
                            "id": chunk_counter,
                            "vector": chunk_embedding,
                            "chunk_id": chunk_id,
                            "property_id": property_id,
                            "section": section,
                            "type": chunk_type,
                            "chunk_text": chunk_text,
                            "metadata": {
                                "semantic_text": semantic_text,
                                "data": data  # Store the original structured data
                            }
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

            # Create index after data insertion
            try:
                from pymilvus.milvus_client.index import IndexParams
                index_params = IndexParams()
                index_params.add_index(
                    field_name="vector",
                    index_type="HNSW",
                    metric_type="COSINE",
                    params={"M": 32, "efConstruction": 200}
                )
                self.milvus_client.create_index(
                    collection_name=self.level2_collection_name,
                    index_params=index_params
                )
                logger.info("✅ Created index for Level 2 collection")
            except Exception as e:
                logger.warning(f"⚠️ Failed to create index for Level 2: {str(e)}")

            # Load collection after index creation
            try:
                self.milvus_client.load_collection(collection_name=self.level2_collection_name)
                logger.info("✅ Level 2 collection loaded")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load Level 2 collection: {str(e)}")

            logger.info(f"✅ Level 2 Index built with {len(level2_data)} children chunks")
        else:
            logger.warning("No data to insert into Level 2 Index")
    
    def detect_query_type(self, query: str) -> str:
        """
        Detect if a query is property-specific or general.

        Args:
            query: The search query

        Returns:
            "property_specific" - all queries are treated as property-specific
        """
        # All queries are treated as property-specific
        return "property_specific"

    def search_hierarchical(self, query: str, address: Optional[str] = None, relevant_sections: List[str] = [], level1_limit: int = 1, level2_limit: int = 5) -> List[Dict]:
        """
        Perform hierarchical search: Level 1 → Level 2

        Args:
            query: Search query
            address: Optional specific address to search for
            level1_limit: Number of documents to retrieve from Level 1
            level2_limit: Number of chunks to retrieve from Level 2

        Returns:
            List of relevant chunks with metadata
        """
        logger.info(f"🔍 Starting hierarchical search for: '{query}' (Address: {address})")

        try:
            # If address is provided, first try exact scalar search
            if address:
                logger.info(f"🔍 Checking for exact address match: '{address}'")
                exact_address_results = self.milvus_client.query(
                    collection_name=self.level1_collection_name,
                    filter=f'address == "{address}"',
                    output_fields=["property_id", "address", "metadata"],
                    limit=1
                )

                if exact_address_results and len(exact_address_results) > 0:
                    logger.info(f"✅ Found exact address match: {address}")
                    # Continue with Level 2 search for this exact match
                    exact_match = exact_address_results[0]
                    property_id = exact_match.get("property_id")
                    metadata = exact_match.get("metadata", {})
                    child_chunk_ids = metadata.get("child_chunk_ids", [])

                    if not child_chunk_ids:
                        logger.warning("No child chunks found for exact address match")
                        return []

                    # Get Level 2 results for this specific property
                    query_vec_l2 = self.titan_embed_text(query, target_dim=1536)
                    level2_search_params = {
                        "metric_type": "COSINE",
                        "params": {"ef": 96}
                    }

                    # Build filter for relevant sections if specified
                    section_filter = ""
                    if relevant_sections:
                        # Use LIKE queries for prefix matching
                        like_conditions = [f'section like "{prefix}%"' for prefix in relevant_sections]
                        section_filter = f' && ({ " || ".join(like_conditions) })'

                    res2 = self.milvus_client.search(
                        collection_name=self.level2_collection_name,
                        data=[query_vec_l2],
                        limit=level2_limit,
                        filter=f'property_id == "{property_id}"{section_filter}',
                        output_fields=["chunk_text", "section", "property_id", "type", "chunk_id", "metadata"],
                        search_params=level2_search_params
                    )

                    if not res2 or not res2[0]:
                        logger.warning("No Level 2 results found for exact address match")
                        return []

                    # Format results for exact address match
                    final_results = []
                    for result in res2[0]:
                        chunk_info = {
                            "chunk_id": result.get("chunk_id"),
                            "property_id": property_id,
                            "section": result.get("section"),
                            "chunk_type": result.get("type"),
                            "chunk_text": result.get("chunk_text"),
                            "data": result.get("metadata", {}).get("data", {}),  # Get data from metadata
                            "distance": result.get("distance", 0),
                            "doc_address": address,
                            "report_id": metadata.get("report_id"),
                            "pdf_filename": metadata.get("pdf_filename")
                        }
                        final_results.append(chunk_info)

                    final_results.sort(key=lambda x: x["distance"])
                    logger.info(f"✅ Found {len(final_results)} chunks for exact address match")
                    return final_results
                else:
                    logger.info(f"❌ No exact address match found for: '{address}'")
                    # Fall back to vector search for similar addresses
                    logger.info("🔄 Falling back to vector search for similar addresses")

            # Step 1: Embed query for Level 1 (1536 dimensions for address/geometry lookup)
            query_vec_l1 = self.titan_embed_text(query, target_dim=1536)

            # Step 2: Embed query for Level 2 (1536 dimensions for semantic search) - all collections use 1536 dimensions
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
            # If we have an address but no exact match, search for 6 similar addresses
            # Otherwise, use the default limit
            search_limit = 6 if address else level1_limit
            logger.info(f"📊 Searching Level 1 (Parent Chunk Index) with limit {search_limit}...")

            res1 = self.milvus_client.search(
                collection_name=self.level1_collection_name,
                data=[query_vec_l1],
                limit=search_limit,
                output_fields=["property_id", "address", "metadata"],
                search_params=level1_search_params
            )
            
            if not res1 or not res1[0]:
                logger.warning("No results found in Level 1 Index")
                return []

            # If we had an address but no exact match was found, return similar addresses from Level 1
            if address:
                logger.info("📋 Returning similar addresses from Level 1 (no exact match found)")
                similar_addresses = []
                for hit in res1[0]:
                    # Transform address info into chunk-like format for display
                    address_info = {
                        "property_id": hit.get("property_id"),
                        "doc_id": hit.get("metadata", {}).get("report_id", "unknown"),  # Add doc_id
                        "chunk_id": f"addr_{hit.get('property_id')}",  # Create a chunk-like ID
                        "section": "Address Information",
                        "chunk_type": "address",  # Special type for addresses
                        "chunk_text": f"Similar Property Address: {hit.get('address', 'Unknown Address')}\nSimilarity Score: {(hit.get('distance', 0)):.4f}",
                        "data": {
                            "address": hit.get("address", "Unknown Address"),
                            "report_id": hit.get("metadata", {}).get("report_id"),
                            "pdf_filename": hit.get("metadata", {}).get("pdf_filename"),
                            "similarity_score": 1 - hit.get("distance", 0)
                        },
                        "doc_address": hit.get("address", "Unknown Address"),
                        "report_id": hit.get("metadata", {}).get("report_id"),
                        "pdf_filename": hit.get("metadata", {}).get("pdf_filename"),
                        "distance": hit.get("distance", 0)
                    }
                    similar_addresses.append(address_info)

                logger.info(f"✅ Found {len(similar_addresses)} similar addresses")
                return similar_addresses

            print(f"Fetching Chunks from Level 1 Index - Address matching")
            print(f"res1 ::: {res1}")
            # Get relevant chunk_ids
            retrieved_chunk_ids = []
            level1_docs = []

            for hit in res1[0]:
                # Extract data from the new Level 1 structure
                hit_address = hit.get("address", "Unknown Address")
                metadata = hit.get("metadata", {})

                doc_info = {
                    "property_id": hit.get("property_id"),
                    "address": hit_address,
                    "report_id": metadata.get("report_id"),
                    "pdf_filename": metadata.get("pdf_filename"),
                    "distance": hit.get("distance", 0)
                }
                level1_docs.append(doc_info)

                print(f"\n{'='*80}")
                print(f"property_id: {doc_info['property_id']}")
                print(f"address: {doc_info['address']}")
                print(f"report_id: {doc_info['report_id']}")
                print(f"pdf_filename: {doc_info['pdf_filename']}")
                print(f"distance: {doc_info['distance']}")

                # Get child chunk IDs from metadata
                child_chunk_ids = metadata.get("child_chunk_ids", [])
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

            # Build filter for relevant sections if specified
            section_filter = ""
            if relevant_sections:
                # Use LIKE queries for prefix matching
                like_conditions = [f'section like "{prefix}%"' for prefix in relevant_sections]
                section_filter = f' && ({ " || ".join(like_conditions) })'

            # Search Level 2 chunks filtered by property_id from Level 1 results
            res2 = self.milvus_client.search(
                collection_name=self.level2_collection_name,
                data=[query_vec_l2],
                limit=5,  # Use the level2_limit parameter
                filter=f'property_id in [{",".join(property_ids_quoted)}]{section_filter}',
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
                        "data": result.get("metadata", {}).get("data", {}),  # Get data from metadata
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

    def search_flow2(self, query: str, relevant_sections: List[str] = [], level1_limit: int = 10, level2_limit: int = 10) -> List[Dict]:
        """
        Perform Flow 2 search: Level 2 first (find relevant chunks) → Level 1 (get property details)

        This flow is used for non-property-specific queries where we want to find properties
        that match certain criteria (e.g., "find properties with area > 2000 sq ft").

        Args:
            query: Search query (criteria for finding properties)
            level1_limit: Number of documents to retrieve from Level 1 (after finding relevant chunks)
            level2_limit: Number of chunks to retrieve from Level 2

        Returns:
            List of relevant chunks with property metadata
        """
        logger.info(f"🔍 Starting Flow 2 search for: '{query}'")

        try:
            # Step 1: Embed query for Level 2 (1536 dimensions for semantic search) - all collections use 1536 dimensions
            query_vec_l2 = self.titan_embed_text(query, target_dim=1536)

            # Level 2 search parameters
            level2_search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 96}
            }

            # Build filter for relevant sections if specified
            section_filter = ""
            if relevant_sections:
                # Use LIKE queries for prefix matching
                like_conditions = [f'section like "{prefix}%"' for prefix in relevant_sections]
                section_filter = f' && ({ " || ".join(like_conditions) })'

            # Step 2: Search Level 2 (Chunks) - find chunks that match the criteria
            logger.info("📊 Searching Level 2 (Chunk Index) for property criteria...")
            res2 = self.milvus_client.search(
                collection_name=self.level2_collection_name,
                data=[query_vec_l2],
                limit=20,
                filter=section_filter.lstrip(' && ') if section_filter else "",  # Remove leading '&&' if present
                output_fields=["chunk_text", "section", "property_id", "type", "chunk_id", "metadata"],
                search_params=level2_search_params
            )

            if not res2 or not res2[0]:
                logger.warning("No results found in Level 2 Index")
                return []

            # Step 3: Collect relevant property IDs from Level 2 results
            relevant_property_ids = []
            level2_chunks = []

            print(f"Fetching Chunks from Level 2 Index")
            print(f"\n{'='*80}")

            for result in res2[0]:
                entity = result.get("entity", {})
                property_id = entity.get("property_id")
                if property_id and property_id not in relevant_property_ids:
                    relevant_property_ids.append(property_id)

                chunk_info = {
                    "chunk_id": entity.get("chunk_id"),
                    "property_id": property_id,
                    "section": entity.get("section"),
                    "chunk_type": entity.get("type"),
                    "chunk_text": entity.get("chunk_text"),
                    "metadata": entity.get("metadata", {}),
                    "distance": result.get("distance", 0)
                }
                level2_chunks.append(chunk_info)

            logger.info(f"📋 Found {len(level2_chunks)} relevant chunks from {len(relevant_property_ids)} properties")

            # Step 4: Search Level 1 to get property details for relevant properties
            logger.info("📊 Searching Level 1 (Property Index) for property details...")

            # Get property details from Level 1 for all relevant properties found in Level 2
            level1_docs = []
            if relevant_property_ids:
                # Create filter for property IDs found in Level 2
                property_ids_quoted = [f'"{pid}"' for pid in relevant_property_ids]
                property_filter = f'property_id in [{",".join(property_ids_quoted)}]'

                # Use a simple vector for the search (we're filtering by property_id anyway)
                level1_query_vec = self.titan_embed_text("property", target_dim=1536)

                

                # Search Level 1 filtered by property IDs from Level 2 results
                res1 = self.milvus_client.search(
                    collection_name=self.level1_collection_name,
                    data=[level1_query_vec],
                    limit=10,
                    filter=property_filter,
                    output_fields=["property_id", "address", "metadata"],
                )

                if res1 and res1[0]:
                    for hit in res1[0]:
                        entity = hit.get("entity", {})
                        prop_id = entity.get("property_id")
                        address = entity.get("address", "Unknown Address")
                        metadata = entity.get("metadata", {})

                        doc_info = {
                            "property_id": prop_id,
                            "address": address,
                            "report_id": metadata.get("report_id"),
                            "pdf_filename": metadata.get("pdf_filename"),
                            "distance": hit.get("distance", 0)
                        }
                        level1_docs.append(doc_info)

            # Step 5: Combine results - attach property details to chunks
            final_results = []

            for chunk in level2_chunks:
                chunk_property_id = chunk["property_id"]

                # Find matching property details
                property_details = None
                for doc_info in level1_docs:
                    if doc_info["property_id"] == chunk_property_id:
                        property_details = doc_info
                        break

                # Attach property details to chunk
                if property_details:
                    chunk_with_property = chunk.copy()
                    chunk_with_property["doc_address"] = property_details["address"]
                    chunk_with_property["report_id"] = property_details["report_id"]
                    chunk_with_property["pdf_filename"] = property_details["pdf_filename"]
                    final_results.append(chunk_with_property)

            # Sort by relevance (distance)
            final_results.sort(key=lambda x: x["distance"])

            # Limit final results
            final_results = final_results[:level2_limit]

            print(f"Total chunks found: {len(level2_chunks)}")
            print(f"Properties found: {len(level1_docs)}")
            print(f"Final results: {len(final_results)}")
            print("=" * 80)

            logger.info(f"✅ Flow 2 search completed. Found {len(final_results)} relevant chunks from {len(level1_docs)} properties")
            return final_results

        except Exception as e:
            logger.error(f"Error during Flow 2 search: {str(e)}")
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
        Complete query answering pipeline: analyze query with router and route to appropriate search flow

        Note: For server/API use, use answer_query_with_raw_results() instead to get both LLM response and raw search results.

        Args:
            query: User's question
            level1_limit: Number of documents to retrieve from Level 1 (for hierarchical search)
            level2_limit: Number of chunks to retrieve from Level 2 (for hierarchical search)
            show_raw_results: Whether to print raw search results

        Returns:
            LLM-generated response string
        """
        # Analyze query with router
        analysis = self.query_router.analyze_query(query)
        
        logger.info(f"🔍 Query Analysis: Flow {analysis.flow}, Address: {analysis.address}, Query: {analysis.query}, Relevant Sections: {analysis.relevant_sections}")

        # Route to appropriate search flow
        if analysis.flow == "1":
            logger.info("🏠 Using Flow 1: Property-specific hierarchical search")
            results = self.search_hierarchical(query=analysis.query, address=analysis.address, relevant_sections=analysis.relevant_sections, level1_limit=level1_limit, level2_limit=level2_limit)
        elif analysis.flow == "2":
            logger.info("🔍 Using Flow 2: Non-property-specific criteria search")
            # For Flow 2, we want to find multiple properties, so use higher limits
            flow2_level1_limit = max(level1_limit, 10)  # At least 10 properties for Flow 2
            flow2_level2_limit = max(level2_limit, 20)  # At least 20 chunks for Flow 2
            results = self.search_flow2(query=analysis.query, relevant_sections=analysis.relevant_sections, level1_limit=flow2_level1_limit, level2_limit=flow2_level2_limit)
        else:
            results = self.search_flow3(query=analysis.query, address = analysis.address, relevant_sections=analysis.relevant_sections)

        # Optionally show raw results
        if show_raw_results:
            self.print_search_results(query, results)

        # Generate LLM response
        llm_response = self.llm_handlers.generate_llm_response(query, results)

        return llm_response

    def answer_query_with_raw_results(self, query: str, level1_limit: int = 1, level2_limit: int = 5 ) -> Tuple[str, List[Dict], QueryAnalysis]:
        """
        Complete query answering pipeline that returns LLM response, raw search results, and query analysis

        Args:
            query: User's question
            level1_limit: Number of documents to retrieve from Level 1 (for hierarchical search)
            level2_limit: Number of chunks to retrieve from Level 2 (for hierarchical search)

        Returns:
            Tuple of (LLM response string, raw search results, QueryAnalysis object)
        """
        # Analyze query with router
        analysis = self.query_router.analyze_query(query)
        logger.info(f"🔍 Query Analysis: Flow {analysis.flow}, Address: {analysis.address} Relevant Sections: {analysis.relevant_sections}")

        # Route to appropriate search flow
        if analysis.flow == "1":
            logger.info("🏠 Using Flow 1: Property-specific hierarchical search")
            results = self.search_hierarchical(analysis.query, analysis.address, analysis.relevant_sections, level1_limit, level2_limit)
        elif analysis.flow == "2":
            logger.info("🔍 Using Flow 2: Non-property-specific criteria search")
            # For Flow 2, we want to find multiple properties, so use higher limits
            flow2_level1_limit = max(level1_limit, 10)  # At least 10 properties for Flow 2
            flow2_level2_limit = max(level2_limit, 10)  # At least 20 chunks for Flow 2
            results = self.search_flow2(query=analysis.query, relevant_sections=analysis.relevant_sections, level1_limit=flow2_level1_limit, level2_limit=flow2_level2_limit)
        elif analysis.flow == "3":
            logger.info("🔍 Using Flow 3: Address not found query")
            results = self.search_flow3(query=analysis.query, address=analysis.address, relevant_sections=analysis.relevant_sections)
            results = []
        else:
            # Fallback to hierarchical search
            logger.warning(f"Unknown flow {analysis.flow}, falling back to hierarchical search")
            results = self.search_hierarchical(query=analysis.query, address=analysis.address, relevant_sections=analysis.relevant_sections, level1_limit=level1_limit, level2_limit=level2_limit)

        self.print_search_results(query, results)

        # Use different response generation based on flow
        if analysis.flow == "1":
            llm_response = self.llm_handlers.generate_flow1_response(query, results)
        elif analysis.flow == "2":
            llm_response = self.llm_handlers.generate_flow2_response(query, results)
        elif analysis.flow == "3":
           llm_response = "I'm sorry, I currently don't have the ability to search for addresses that don't exist. Please try again with a different address or query."
        else:
            # Fallback to generic response
            llm_response = self.llm_handlers.generate_flow1_response(query, results)

        return llm_response, results, analysis

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
    

    def search_flow3(self, query: str, address: Optional[str] = None, relevant_sections: List[str] = []) -> List[Dict]:
        """
        Perform Flow 3 search: Search for properties that match the query and address
        """
        logger.info(f"🔍 Starting Flow 3 search for: '{query}' (Address: {address})")
        
        # For Flow 3, we trigger the image + outline generation pipeline for the given address
        if not address:
            logger.warning("Flow 3 requires an address. None provided.")
            return []

        try:
            from flow3 import run_flow_for_address
        except Exception as e:
            logger.error(f"Failed to import flow3 runner: {e}")
            return []

        try:
            summary = run_flow_for_address(address)
            # Return as a single-result list to match expected return type
            return [summary]
        except Exception as e:
            logger.error(f"Error running flow3 for address '{address}': {e}")
            return []

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
            
            # Extract document ID from filename (e.g., report_67668772.json -> 67668772)
            doc_id = chunk_file.stem.replace("report_", "")
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

    # Update database status for processed reports after loading all documents
    logger.info("Updating database status for processed reports...")
    updated_reports = set()

    # Extract unique report IDs from the processed files
    for doc in documents:
        source_file = doc.get('source_file', '')
        if 'report_' in source_file:
            try:
                report_id_match = re.search(r'report_(\d+)', source_file)
                if report_id_match:
                    report_id = report_id_match.group(1)
                    if report_id not in updated_reports:
                        # Get the database record ID by report_id first
                        existing_records = PropertyRAGStatusDAO.get_records_by_report_id(report_id)
                        if existing_records and len(existing_records) > 0:
                            record_id = existing_records[0]['id']  # Get the actual database record ID

                            # Update chunking status to completed
                            success_chunking = PropertyRAGStatusDAO.update_chunking_status(record_id, "completed")
                            # Update vector save status to completed
                            success_vector = PropertyRAGStatusDAO.update_vector_save_status(record_id, "completed")
                            # Update final status to success
                            success_final = PropertyRAGStatusDAO.update_final_status(record_id, "success")
                        else:
                            # No database record found
                            success_chunking = False
                            success_vector = False
                            success_final = False
                            logger.warning(f"No database record found for report {report_id}")

                        if success_chunking and success_vector and success_final:
                            logger.info(f"Updated database status for report {report_id}")
                            updated_reports.add(report_id)
                        elif success_chunking is False and success_vector is False and success_final is False:
                            # Either database not available or no record found - mark as processed
                            logger.info(f"Database not available or no record found - marking report {report_id} as processed")
                            updated_reports.add(report_id)
                        else:
                            logger.warning(f"Failed to update database status for report {report_id}")
            except Exception as db_error:
                logger.warning(f"Error updating database for report {source_file}: {db_error}")

    logger.info(f"Updated database status for {len(updated_reports)} reports")

    logger.info(f"📊 Loaded {len(documents)} documents with total chunks")
    return documents


def main(build_index: bool = True):
    """
    Main function to build hierarchical indices

    Args:
        build_index: Whether to build the indices (set to False if already built)
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

    logger.info("✅ Hierarchical RAG system setup completed!")


if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Hierarchical RAG System with Two-Level Indices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Build indices (clears existing collections)
        python src/hierarchical_rag_working.py

        # Build indices without clearing existing collections
        python src/hierarchical_rag_working.py --no-build

        # Show current collection status
        python src/hierarchical_rag_working.py --show-status

        # Clear all collections (use with caution!)
        python src/hierarchical_rag_working.py --clear-collections

        # Just build indices without clearing, then exit
        python src/hierarchical_rag_working.py --build-only
        """
    )
    
    parser.add_argument(
        "--no-build",
        action="store_true",
        help="Skip building indices (use existing indices)"
    )

    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only build indices, don't run queries"
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
        main(build_index=True)
        print("\n✅ Indices built successfully! You can now use the system.")
        sys.exit(0)

    # Run main function with parsed arguments
    build_index = not args.no_build
    main(build_index=build_index)
