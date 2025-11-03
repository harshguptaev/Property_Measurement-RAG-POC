#!/usr/bin/env python3
"""
Run Create Embeddings - Hierarchical RAG System
Main script to create hierarchical embeddings for property measurement data.
Supports both Level 2 chunk indexing (from input_data JSON files) and image embeddings.
Implements the same Level 2 indexing as hierarchical_rag_working 1.py for efficient RAG retrieval.
"""

import os
import sys
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import argparse

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import boto3
from pymilvus import MilvusClient
from tqdm import tqdm

from milvus_embeddings import MilvusEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
            if "level2" in collection_name.lower():
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

class HierarchicalEmbeddingsController:
    """
    Hierarchical Embeddings Controller for Level 2 (Chunk-level) processing
    """

    def __init__(self, region_name: str = "us-east-1", model_id: str = "anthropic.claude-3-haiku-20240307-v1:0"):
        """
        Initialize the Hierarchical Embeddings Controller
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

        # Collection name for Level 2 (same as hierarchical_rag_working 1.py)
        self.level2_collection_name = "hierarchical_level2"

        # Embedding dimensions: Use 1536 (full semantic embeddings)
        self.embedding_dim = 1536  # Default Titan embedding dimension

        logger.info(f"🔧 Initialized Hierarchical Embeddings Controller with model: {model_id}")
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
            return embedding

        except Exception as e:
            logger.error(f"Error generating text embedding: {str(e)}")
            raise

    def create_level2_collection(self, clear_existing: bool = False):
        """
        Create Level 2 collection using the collection manager

        Args:
            clear_existing: Whether to clear existing collection to prevent duplicates
        """
        logger.info("🔧 Creating Level 2 collection for chunk indexing")

        if clear_existing:
            logger.info("🗑️  Clearing existing Level 2 collection to prevent duplicates...")

        # Level 2 Collection Configuration (Semantic Chunks, Embeddings)
        level2_index_params = {
            "index_type": "HNSW",
            "metric_type": "COSINE",
            "params": {
                "M": 32,  # Higher degree for better recall in L2
                "efConstruction": 200  # Higher construction effort for L2
            }
        }

        # Create Level 2 collection (Chunks) - 1536 dimensions
        self.milvus_manager.create_collection_safely(
            collection_name=self.level2_collection_name,
            embedding_dim=1536,  # L2 uses full Titan embeddings for semantic search
            metric_type="COSINE",
            clear_existing=clear_existing,
            index_params=level2_index_params
        )

        logger.info(f"✅ Level 2 collection ready: {self.level2_collection_name}")

        # Show collection status
        level2_info = self.milvus_manager.get_collection_info(self.level2_collection_name)
        logger.info(f"📊 Level 2 entities: {level2_info.get('entity_count', 0)}")

    def build_level2_index(self, documents_data: List[Dict], clear_existing: bool = False) -> None:
        """
        Build Level 2 Index (Children Chunks Index)

        Args:
            documents_data: List of documents with their chunks
            clear_existing: Whether to clear existing collections
        """
        logger.info("🏗️ Building Level 2 Index (Children Chunks Index)")

        # Honor clear_existing by dropping and recreating the collection
        if clear_existing and self.milvus_client.has_collection(self.level2_collection_name):
            logger.info("🗑️  Clearing existing Level 2 collection before insert")
            self.milvus_manager.clear_collection(self.level2_collection_name)
            self.create_level2_collection(clear_existing=False)
        elif not self.milvus_client.has_collection(self.level2_collection_name):
            logger.warning("Level 2 collection doesn't exist, creating it...")
            self.create_level2_collection(clear_existing=False)

        level2_data = []
        chunk_counter = 0

        for doc in tqdm(documents_data, desc="Building Level 2 Index"):
            try:
                doc_id = doc.get("doc_id", "unknown")
                chunks = doc.get("chunks", [])
                # Derive a stable numeric prefix from doc_id to avoid primary-key collisions across runs
                try:
                    doc_id_prefix = int(str(doc_id))
                except Exception:
                    doc_id_prefix = abs(hash(str(doc_id))) % 1_000_000_000  # 9 digits max

                # Process all chunks (including the first property metadata chunk for Level 2)
                for chunk in chunks:
                    try:
                        # Extract chunk information in the specified format
                        chunk_id = chunk.get("chunk_id", f"chunk_{chunk_counter}")
                        section = chunk.get("section", "Unknown Section")
                        chunk_type = chunk.get("type", "text")

                        # Get property_id from the chunk (should be set)
                        property_id = chunk.get("property_id", f"PROP_{doc_id}")

                        # Get the data field directly (contains the structured data)
                        data = chunk.get("data", {})

                        # Skip parent/empty placeholder chunks
                        section_norm = (section or "").strip().lower()
                        if section_norm in ("property metadata", "address information"):
                            continue
                        if section_norm == "unknown section" and (not isinstance(data, dict) or len(data) == 0):
                            continue
                        if isinstance(data, dict):
                            prop_keys = {"address", "latitude", "longitude"}
                            if prop_keys.issubset(set(data.keys())) and len(data) <= 5:
                                # looks like a parent metadata record; skip from Level 2
                                continue

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

                        # Build a globally unique INT64 primary key per chunk
                        # Layout: <doc_prefix>*1_000_000 + local_counter (assumes <1M chunks/report)
                        unique_id = (doc_id_prefix * 1_000_000) + chunk_counter

                        level2_data.append({
                            "id": unique_id,
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

            # Ensure data is persisted and queryable
            try:
                self.milvus_client.flush(collection_name=self.level2_collection_name)
            except Exception:
                pass

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

            # Reload collection after index creation to refresh segments
            try:
                # Release first to force a fresh load of all segments
                try:
                    self.milvus_client.release_collection(collection_name=self.level2_collection_name)
                except Exception:
                    pass
                self.milvus_client.load_collection(collection_name=self.level2_collection_name)
                logger.info("✅ Level 2 collection loaded")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load Level 2 collection: {str(e)}")

            logger.info(f"✅ Level 2 Index built with {len(level2_data)} children chunks")
        else:
            logger.warning("No data to insert into Level 2 Index")

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

    def show_level2_collection_status(self):
        """Show the current status of Level 2 collection"""
        logger.info("📊 Checking Level 2 collection status...")

        collections = self.milvus_manager.list_collections()
        print("\n" + "="*60)
        print("🗂️  HIERARCHICAL LEVEL 2 COLLECTION STATUS")
        print("="*60)

        # Check Level 2 collection
        if self.level2_collection_name in collections:
            level2_info = self.milvus_manager.get_collection_info(self.level2_collection_name)
            print(f"✅ Level 2 ({self.level2_collection_name}):")
            print(f"   📝 Chunks: {level2_info.get('entity_count', 'unknown')}")
        else:
            print(f"❌ Level 2 ({self.level2_collection_name}): Not found")

        print("="*60)

    def clear_level2_collection(self, confirm: bool = False):
        """
        Clear Level 2 collection

        Args:
            confirm: Must be True to actually clear collection
        """
        if not confirm:
            logger.warning("⚠️  clear_level2_collection called without confirmation")
            return

        logger.info("🧹 Clearing Level 2 collection...")
        self.milvus_manager.clear_collection(self.level2_collection_name)
        logger.info("✅ Level 2 collection cleared")


def load_input_data_chunks() -> List[Dict]:
    """
    Load chunks from input_data directory JSON files and convert to hierarchical format

    Returns:
        List of documents in hierarchical format
    """
    logger.info("📂 Loading chunks from input_data directory...")

    # Load the input_data directory
    input_data_dir = Path("input_data")
    if not input_data_dir.exists():
        logger.error("input_data directory not found")
        return []

    documents = []

    # Process each subdirectory in input_data
    for report_dir in input_data_dir.iterdir():
        if report_dir.is_dir():
            try:
                # Extract report ID from directory name
                report_id = report_dir.name
                json_file_path = report_dir / f"{report_id}.json"

                if json_file_path.exists():
                    logger.info(f"Processing {report_id}")

                    with open(json_file_path, 'r') as f:
                        chunks_data = json.load(f)

                    # The JSON file contains chunks directly
                    if isinstance(chunks_data, list):
                        # Create document entry
                        doc_entry = {
                            "doc_id": report_id,
                            "source_file": f"{report_id}.json",
                            "chunks": chunks_data
                        }

                        documents.append(doc_entry)
                        logger.info(f"Loaded {len(chunks_data)} chunks from {report_id}")
                    else:
                        logger.warning(f"Unexpected JSON format in {report_id}.json")
                else:
                    logger.warning(f"JSON file not found for report {report_id}")

            except Exception as e:
                logger.error(f"Error loading {report_dir.name}: {str(e)}")
                continue

    logger.info(f"📊 Loaded {len(documents)} documents with chunks from input_data")
    return documents


class EmbeddingsController:
    def __init__(self):
        """Initialize the embeddings controller"""
        self.milvus_embeddings = MilvusEmbeddings()
        self.processed_dir = "source_data/processed"
    
    def run_embeddings_pipeline(self) -> bool:
        """
        Run the complete embeddings pipeline
        
        Returns:
            True if pipeline completed successfully, False otherwise
        """
        try:
            print("🚀 Starting Milvus Embeddings Pipeline")
            print("=" * 60)
            
            # Step 1: Check Milvus connection
            print("\n🔗 Step 1: Checking Milvus Connection")
            if not self.milvus_embeddings.check_milvus_connection():
                print("❌ Cannot connect to Milvus server. Please ensure it's running on localhost:19530")
                return False
            print("✅ Milvus connection successful!")
            
            # Step 2: Create collection
            print("\n📊 Step 2: Setting up Collection")
            if not self.milvus_embeddings.create_collection():
                print("❌ Failed to create collection. Exiting.")
                return False
            print("✅ Collection setup complete!")
            
            # Step 3: Process images and create embeddings
            print("\n🖼️  Step 3: Processing Images and Creating Embeddings")
            results = self.milvus_embeddings.process_all_images(self.processed_dir)
            
            # Step 4: Load collection into memory
            print("\n🔄 Step 4: Loading Collection into Memory")
            if self.milvus_embeddings.load_collection():
                print("✅ Collection loaded successfully!")
            else:
                print("⚠️ Collection loading failed, but data is stored.")
            
            # Step 5: Display results
            print("\n📈 Step 5: Processing Results")
            self.print_results(results)
            
            # Step 6: Collection info
            print("\n📋 Step 6: Collection Information")
            collection_info = self.milvus_embeddings.get_collection_info()
            self.print_collection_info(collection_info)
            
            print("\n🎉 Embeddings pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed with error: {str(e)}")
            return False
    
    def print_results(self, results: dict):
        """Print processing results"""
        print(f"📊 Total files processed: {results['total_files']}")
        print(f"✅ Successful: {results['successful']}")
        print(f"❌ Failed: {results['failed']}")
        
        if results['errors']:
            print(f"\n⚠️  Failed files:")
            for error_file in results['errors']:
                print(f"   - {error_file}")
    
    def print_collection_info(self, info: dict):
        """Print collection information"""
        if 'error' in info:
            print(f"❌ Error: {info['error']}")
        else:
            print(f"📁 Collection: {info['collection_name']}")
            print(f"📝 Description: {info['description']}")
            print(f"📊 Total entities: {info['num_entities']}")
            print(f"🔢 Vector dimension: {info['vector_dim']}")

def create_hierarchical_embeddings_for_report(report_id: str) -> bool:
    """
    Create hierarchical embeddings for a specific report's chunks

    Args:
        report_id: The report ID to process

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"🎯 Creating hierarchical embeddings for report: {report_id}")

        # Check if chunked data exists in input_data
        chunk_file_path = f"input_data/{report_id}/{report_id}.json"
        if not os.path.exists(chunk_file_path):
            print(f"❌ Chunked data not found: {chunk_file_path}")
            print("Please run the chunking pipeline first.")
            return False

        print(f"📁 Found chunked data: {chunk_file_path}")

        # Load the chunked data for this report
        try:
            with open(chunk_file_path, 'r') as f:
                chunks_data = json.load(f)
        except Exception as e:
            print(f"❌ Error loading chunked data: {e}")
            return False

        # Create document entry for hierarchical processing
        doc_entry = {
            "doc_id": report_id,
            "source_file": f"{report_id}.json",
            "chunks": chunks_data if isinstance(chunks_data, list) else []
        }

        print(f"📊 Loaded {len(doc_entry['chunks'])} chunks for report {report_id}")

        # Initialize hierarchical controller
        hierarchical_controller = HierarchicalEmbeddingsController()

        # Build indices for this single document (append to existing collection)
        try:
            print("🏗️ Building Level 2 hierarchical index for report...")
            hierarchical_controller.build_level2_index([doc_entry], clear_existing=False)

            print(f"✅ Successfully created hierarchical embeddings for report {report_id}")
            return True

        except Exception as e:
            print(f"❌ Failed to build hierarchical index for report {report_id}: {e}")
            return False

    except Exception as e:
        print(f"❌ Error creating hierarchical embeddings for report {report_id}: {e}")
        return False

def create_embeddings_for_report(report_id: str) -> bool:
    """
    Create embeddings for a specific report's processed image

    Args:
        report_id: The report ID to process

    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"🎯 Creating embeddings for report: {report_id}")

        # Check if processed image exists
        processed_image_path = f"input_data/{report_id}/processed_DDD.png"
        if not os.path.exists(processed_image_path):
            print(f"❌ Processed image not found: {processed_image_path}")
            return False

        # Check if JSON exists for metadata
        json_path = f"input_data/{report_id}/{report_id}.json"
        address = "Unknown Address"
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    # Get address from the first item (property info)
                    if data and isinstance(data, list) and len(data) > 0:
                        address = data[0].get("address", "Unknown Address")
            except Exception as e:
                print(f"⚠️ Could not read address from JSON: {e}")
        else:
            print(f"❌ JSON file not found: {json_path}")
            return False

        print(f"📁 Found processed image: {processed_image_path}")

        if address == "Unknown Address":
            print(f"❌ Unknown address for report {report_id}")
            return False

        print(f"🏠 Address: {address}")

        # Initialize Milvus embeddings handler
        milvus_embeddings = MilvusEmbeddings()

        # Check Milvus connection
        if not milvus_embeddings.check_milvus_connection():
            print("❌ Cannot connect to Milvus server")
            return False

        # Create collection if needed
        if not milvus_embeddings.create_collection():
            print("❌ Failed to create/setup collection")
            return False

        # Process the specific image
        success = milvus_embeddings.process_image_file(processed_image_path, address)

        if success:
            print(f"✅ Successfully created embeddings for report {report_id}")

            # Load collection into memory
            milvus_embeddings.load_collection()

            return True
        else:
            print(f"❌ Failed to create embeddings for report {report_id}")
            return False

    except Exception as e:
        print(f"❌ Error creating embeddings for report {report_id}: {e}")
        return False

def main(process_chunks: bool = True, process_images: bool = True):
    """Main entry point"""
    print("🏠 Roof Measurement RAG - Milvus Embeddings Creator")
    print("=" * 60)

    success_count = 0
    total_operations = 0

    # Process chunks from input_data if requested
    if process_chunks:
        total_operations += 1
        print("\n🔍 Step 1: Processing chunks from input_data")
        print("-" * 50)

        # Load chunk data from input_data
        documents = load_input_data_chunks()

        if not documents:
            print(f"❌ No chunk data found in input_data directory")
        else:
            print(f"📁 Found {len(documents)} reports with chunks")

            # Initialize hierarchical controller
            hierarchical_controller = HierarchicalEmbeddingsController()

            # Show initial collection status
            hierarchical_controller.show_level2_collection_status()

            # Build Level 2 index
            print("🏗️ Building Level 2 hierarchical index...")
            hierarchical_controller.build_level2_index(documents, clear_existing=True)

            # Show final collection status
            hierarchical_controller.show_level2_collection_status()

            print("\n✅ Chunk processing completed successfully!")
            success_count += 1

    # Process images if requested
    if process_images:
        total_operations += 1
        print("\n🖼️ Step 2: Processing images")
        print("-" * 50)

        # Check if processed images exist
        processed_dir = "source_data/processed"
        if not os.path.exists(processed_dir):
            print(f"❌ Processed directory not found: {processed_dir}")
            print("Please run the main pipeline first to process images.")
        else:
            # Count processed images
            image_files = [f for f in os.listdir(processed_dir) if f.lower().endswith('.png')]
            if not image_files:
                print(f"❌ No processed images found in: {processed_dir}")
                print("Please run the main pipeline first to process images.")
            else:
                allowOnlyReportIds = [65287819,65167520,65145525,65143839,65088612,65072336,65033234,65033227,65033221,65030115,65025964,65008550,64975447,64951635,64920454,64920436,64903555,64895822,64892357,64891144];
                #allowOnlyReportIds = [];
                if allowOnlyReportIds:
                    image_files = [f for f in image_files if f.startswith("DDD_") and f.endswith(".png") and f.replace("DDD_", "").replace(".png", "") in allowOnlyReportIds]

                print(f"📁 Found {len(image_files)} processed images to create embeddings for")
                print(f"🔗 Milvus server: localhost:19530")
                print(f"🤖 Embedding model: amazon.titan-embed-image-v1")

                # Initialize and run image pipeline
                controller = EmbeddingsController()
                image_success = controller.run_embeddings_pipeline()

                if image_success:
                    print("\n✅ Image processing completed successfully!")
                    success_count += 1
                else:
                    print("\n❌ Image processing failed. Please check the logs for details.")

    print("\n" + "=" * 60)
    if success_count == total_operations:
        print("🎉 All operations completed successfully!")
    else:
        print(f"⚠️ Completed {success_count}/{total_operations} operations")
    print("=" * 60)

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Roof Measurement RAG - Milvus Embeddings Creator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Process both chunks and images (default)
        python run_create_embeddings.py

        # Process only chunks from input_data
        python run_create_embeddings.py --chunks-only

        # Process only images
        python run_create_embeddings.py --images-only

        # Show Level 2 collection status
        python run_create_embeddings.py --show-level2-status

        # Clear Level 2 collection (use with caution!)
        python run_create_embeddings.py --clear-level2
        """
    )

    parser.add_argument(
        "--chunks-only",
        action="store_true",
        help="Process only chunks from input_data (skip images)"
    )

    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Process only images (skip chunks)"
    )

    parser.add_argument(
        "--show-level2-status",
        action="store_true",
        help="Show Level 2 collection status and exit"
    )

    parser.add_argument(
        "--clear-level2",
        action="store_true",
        help="Clear Level 2 collection and exit (use with caution!)"
    )

    args = parser.parse_args()

    # Handle status and clear operations
    if args.show_level2_status:
        hierarchical_controller = HierarchicalEmbeddingsController()
        hierarchical_controller.show_level2_collection_status()
        sys.exit(0)

    if args.clear_level2:
        hierarchical_controller = HierarchicalEmbeddingsController()
        hierarchical_controller.show_level2_collection_status()
        print("\n⚠️  WARNING: This will clear the Level 2 collection!")
        confirm = input("Type 'yes' to confirm: ").strip().lower()
        if confirm == 'yes':
            hierarchical_controller.clear_level2_collection(confirm=True)
            hierarchical_controller.show_level2_collection_status()
        else:
            print("❌ Operation cancelled.")
        sys.exit(0)

    # Determine what to process
    process_chunks = not args.images_only  # Process chunks unless explicitly images-only
    process_images = not args.chunks_only  # Process images unless explicitly chunks-only

    if args.chunks_only and args.images_only:
        print("❌ Cannot specify both --chunks-only and --images-only")
        sys.exit(1)

    main(process_chunks=process_chunks, process_images=process_images)