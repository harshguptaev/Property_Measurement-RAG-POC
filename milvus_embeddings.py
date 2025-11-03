#!/usr/bin/env python3
"""
Milvus Embeddings Module
Creates embeddings for processed roof images and stores them in Milvus vector database
"""

import os
import json
import base64
import logging
from typing import List, Dict, Any, Optional
import boto3
from pymilvus import MilvusClient, DataType, FieldSchema, CollectionSchema

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MilvusEmbeddings:
    def __init__(self, milvus_host: str = "localhost", milvus_port: int = 19530):
        """
        Initialize Milvus embeddings handler
        
        Args:
            milvus_host: Milvus server host
            milvus_port: Milvus server port
        """
        self.milvus_host = milvus_host
        self.milvus_port = milvus_port
        self.collection_name = "roof_outline_embeddings"
        self.vector_dim = 1024  # Titan image embedding dimension
        
        # Initialize Milvus client
        self.milvus_client = MilvusClient(
            uri=f"http://{milvus_host}:{milvus_port}"
        )
        
        # Initialize AWS Bedrock client for Titan embeddings
        self.bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        logger.info(f"Initialized Milvus client: {milvus_host}:{milvus_port}")
    
    def check_milvus_connection(self) -> bool:
        """
        Check if Milvus server is running and accessible
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Try to list collections to test connection
            collections = self.milvus_client.list_collections()
            logger.info(f"✅ Milvus connection successful. Found {len(collections)} collections.")
            return True
        except Exception as e:
            logger.error(f"❌ Milvus connection failed: {str(e)}")
            return False
    
    def create_collection(self) -> bool:
        """
        Create the roof outline embeddings collection
        
        Returns:
            True if collection created successfully, False otherwise
        """
        try:
            # Check if collection already exists
            if self.milvus_client.has_collection(self.collection_name):
                logger.info(f"Collection '{self.collection_name}' already exists.")
                # Check if index exists, create if not
                if not self.has_index():
                    logger.info("Creating missing index...")
                    if self.create_index():
                        logger.info("✅ Index created successfully.")
                    else:
                        logger.warning("⚠️ Index creation failed.")
                return True
            
            # Define collection schema
            fields = [
                FieldSchema(name="property_id", dtype=DataType.VARCHAR, max_length=64, is_primary=True),
                FieldSchema(name="address", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
                FieldSchema(name="report_id", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="num_facets", dtype=DataType.INT64),
            ]
            
            schema = CollectionSchema(
                fields=fields,
                description="Roof outline image embeddings for visual similarity search"
            )
            
            # Create collection
            self.milvus_client.create_collection(
                collection_name=self.collection_name,
                schema=schema
            )
            
            logger.info(f"✅ Collection '{self.collection_name}' created successfully.")
            
            # Create index for vector field
            if self.create_index():
                logger.info("✅ Index created successfully.")
            else:
                logger.warning("⚠️ Index creation failed, but collection exists.")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create collection: {str(e)}")
            return False
    
    def has_index(self) -> bool:
        """
        Check if the collection has an index
        
        Returns:
            True if index exists, False otherwise
        """
        try:
            indexes = self.milvus_client.list_indexes(self.collection_name)
            return len(indexes) > 0
        except Exception as e:
            logger.error(f"❌ Error checking index: {str(e)}")
            return False
    
    def create_index(self) -> bool:
        """
        Create index for the vector field
        
        Returns:
            True if index created successfully, False otherwise
        """
        try:
            # Use prepare_index_params to create proper IndexParams object
            index_params = self.milvus_client.prepare_index_params(
                field_name="vector",
                metric_type="COSINE",  # Titan uses normalized vectors
                index_type="HNSW",
                M=32,
                efConstruction=200
            )
            
            self.milvus_client.create_index(
                collection_name=self.collection_name,
                index_params=index_params
            )
            
            logger.info("✅ Vector index created successfully.")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create index: {str(e)}")
            return False
    
    def titan_embed_image(self, image_path: str) -> Optional[List[float]]:
        """
        Get image embeddings from Amazon Titan Embed Image v1
        
        Args:
            image_path: Path to the image file
            
        Returns:
            List of embedding values or None if failed
        """
        try:
            # Read and encode image
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Prepare request body
            body = json.dumps({
                "inputImage": image_data
            })
            
            # Call Titan embedding model
            response = self.bedrock_client.invoke_model(
                modelId="amazon.titan-embed-image-v1",
                body=body,
                accept="application/json",
                contentType="application/json"
            )
            
            result = json.loads(response["body"].read())
            embedding = result["embedding"]
            
            logger.info(f"✅ Generated embedding for {os.path.basename(image_path)} (dim: {len(embedding)})")
            return embedding
            
        except Exception as e:
            logger.error(f"❌ Error generating image embedding for {image_path}: {str(e)}")
            return None
    
    def insert_embedding(self, property_id: str, address: str, report_id: str,
                        embedding: List[float], num_facets: int = 0) -> bool:
        """
        Insert embedding into Milvus collection
        
        Args:
            property_id: Unique property identifier
            address: Property address
            report_id: Report ID
            embedding: Vector embedding
            
        Returns:
            True if insertion successful, False otherwise
        """
        try:
            data = [{
                "property_id": property_id,
                "address": address,
                "vector": embedding,
                "report_id": report_id,
                "num_facets": num_facets
            }]
            
            self.milvus_client.insert(
                collection_name=self.collection_name,
                data=data
            )
            
            # Flush to ensure data is persisted
            self.milvus_client.flush(self.collection_name)
            
            logger.info(f"✅ Inserted embedding for property {property_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to insert embedding for property {property_id}: {str(e)}")
            return False
    
    def process_image_file(self, image_path: str, address: str = "Unknown Address") -> bool:
        """
        Process a single image file and store its embedding

        Args:
            image_path: Path to the processed image file
            address: Property address (optional)

        Returns:
            True if processing successful, False otherwise
        """
        try:
            # Extract report ID from filename (processed_ddd.png)
            filename = os.path.basename(image_path)
            if filename == "processed_DDD.png":
                # Get report_id from the directory name
                report_id = os.path.basename(os.path.dirname(image_path))
            else:
                logger.error(f"❌ Invalid filename format: {filename}")
                return False

            # Generate property ID
            property_id = f"PROP_{report_id}"

            # Extract num_facets from JSON data
            num_facets = 0
            json_path = os.path.join(os.path.dirname(image_path), f"{report_id}.json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                        # Look for num_facets in the chunks
                        for item in data:
                            if "data" in item and "total_roof_facets" in item["data"]:
                                try:
                                    num_facets = int(item["data"]["total_roof_facets"])
                                    break
                                except (ValueError, TypeError):
                                    continue
                except (json.JSONDecodeError, IOError) as e:
                    logger.warning(f"Could not read num_facets from JSON: {e}")

            logger.info(f"🔄 Processing image: {filename} (facets: {num_facets})")

            # Generate embedding
            embedding = self.titan_embed_image(image_path)
            if not embedding:
                return False

            # Insert into Milvus with num_facets
            success = self.insert_embedding(property_id, address, report_id, embedding, num_facets)
            return success

        except Exception as e:
            logger.error(f"❌ Error processing image {image_path}: {str(e)}")
            return False
    
    def process_all_images(self, processed_dir: str = "source_data/processed") -> Dict[str, Any]:
        """
        Process all images in the processed directory
        
        Args:
            processed_dir: Directory containing processed images
            
        Returns:
            Dictionary with processing results
        """
        results = {
            "total_files": 0,
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        try:
            if not os.path.exists(processed_dir):
                logger.error(f"❌ Processed directory not found: {processed_dir}")
                return results
            
            # Get all PNG files
            image_files = [f for f in os.listdir(processed_dir) 
                          if f.lower().endswith('.png')]
            
            results["total_files"] = len(image_files)
            logger.info(f"📁 Found {len(image_files)} image files to process")
            
            # Process each image
            for filename in sorted(image_files):
                image_path = os.path.join(processed_dir, filename)
                
                if self.process_image_file(image_path):
                    results["successful"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append(filename)
            
            logger.info(f"✅ Processing complete: {results['successful']}/{results['total_files']} successful")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error processing images: {str(e)}")
            return results
    
    def load_collection(self) -> bool:
        """
        Load the collection into memory
        
        Returns:
            True if collection loaded successfully, False otherwise
        """
        try:
            self.milvus_client.load_collection(self.collection_name)
            logger.info(f"✅ Collection '{self.collection_name}' loaded into memory")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load collection: {str(e)}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection
        
        Returns:
            Dictionary with collection information
        """
        try:
            if not self.milvus_client.has_collection(self.collection_name):
                return {"error": "Collection does not exist"}
            
            info = self.milvus_client.describe_collection(self.collection_name)
            stats = self.milvus_client.get_collection_stats(self.collection_name)
            
            return {
                "collection_name": self.collection_name,
                "description": info.get("description", ""),
                "num_entities": stats.get("row_count", 0),
                "vector_dim": self.vector_dim
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting collection info: {str(e)}")
            return {"error": str(e)}
