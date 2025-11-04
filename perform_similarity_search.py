#!/usr/bin/env python3
"""
Perform Similarity Search Module
Downloads image from S3, creates embeddings, and performs similarity search in Milvus
"""

import os
import json
import base64
import logging
import tempfile
from typing import List, Dict, Any, Optional
import boto3
from pymilvus import MilvusClient
from botocore.exceptions import ClientError

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimilaritySearcher:
    def __init__(self, milvus_host: str = "localhost", milvus_port: int = 19530):
        """
        Initialize similarity search handler
        
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
        
        # Initialize AWS clients with specific credentials
        # Use S3-specific credentials for S3 operations
        s3_session = boto3.Session(
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID_S3"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY_S3"),
            aws_session_token=os.getenv("AWS_SESSION_TOKEN_S3"),
            region_name=os.getenv("AWS_REGION_S3", "us-east-2"),
        )
        self.s3_client = s3_session.client('s3')
        self.bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        logger.info(f"Initialized SimilaritySearcher: {milvus_host}:{milvus_port}")
    
    def check_milvus_connection(self) -> bool:
        """
        Check if Milvus server is running and accessible
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            collections = self.milvus_client.list_collections()
            logger.info(f"✅ Milvus connection successful. Found {len(collections)} collections.")
            return True
        except Exception as e:
            logger.error(f"❌ Milvus connection failed: {str(e)}")
            return False
    
    def download_image_from_s3(self, s3_url: str, local_path: str) -> bool:
        """
        Download image from S3 URL
        
        Args:
            s3_url: S3 URL (e.g., s3://bucket-name/path/to/image.png)
            local_path: Local path to save the image
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Parse S3 URL
            if not s3_url.startswith("s3://"):
                logger.error(f"❌ Invalid S3 URL format: {s3_url}")
                return False
            
            # Remove s3:// prefix and split
            path_parts = s3_url.replace("s3://", "").split("/", 1)
            bucket_name = path_parts[0]
            object_key = path_parts[1] if len(path_parts) > 1 else ""
            
            logger.info(f"🔄 Downloading image from S3: {s3_url}")
            
            # Download file
            self.s3_client.download_file(bucket_name, object_key, local_path)
            
            logger.info(f"✅ Image downloaded successfully to: {local_path}")
            return True
            
        except ClientError as e:
            logger.error(f"❌ S3 download error: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ Error downloading image from S3: {str(e)}")
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
    
    def load_collection(self) -> bool:
        """
        Load collection into memory for search
        
        Returns:
            True if load successful, False otherwise
        """
        try:
            if not self.milvus_client.has_collection(self.collection_name):
                logger.error(f"❌ Collection '{self.collection_name}' does not exist.")
                return False
            
            # Load collection into memory
            self.milvus_client.load_collection(self.collection_name)
            logger.info(f"✅ Collection '{self.collection_name}' loaded into memory.")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading collection: {str(e)}")
            return False
    
    def search_similar_roofs(self, query_vector: List[float], limit: int = 10, 
                           filter_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform similarity search on roof outline embeddings with optional filtering
        
        Args:
            query_vector: Embedding vector of the query image
            limit: Number of results to return (default: 10)
            filter_params: Optional filter parameters (facet_count, area, etc.)
            
        Returns:
            Dictionary with search results
        """
        results = {
            'success': False,
            'matches': [],
            'total_found': 0,
            'errors': []
        }
        
        try:
            # Ensure collection is loaded
            if not self.load_collection():
                results['errors'].append("Failed to load collection")
                return results
            
            # Check if collection exists
            if not self.milvus_client.has_collection(self.collection_name):
                results['errors'].append(f"Collection '{self.collection_name}' does not exist")
                return results
            
            # Define search parameters
            search_params = {
                "metric_type": "COSINE",
                "params": {"ef": 128}
            }
            
            # Build filter expression if facet_count is provided
            filter_expr = None
            if filter_params and 'facet_count' in filter_params:
                facet_count = filter_params['facet_count']
                if facet_count is not None:
                    # Calculate +/- 20% range
                    min_facets = int(facet_count * 0.8)
                    max_facets = int(facet_count * 1.2)
                    # Use num_facets field name (INT64 type)
                    filter_expr = f"num_facets >= {min_facets} && num_facets <= {max_facets}"
                    logger.info(f"🔍 Filtering by num_facets: {min_facets} <= num_facets <= {max_facets}")
            
            # Get more results if filtering to ensure we have enough after filtering
            search_limit = limit * 3 if filter_expr else limit
            
            logger.info(f"🔍 Performing similarity search with limit={search_limit}")
            
            # Perform search - use the working format with search_params
            # Add filter parameter if filter expression exists
            if filter_expr:
                search_results = self.milvus_client.search(
                    collection_name=self.collection_name,
                    data=[query_vector],
                    anns_field="vector",
                    search_params=search_params,
                    limit=search_limit,
                    filter=filter_expr,  # Use 'filter' parameter for filter expression
                    output_fields=["property_id", "address", "report_id", "num_facets"]
                )
            else:
                search_results = self.milvus_client.search(
                    collection_name=self.collection_name,
                    data=[query_vector],
                    anns_field="vector",
                    search_params=search_params,
                    limit=search_limit,
                    output_fields=["property_id", "address", "report_id", "num_facets"]
                )
             
            
            # Process results - MilvusClient returns list of lists (one per query vector)
            if search_results and len(search_results) > 0:
                hits = search_results[0]  # First query's results
                logger.debug(f"Found {len(hits)} hits in search results")

                for hit in hits:
                    # Extract data from hit - pymilvus returns Hit objects
                    try:
                        # pymilvus Hit objects have output_fields as direct attributes
                        # Access them directly: hit.property_id, hit.address, hit.report_id, hit.distance
                        
                        # Get distance
                        distance = getattr(hit, 'distance', 0.0)
                        
                        # Direct attribute access (pymilvus Hit objects have these as attributes)
                        # Check if attribute exists first, then get value
                        if hasattr(hit, 'property_id'):
                            property_id = hit.property_id
                        elif hasattr(hit, 'id'):
                            property_id = hit.id
                        else:
                            property_id = None
                        
                        if hasattr(hit, 'address'):
                            address = hit.address
                        else:
                            address = None
                        
                        if hasattr(hit, 'report_id'):
                            report_id = hit.report_id
                        else:
                            report_id = None
                        
                        # Fallback: try entity dict if attributes not found
                        if (property_id is None or address is None or report_id is None) and hasattr(hit, 'entity'):
                            entity = hit.entity
                            if isinstance(entity, dict):
                                property_id = property_id or entity.get('property_id') or entity.get('id')
                                address = address or entity.get('address')
                                report_id = report_id or entity.get('report_id')
                        
                        # Final fallback: use defaults
                        property_id = property_id or getattr(hit, 'id', None) or getattr(hit, 'pk', 'Unknown')
                        address = address or 'Unknown Address'
                        report_id = report_id or 'Unknown'
                    
                    except Exception as e:
                        logger.error(f"Error extracting data from hit: {str(e)}")
                        property_id = 'Unknown'
                        address = 'Unknown Address'
                        report_id = 'Unknown'
                        distance = 0.0
                    
                    match = {
                        'property_id': property_id,
                        'address': address,
                        'report_id': report_id,
                        'similarity_score': distance,
                        #'similarity_score': 1 - distance  # COSINE distance to similarity (higher = more similar)
                    }
                    results['matches'].append(match)
                
                # Filter by similarity score threshold (>= 0.80)
                score_threshold = 0.80
                original_count = len(results['matches'])
                filtered_matches = []
                excluded_matches = []
                
                for match in results['matches']:
                    if match['similarity_score'] >= score_threshold:
                        filtered_matches.append(match)
                    else:
                        excluded_matches.append(match)
                        logger.info(f"⚠️ Excluded result (score {match['similarity_score']:.4f} < {score_threshold}): "
                                  f"Property ID: {match['property_id']}, Address: {match['address']}")
                
                # Update results with filtered matches
                results['matches'] = filtered_matches
                
                # Log filtering summary
                if excluded_matches:
                    logger.info(f"📊 Score filtering: {original_count} results -> {len(filtered_matches)} results "
                              f"(excluded {len(excluded_matches)} with score < {score_threshold})")
                else:
                    logger.info(f"📊 All {original_count} results passed score threshold ({score_threshold})")
                
                # Limit to top 5 for level 1 search
                results['matches'] = results['matches'][:5]
                results['total_found'] = len(results['matches'])
                results['success'] = True
                logger.info(f"✅ Found {results['total_found']} similar roofs (top 5, score >= {score_threshold})")
            else:
                logger.warning("⚠️ No results found")
                results['errors'].append("No results found")
            
        except Exception as e:
            logger.error(f"❌ Error performing similarity search: {str(e)}")
            results['errors'].append(str(e))
        
        return results
    
    def search_by_s3_url(self, s3_url: str, limit: int = 5, filter_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Complete pipeline: Download image from S3, create embedding, and search
        
        Args:
            s3_url: S3 URL of the roof_outline_simplified.png image
            limit: Number of results to return (default: 5)
            filter_params: Optional filter parameters (facet_count, area, etc.)
            
        Returns:
            Dictionary with search results
        """
        results = {
            'success': False,
            's3_url': s3_url,
            'matches': [],
            'total_found': 0,
            'errors': []
        }
        
        try:
            # Step 1: Download image from S3
            logger.info(f"📥 Step 1: Downloading image from S3")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                local_image_path = tmp_file.name
                
                if not self.download_image_from_s3(s3_url, local_image_path):
                    results['errors'].append("Failed to download image from S3")
                    return results
                
                # Step 2: Create embedding
                logger.info(f"🧠 Step 2: Creating embedding using Titan")
                query_vector = self.titan_embed_image(local_image_path)
                
                if not query_vector:
                    results['errors'].append("Failed to create embedding")
                    return results
                
                # Step 3: Perform similarity search
                logger.info(f"🔍 Step 3: Performing similarity search")
                search_results = self.search_similar_roofs(query_vector, limit, filter_params)
                
                # Merge results
                results['success'] = search_results['success']
                results['matches'] = search_results['matches']
                results['total_found'] = search_results['total_found']
                results['errors'].extend(search_results['errors'])
                
                # Clean up temporary file
                try:
                    os.unlink(local_image_path)
                except Exception:
                    pass
                
                return results
                
        except Exception as e:
            logger.error(f"❌ Error in search_by_s3_url: {str(e)}")
            results['errors'].append(str(e))
            return results

