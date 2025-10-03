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

# Simple MilvusCollectionManager class
class MilvusCollectionManager:
    """Simple collection manager for Milvus operations"""
    
    def __init__(self, uri: str):
        self.client = MilvusClient(uri=uri)
    
    def create_collection_safely(self, collection_name: str, embedding_dim: int, metric_type: str = "IP", 
                                clear_existing: bool = True, index_params: Dict = None):
        """Create a collection safely, optionally clearing existing data"""
        if clear_existing and self.client.has_collection(collection_name):
            logger.info(f"Dropping existing collection: {collection_name}")
            self.client.drop_collection(collection_name)
        
        if not self.client.has_collection(collection_name):
            logger.info(f"Creating collection: {collection_name}")
            self.client.create_collection(
                collection_name=collection_name,
                dimension=embedding_dim,
                metric_type=metric_type,
                index_type="HNSW",
                max_length=65535
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
        
        # Initialize Milvus client (using Milvus Lite)
        self.milvus_client = MilvusClient(uri="./milvus_demo.db")
        
        # Initialize Milvus collection manager for safe collection handling
        self.milvus_manager = MilvusCollectionManager(uri="./milvus_demo.db")
        
        # Collection names for the two levels
        self.level1_collection_name = "hierarchical_level1"
        self.level2_collection_name = "hierarchical_level2"
        
        # Embedding dimension for Titan (will be determined dynamically)
        self.embedding_dim = 1536  # Titan embedding dimension
        
        logger.info(f"🔧 Initialized Hierarchical RAG with model: {model_id}")
        logger.info(f"🔧 Level 1 collection: {self.level1_collection_name}")
        logger.info(f"🔧 Level 2 collection: {self.level2_collection_name}")
    
    def titan_embed_text(self, text: str) -> List[float]:
        """
        Get text embeddings from Amazon Titan Embed Text v1
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        try:
            body = json.dumps({
                "inputText": text
            })
            
            response = self.bedrock_client.invoke_model(
                modelId="amazon.titan-embed-text-v1",
                #modelId="amazon.titan-embed-image-v1"
                body=body,
                accept="application/json",
                contentType="application/json"
            )
            
            result = json.loads(response["body"].read())
            embedding = result["embedding"]
            
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
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 200,
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
        
        index_params = {
            "index_type": "HNSW",
            "metric_type": "IP",
            "params": {
                "M": 48,
                "efConstruction": 200
            }
        }
        # Create Level 1 collection (Document summaries) - safely with clearing
        self.milvus_manager.create_collection_safely(
            collection_name=self.level1_collection_name,
            embedding_dim=self.embedding_dim,
            metric_type="IP",
            clear_existing=clear_existing,
            index_params=index_params
        )
        
        # Create Level 2 collection (Chunks) - safely with clearing
        self.milvus_manager.create_collection_safely(
            collection_name=self.level2_collection_name,
            embedding_dim=self.embedding_dim,
            metric_type="IP",
            clear_existing=clear_existing,
            index_params=index_params
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
        Build Level 1 Index (Summary/Metadata Index)
        
        Args:
            documents_data: List of documents with their chunks
            clear_existing: Whether to clear existing collections to prevent duplicates
        """
        logger.info("🏗️ Building Level 1 Index (Summary/Metadata Index)")
        
        # Create collections first (with duplicate prevention)
        self.create_milvus_collections(clear_existing=clear_existing)
        
        level1_data = []
        
        for i, doc in enumerate(tqdm(documents_data, desc="Building Level 1 Index")):
            try:
                # Extract document metadata
                doc_id = doc.get("doc_id", f"doc_{i}")
                source_file = doc.get("source_file", "unknown.pdf")
                chunks = doc.get("chunks", [])
                
                # Extract address from chunks - try multiple approaches
                address = "Unknown Address"
                date = "Unknown Date"
                
                # First try Report Header
                for chunk in chunks:
                    if chunk.get("section") == "Report Header":
                        data = chunk.get("data", {})
                        if "property_address" in data:
                            address = data["property_address"]
                        if "date" in data:
                            date = data["date"]
                        break
                
                # If still unknown, try other sections that might have address
                if address == "Unknown Address":
                    for chunk in chunks:
                        data = chunk.get("data", {})
                        if isinstance(data, dict):
                            # Look for any field containing address
                            for key, value in data.items():
                                if "address" in key.lower() and value and value != "Unknown Address":
                                    address = value
                                    break
                            if address != "Unknown Address":
                                break
                
                # Final fallback - extract from doc_id if it contains address info
                if address == "Unknown Address":
                    # Some doc_ids might contain property info
                    address = f"Property {doc_id}"
                
                print("=" * 80)
                print(f"address ::: {address}")
                #print(f"date --------->: {chunks}")

                # Create document summary
                #summary = self.create_document_summary(chunks)
                
                # Collect all chunk IDs
                chunk_ids = [chunk.get("chunk_id", "") for chunk in chunks]
                chunk_ids_str = ",".join(chunk_ids)

                print(f"chunk_ids ::: {chunk_ids_str}")
                print("=" * 80)
                
                # Generate embedding for summary
                summary_embedding = self.titan_embed_text(address)


                #print(f"summary_embedding --------->: {summary}")
               
                level1_data.append({
                    "id": i,
                    "vector": summary_embedding,
                    "doc_id": doc_id,
                    "summary": "Summary",
                    "chunk_ids": chunk_ids_str,
                    "address": address,
                    "source_file": source_file,
                    "date": date
                })
                
                #print(f"summary json--------->: {level1_data}")
                

                logger.info(f"Created Level 1 entry for {doc_id}: {address}")
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
            logger.info(f"✅ Level 1 Index built with {len(level1_data)} documents")
        else:
            logger.warning("No data to insert into Level 1 Index")
    
    def build_level2_index(self, documents_data: List[Dict], clear_existing: bool = False) -> None:
        """
        Build Level 2 Index (Chunk Index)
        
        Args:
            documents_data: List of documents with their chunks
            clear_existing: Whether to clear existing collections (usually False since Level 1 already did this)
        """
        logger.info("🏗️ Building Level 2 Index (Chunk Index)")
        
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
                
                for chunk in chunks:
                    try:
                        # Extract chunk information
                        chunk_id = chunk.get("chunk_id", f"chunk_{chunk_counter}")
                        section = chunk.get("section", "Unknown Section")
                        chunk_type = chunk.get("type", "text")
                        source_file = doc.get("source_file", "unknown.pdf")
                        
                        # Create chunk text for embedding
                        chunk_text = ""
                        data = chunk.get("data", {})
                        
                        if isinstance(data, dict):
                            # Flatten the data dictionary into readable text
                            for key, value in data.items():
                                chunk_text += f"{key}: {value}\n"
                        else:
                            chunk_text = str(data)
                        
                        # Add section and type information
                        chunk_text = f"Section: {section}\nType: {chunk_type}\nContent: {chunk_text}"
                        
                        # Generate embedding for chunk text
                        chunk_embedding = self.titan_embed_text(chunk_text)
                        
                        level2_data.append({
                            "id": chunk_counter,
                            "vector": chunk_embedding,
                            "chunk_id": chunk_id,
                            "doc_id": doc_id,
                            "section": section,
                            "chunk_text": chunk_text[:5000],  # Truncate if too long
                            "chunk_type": chunk_type,
                            "source_file": source_file
                        })
                        
                        chunk_counter += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
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
            logger.info(f"✅ Level 2 Index built with {len(level2_data)} chunks")
        else:
            logger.warning("No data to insert into Level 2 Index")
    
    def search_hierarchical(self, query: str, level1_limit: int = 1, level2_limit: int = 3) -> List[Dict]:
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
            # Step 1: Embed query
            query_vec = self.titan_embed_text(query)

            search_params = {
                "metric_type": "IP",
                "params": {"ef": 64}   # higher ef = better recall, slower search
                }
            
            # Step 2: Search Level 1 (Summaries)
            logger.info("📊 Searching Level 1 (Summary Index)...")
            res1 = self.milvus_client.search(
                collection_name=self.level1_collection_name,
                data=[query_vec],
                limit=1,
                output_fields=["doc_id", "chunk_ids", "address", "summary"],
                search_params=search_params
            )
            
            if not res1 or not res1[0]:
                logger.warning("No results found in Level 1 Index")
                return []


            print(f"Fetching Chunks from Level 1 Index - Address matching")

            # Get relevant chunk_ids
            retrieved_chunk_ids = []
            level1_docs = []
            
            for hit in res1[0]:
                doc_info = {
                    "doc_id": hit.get("doc_id"),
                    "address": hit.get("address"),
                    "summary": hit.get("summary"),
                    "distance": hit.get("distance", 0)
                }
                level1_docs.append(doc_info)

                print(f"\n{'='*80}")
                print(f"address: {doc_info['address']}")
                print(f"doc_id: {doc_info['doc_id']}")
                print(f"distance: {doc_info['distance']}")
            
                
                # Parse chunk IDs
                chunk_ids_str = hit.get("chunk_ids", "")
                #print(f"Retrieved chunk_ids_str --------->: {chunk_ids_str}")

                #lets include chunk from 1st index
                if chunk_ids_str:
                    retrieved_chunk_ids.extend(chunk_ids_str.split(",")[1:])

                print(f"retrieved_chunk_ids ::: {retrieved_chunk_ids}")
                print("=" * 80)
            
            logger.info(f"📋 Found {len(level1_docs)} relevant documents with {len(retrieved_chunk_ids)} total chunks")
            
            # Step 3: Search Level 2 (Chunks within retrieved docs)
            logger.info("📊 Searching Level 2 (Chunk Index)...")
            
            if not retrieved_chunk_ids:
                logger.warning("No chunk IDs found from Level 1 search")
                return []


            # Step 3: Search Level 2 (Chunks within retrieved docs)
            #expr = f'chunk_id in {retrieved_chunk_ids}'

            # Safer way: use repr() to auto-quote strings, then replace single quotes with double quotes
            chunk_ids_quoted = [f'"{cid}"' for cid in retrieved_chunk_ids]
            expr = f'chunk_id in [{",".join(chunk_ids_quoted)}]'

            # For simplicity, let's search all chunks and then filter by relevance
            res2 = self.milvus_client.search(
                collection_name=self.level2_collection_name,
                data=[query_vec],
                limit=7,  # Get more results to include images
                filter=expr,
                output_fields=["chunk_text", "section", "doc_id", "chunk_type", "chunk_id"]
            )
            
            if not res2 or not res2[0]:
                logger.warning("No results found in Level 2 Index")
                return []
            
            # Step 4: Filter results to only include chunks from relevant documents
            final_results = []
            relevant_doc_ids = [doc["doc_id"] for doc in level1_docs]

            print(f"Fetching Chunks from Level 2 Index")

            print(f"\n{'='*80}")
            print(f"relevant_doc_ids :: {relevant_doc_ids}")
            print(f"Length :: {len(res2[0])}")
            print("=" * 80)

            # Separate text and image chunks
            text_chunks = []
            image_chunks = []

            for result in res2[0]:
                chunk_doc_id = result.get("doc_id")
                if chunk_doc_id in relevant_doc_ids:
                    chunk_info = {
                        "chunk_id": result.get("chunk_id"),
                        "doc_id": chunk_doc_id,
                        "section": result.get("section"),
                        "chunk_type": result.get("chunk_type"),
                        "chunk_text": result.get("chunk_text"),
                        "distance": result.get("distance", 0)
                    }
                    
                    # Add document-level info from Level 1
                    for doc_info in level1_docs:
                        if doc_info["doc_id"] == chunk_doc_id:
                            chunk_info["doc_address"] = doc_info["address"]
                            chunk_info["doc_summary"] = doc_info["summary"]
                            break
                    
                    # Separate by type
                    if result.get("chunk_type") == "image":
                        image_chunks.append(chunk_info)
                    else:
                        text_chunks.append(chunk_info)

            # Combine results: prioritize text chunks but include all relevant images
            final_results = text_chunks[:level2_limit]  # Take top text chunks based on limit
            
            # Add all image chunks from relevant documents (they're important for frontend display)
            final_results.extend(image_chunks)
            
            # If we don't have many image chunks, try to get ALL images from relevant documents
            if len(image_chunks) < 3:
                try:
                    # Search for all image chunks from relevant documents
                    doc_ids_quoted = [f'"{doc_id}"' for doc_id in relevant_doc_ids]
                    doc_filter = f'doc_id in [{",".join(doc_ids_quoted)}]'
                    
                    all_images_res = self.milvus_client.search(
                        collection_name=self.level2_collection_name,
                        data=[query_vec],
                        limit=20,  # Get more images
                        filter=doc_filter,
                        output_fields=["chunk_text", "section", "doc_id", "chunk_type", "chunk_id"]
                    )
                    
                    if all_images_res and all_images_res[0]:
                        for result in all_images_res[0]:
                            if result.get("chunk_type") == "image" and result.get("doc_id") in relevant_doc_ids:
                                # Check if we already have this image
                                chunk_id = result.get("chunk_id")
                                if not any(chunk["chunk_id"] == chunk_id for chunk in final_results):
                                    chunk_info = {
                                        "chunk_id": chunk_id,
                                        "doc_id": result.get("doc_id"),
                                        "section": result.get("section"),
                                        "chunk_type": result.get("chunk_type"),
                                        "chunk_text": result.get("chunk_text"),
                                        "distance": result.get("distance", 0)
                                    }
                                    
                                    # Add document-level info
                                    for doc_info in level1_docs:
                                        if doc_info["doc_id"] == result.get("doc_id"):
                                            chunk_info["doc_address"] = doc_info["address"]
                                            chunk_info["doc_summary"] = doc_info["summary"]
                                            break
                                    
                                    final_results.append(chunk_info)
                except Exception as e:
                    logger.warning(f"Error fetching additional images: {e}")
            
            print(f"Text chunks: {len(text_chunks)}, Image chunks: {len(image_chunks)}")
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
                print(f"📋 Doc ID: {chunk.get('doc_id', 'N/A')}")
                if chunk.get('doc_summary'):
                    print(f"📝 Summary: {chunk.get('doc_summary', 'N/A')}")

                print("\n📖 Content:")
                content = chunk.get('chunk_text', 'N/A')
                if len(content) > 500:
                    print(f"   {content[:500]}...")
                else:
                    print(f"   {content}")

                print(f"{'='*60} END CHUNK #{i} {'='*60}")

            print("\n" + "="*80)
            print("END OF FINAL RESULTS")
            print("="*80)
            
            logger.info(f"✅ Hierarchical search completed. Found {len(final_results)} relevant chunks")
            return final_results
            
        except Exception as e:
            logger.error(f"Error during hierarchical search: {str(e)}")
            return []
    
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
        images_found = []
        
        for i, result in enumerate(results, 1):
            doc_address = result.get('doc_address', 'Unknown Address')
            section = result.get('section', 'Unknown Section')
            chunk_type = result.get('chunk_type', 'text')
            chunk_text = result.get('chunk_text', '')
            
            # Extract image information if this is an image chunk
            if chunk_type == 'image' and 'image_file:' in chunk_text:
                image_path = None
                for line in chunk_text.split('\n'):
                    if line.startswith('image_file:'):
                        image_path = line.split('image_file:')[1].strip()
                        break
                
                if image_path:
                    # Create user-friendly image title
                    filename = image_path.split('/')[-1].replace('.png', '')
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
                    
                    display_title = title_mappings.get(filename, section)
                    images_found.append({
                        'title': display_title,
                        'section': section,
                        'filename': filename,
                        'address': doc_address
                    })
            
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
        
        context = "\n".join(context_parts)
        
        # Create comprehensive prompt for LLM
        images_context = ""
        if images_found:
            images_context = f"\n\nAvailable Images:\n"
            for img in images_found:
                images_context += f"- {img['title']} (from {img['address']})\n"
        
        prompt = f"""You are a professional roofing analysis specialist. Based on the retrieved information below, provide a comprehensive and accurate answer to the user's question in JSON format with enhanced structure and detail.

User Question: {query}

Retrieved Information:
{context}{images_context}

Instructions:
1. Return your response as a valid JSON object with the following enhanced structure
2. Provide a direct, comprehensive answer to the user's question
3. Include ALL specific measurements, addresses, and technical details available
4. Organize measurements by logical categories (basic info, roof components, dimensions)
5. Use precise technical language while remaining clear and professional
6. Reference images by their descriptive names and explain their relevance
7. Provide detailed analysis and context for measurements

Enhanced JSON Response Format:
{{
    "answer": "Comprehensive, detailed answer to the user's question with specific measurements and technical details",
    "summary": "Executive summary highlighting the most critical findings and key measurements",
    "properties": [
        {{
            "address": "Complete property address",
            "key_measurements": {{
                "total_area": "Total roof area with units (e.g., '2,450 sq ft')",
                "roof_facets": "Number of roof facets (e.g., '31 facets')",
                "predominant_pitch": "Main roof pitch (e.g., '12/12')",
                "obstructions": "Obstruction details (e.g., '6 obstructions, 28.3 sq ft total')",
                "ridges": "Ridge measurements with units",
                "hips": "Hip measurements with units", 
                "valleys": "Valley measurements with units",
                "rakes": "Rake measurements with units",
                "eaves_starters": "Eaves measurements with counts",
                "drip_edge": "Drip edge measurements with counts",
                "flashing": "Flashing measurements",
                "step_flashing": "Step flashing measurements",
                "coordinates": "Latitude and longitude if available"
            }},
            "additional_details": [
                "Detailed technical specifications",
                "Material requirements",
                "Structural observations",
                "Access considerations",
                "Special conditions or notes"
            ]
        }}
    ],
    "images_mentioned": ["List of specific images referenced in the analysis"],
    "images_available": [
        {{
            "title": "Descriptive image title (e.g., 'Roof Azimuth/Direction Diagram')",
            "section": "Technical section name",
            "filename": "Technical filename",
            "address": "Property address"
        }}
    ],
    "confidence": "high/medium/low based on data completeness and accuracy",
    "notes": "Technical limitations, data quality notes, or additional context for analysis"
}}

Provide only the JSON response, no additional text:"""

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
    
    def answer_query(self, query: str, level1_limit: int = 1, level2_limit: int = 3, show_raw_results: bool = False) -> str:
        """
        Complete query answering pipeline: search + LLM response generation
        
        Args:
            query: User's question
            level1_limit: Number of documents to retrieve from Level 1
            level2_limit: Number of chunks to retrieve from Level 2
            show_raw_results: Whether to print raw search results
            
        Returns:
            LLM-generated response string
        """
        # Perform hierarchical search
        results = self.search_hierarchical(query, level1_limit, level2_limit)
        
        # Optionally show raw results
        if show_raw_results:
            self.print_search_results(query, results)
        
        # Generate LLM response
        llm_response = self.generate_llm_response(query, results)
        
        return llm_response
    
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
            # The structure appears to be {"text": [...chunks...], ...}
            chunks = []
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
    if query:
        # Single query provided as parameter
        test_queries = [query]
    else:
        # Default test queries
        test_queries = [
            "What are the area for address 1407 Moher Blvd Franklin, TN 37067?",
            "What is the property address?",
            "What are the roof obstructions?",
            "What is the roof pitch information?",
            "Show me the imagery and images"
        ]
    
    if raw_only:
        logger.info("🔍 Testing hierarchical search (raw results only)...")
        for test_query in test_queries:
            print(f"\n{'='*80}")
            results = hierarchical_rag.search_hierarchical(test_query, level1_limit=1, level2_limit=3)
            hierarchical_rag.print_search_results(test_query, results)
            time.sleep(1)  # Small delay between queries
    else:
        logger.info("🔍 Testing hierarchical search with LLM response generation...")
        for test_query in test_queries:
            print(f"\n{'='*80}")
            print(f"🤔 Question: {test_query}")
            print("=" * 80)
            
            # Get LLM-generated response
            llm_response = hierarchical_rag.answer_query(test_query, level1_limit=2, level2_limit=3, show_raw_results=show_raw)
            
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
