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
        
        Args:
            region_name: AWS region for Bedrock
            model_id: Bedrock model ID for text processing
        """
        self.bedrock_client = boto3.client('bedrock-runtime', region_name=region_name)
        self.model_id = model_id
        self.region_name = region_name
        
        # Initialize Milvus client (using Milvus Lite)
        self.milvus_client = MilvusClient(uri="./milvus_demo.db")
        
        # Collection names for the two levels
        self.level1_collection_name = "level1_index"
        self.level2_collection_name = "level2_index"
        
        # Embedding dimension for Titan (will be determined dynamically)
        self.embedding_dim = 1536  # Titan embedding dimension

        self.image_collection_name = "pictometry_images_index"
        self.image_embedding_dim = 1024  # amazon.titan-embed-image-v1 default
        
        logger.info(f"🔧 Initialized Hierarchical RAG with model: {model_id}")
    
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
            return f"Roofing report with {len(chunks_data)} sections covering measurements, diagrams, and property details."
    
    def create_milvus_collections(self):
        """Create Milvus collections using simple approach"""
        logger.info("Creating Milvus collections for hierarchical indexing")
        
        # Drop existing collections if they exist
        if self.milvus_client.has_collection(self.level1_collection_name):
            self.milvus_client.drop_collection(self.level1_collection_name)
        if self.milvus_client.has_collection(self.level2_collection_name):
            self.milvus_client.drop_collection(self.level2_collection_name)
        if self.milvus_client.has_collection(self.image_collection_name):
            self.milvus_client.drop_collection(self.image_collection_name)
        
        # Create Level 1 collection (Document summaries)
        self.milvus_client.create_collection(
            collection_name=self.level1_collection_name,
            dimension=self.embedding_dim,
            metric_type="COSINE",
            consistency_level="Bounded"
        )
        
        # Create Level 2 collection (Chunks)
        self.milvus_client.create_collection(
            collection_name=self.level2_collection_name,
            dimension=self.embedding_dim,
            metric_type="COSINE",
            consistency_level="Bounded"
        )

        # Create Image collection (Pictometry images)
        self.milvus_client.create_collection(
            collection_name=self.image_collection_name,
            dimension=self.image_embedding_dim,
            metric_type="COSINE",
            consistency_level="Bounded"
        )
        
        logger.info(f"✅ Created Level 1 collection: {self.level1_collection_name}")
        logger.info(f"✅ Created Level 2 collection: {self.level2_collection_name}")
        logger.info(f"✅ Created Image collection: {self.image_collection_name}")
    
    def build_level1_index(self, documents_data: List[Dict]) -> None:
        """
        Build Level 1 Index (Summary/Metadata Index)
        
        Args:
            documents_data: List of documents with their chunks
        """
        logger.info("🏗️ Building Level 1 Index (Summary/Metadata Index)")
        
        # Create collections first
        self.create_milvus_collections()
        
        level1_data = []
        
        for i, doc in enumerate(tqdm(documents_data, desc="Building Level 1 Index")):
            try:
                # Extract document metadata
                doc_id = doc.get("doc_id", f"doc_{i}")
                source_file = doc.get("source_file", "unknown.pdf")
                chunks = doc.get("chunks", [])
                
                # Extract address from chunks
                address = "Unknown Address"
                date = "Unknown Date"
                
                for chunk in chunks:
                    if chunk.get("section") == "Report Header":
                        data = chunk.get("data", {})
                        if "property_address" in data:
                            address = data["property_address"]
                        if "date" in data:
                            date = data["date"]
                        break
                
                print(f"address --------->: {address}")
                #print(f"date --------->: {chunks}")

                # Create document summary
                summary = self.create_document_summary(chunks)
                
                # Collect all chunk IDs
                chunk_ids = [chunk.get("chunk_id", "") for chunk in chunks]
                chunk_ids_str = ",".join(chunk_ids)

                print(f"chunk_ids_str --------->: {chunk_ids_str}")
                
                # Generate embedding for summary
                summary_embedding = self.titan_embed_text(address)

                #print(f"summary_embedding --------->: {summary}")
               
                level1_data.append({
                    "id": i,
                    "vector": summary_embedding,
                    "doc_id": doc_id,
                    "summary": summary,
                    "chunk_ids": chunk_ids_str,
                    "address": address,
                    "source_file": source_file,
                    "date": date
                })
                
                #print(f"summary json--------->: {level1_data}")
                

                logger.info(f"Created Level 1 entry for {doc_id}: {address}")
                
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
    
    def build_level2_index(self, documents_data: List[Dict]) -> None:
        """
        Build Level 2 Index (Chunk Index)
        
        Args:
            documents_data: List of documents with their chunks
        """
        logger.info("🏗️ Building Level 2 Index (Chunk Index)")
        
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
            
            # Step 2: Search Level 1 (Summaries)
            logger.info("📊 Searching Level 1 (Summary Index)...")
            res1 = self.milvus_client.search(
                collection_name=self.level1_collection_name,
                data=[query_vec],
                limit=level1_limit,
                output_fields=["doc_id", "chunk_ids", "address", "summary"]
            )
            
            if not res1 or not res1[0]:
                logger.warning("No results found in Level 1 Index")
                return []
            
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
                
                
                # Parse chunk IDs
                chunk_ids_str = hit.get("chunk_ids", "")
                print(f"Retrieved chunk_ids_str --------->: {chunk_ids_str}")

                #lets include chunk from 1st index
                if chunk_ids_str:
                    retrieved_chunk_ids.extend(chunk_ids_str.split(",")[1:])

                print(f"retrieved_chunk_ids --------->: {retrieved_chunk_ids}")
            
            logger.info(f"📋 Found {len(level1_docs)} relevant documents with {len(retrieved_chunk_ids)} total chunks")
            
            # Step 3: Search Level 2 (Chunks within retrieved docs)
            logger.info("📊 Searching Level 2 (Chunk Index)...")
            
            if not retrieved_chunk_ids:
                logger.warning("No chunk IDs found from Level 1 search")
                return []


            # Step 3: Search Level 2 (Chunks within retrieved docs)
            #expr = f'chunk_id in {retrieved_chunk_ids}'

            # Safer way: use repr() to auto-quote strings, then replace single quotes with double quotes
            expr = f'chunk_id in [{",".join([repr(cid) for cid in retrieved_chunk_ids]).replace("'", "\"")}]'

            # For simplicity, let's search all chunks and then filter by relevance
            res2 = self.milvus_client.search(
                collection_name=self.level2_collection_name,
                data=[query_vec],
                limit=level2_limit,  # Get more results to filter
                filter=expr,
                output_fields=["chunk_text", "section", "doc_id", "chunk_type", "chunk_id"]
            )
            
            if not res2 or not res2[0]:
                logger.warning("No results found in Level 2 Index")
                return []
            
            # Step 4: Filter results to only include chunks from relevant documents
            final_results = []
            relevant_doc_ids = [doc["doc_id"] for doc in level1_docs]

            print(f"relevant_doc_ids --------->: {relevant_doc_ids}")
            # print(f"res2 --------->: {res2}")
            print(f"level1_docs --------->: {len(res2[0])}")

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
                    
                    final_results.append(chunk_info)
                    
                    # if len(final_results) >= level2_limit:
                    #     break
            
            print(f"final_results --------->: {final_results}")
            print(f"level2_limit --------->: {level2_limit}")
            print(f"len(final_results) --------->: {len(final_results)}")
            
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
        
        context = "\n".join(context_parts)
        
        # Create comprehensive prompt for LLM
        prompt = f"""You are a helpful assistant analyzing roofing report data. Based on the retrieved information below, provide a comprehensive and accurate answer to the user's question in JSON format.

User Question: {query}

Retrieved Information:
{context}

Instructions:
1. Return your response as a valid JSON object with the following structure
2. Answer the user's question directly and accurately
3. Include specific measurements, addresses, and details when available
4. If multiple properties are mentioned, clearly distinguish between them
5. Organize the information logically (e.g., by property, by measurement type)
6. Use a conversational but professional tone
7. If the information contains diagrams or images, mention what they show

Required JSON Response Format:
{{
    "answer": "Direct answer to the user's question",
    "summary": "Brief summary of key findings",
    "properties": [
        {{
            "address": "Property address",
            "key_measurements": {{
                "total_area": "Total roof area if available",
                "roof_facets": "Number of roof facets if available",
                "predominant_pitch": "Main roof pitch if available",
                "obstructions": "Number of obstructions if available"
            }},
            "additional_details": ["List of other relevant details"]
        }}
    ],
    "diagrams_mentioned": ["List of diagrams or images referenced"],
    "confidence": "high/medium/low based on completeness of information",
    "notes": "Any limitations or additional context"
}}

Provide only the JSON response, no additional text:"""

        try:
            body = {
                "anthropic_version": "bedrock-2023-05-31",
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
            llm_text = response_body['content'][0]['text'].strip()
            
            # Try to parse as JSON
            try:
                json_response = json.loads(llm_text)
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
        Format JSON response for better display
        
        Args:
            json_response: JSON string response from LLM
            
        Returns:
            Formatted string for display
        """
        try:
            data = json.loads(json_response)
            
            # Create a nicely formatted display
            formatted_parts = []
            
            # Main answer
            if "answer" in data:
                formatted_parts.append(f"📋 **Answer**: {data['answer']}")
            
            # Summary
            if "summary" in data:
                formatted_parts.append(f"📊 **Summary**: {data['summary']}")
            
            # Properties
            if "properties" in data and data["properties"]:
                formatted_parts.append("\n🏠 **Properties**:")
                for i, prop in enumerate(data["properties"], 1):
                    formatted_parts.append(f"  {i}. **Address**: {prop.get('address', 'N/A')}")
                    
                    if "key_measurements" in prop:
                        measurements = prop["key_measurements"]
                        formatted_parts.append("     📏 **Key Measurements**:")
                        for key, value in measurements.items():
                            if value and value != "N/A":
                                formatted_parts.append(f"       • {key.replace('_', ' ').title()}: {value}")
                    
                    if "additional_details" in prop and prop["additional_details"]:
                        formatted_parts.append("     📝 **Additional Details**:")
                        for detail in prop["additional_details"]:
                            formatted_parts.append(f"       • {detail}")
            
            # Diagrams
            if "diagrams_mentioned" in data and data["diagrams_mentioned"]:
                formatted_parts.append(f"\n📊 **Diagrams/Images**: {', '.join(data['diagrams_mentioned'])}")
            
            # Confidence and notes
            if "confidence" in data:
                confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(data["confidence"], "⚪")
                formatted_parts.append(f"\n{confidence_emoji} **Confidence**: {data['confidence'].title()}")
            
            if "notes" in data and data["notes"]:
                formatted_parts.append(f"💡 **Notes**: {data['notes']}")
            
            return "\n".join(formatted_parts)
            
        except json.JSONDecodeError:
            # If not JSON, return as-is
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

    def build_image_index(self, folder_path: str = "pictometry_images") -> None:
        """
        Traverse `folder_path` (expects subfolders named like "<lat>_<lon>") and for each
        subfolder load `stiched_image_embedings` (JSON array of floats for Titan image
        embeddings, 1024-dim) and insert into the Milvus image collection in a single batch.
        """
        import os

        logger.info("🏗️ Building Image Index (Pictometry Images)")

        # Ensure image collection exists without dropping other collections
        try:
            if not self.milvus_client.has_collection(self.image_collection_name):
                self.milvus_client.create_collection(
                    collection_name=self.image_collection_name,
                    dimension=self.image_embedding_dim,
                    metric_type="COSINE",
                    consistency_level="Bounded",
                )
                logger.info(f"✅ Created Image collection: {self.image_collection_name}")
        except Exception as e:
            logger.error(f"Error ensuring image collection: {e}")
            return

        if not os.path.isdir(folder_path):
            logger.error(f"Image root directory not found: {folder_path}")
            return

        image_data: List[Dict[str, Any]] = []
        counter = 0

        subfolders = [
            os.path.join(folder_path, name)
            for name in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, name))
        ]

        for sub in tqdm(subfolders, desc="Building Image Index"):
            try:
                embed_path = os.path.join(sub, "stiched_image_embedings")
                if not os.path.isfile(embed_path):
                    continue

                # Load embedding
                with open(embed_path, "r", encoding="utf-8") as f:
                    embedding = json.load(f)

                # Extract simple metadata from folder name
                base = os.path.basename(sub)
                try:
                    lat_str, lon_str = base.split("_", 1)
                except ValueError:
                    lat_str, lon_str = "?", "?"
                text = (
                    "Stitched pictometry image in order (top,east,west,north,south) for "
                    f"lat:{lat_str} lon:{lon_str}"
                )

                image_data.append({
                    "id": counter,
                    "vector": embedding,
                    "lat": lat_str,
                    "lon": lon_str,
                    "folder": sub,
                    "text": text,
                })
                counter += 1
            except Exception as e:
                logger.error(f"Error processing image folder {sub}: {e}")
                continue

        if image_data:
            try:
                self.milvus_client.insert(
                    collection_name=self.image_collection_name,
                    data=image_data,
                )
                logger.info(f"✅ Image Index built with {len(image_data)} images")
            except Exception as e:
                logger.error(f"Failed inserting image data into Milvus: {e}")
        else:
            logger.warning("No image embeddings found to insert into Image Index")

    
def load_agentic_rag_output() -> List[Dict]:
    """
    Load the output from the agentic RAG system and convert to hierarchical format
    
    Returns:
        List of documents in hierarchical format
    """
    logger.info("📂 Loading agentic RAG output...")
    
    # Load the LLM response files
    llm_output_dir = Path("llm_output")
    if not llm_output_dir.exists():
        logger.error("llm_output directory not found")
        return []
    
    documents = []
    
    # Process each LLM response file
    for response_file in llm_output_dir.glob("*_llm_response.json"):
        try:
            logger.info(f"Processing {response_file.name}")
            
            with open(response_file, 'r') as f:
                chunks = json.load(f)
            
            # Extract document ID from filename
            doc_id = response_file.stem.replace("_llm_response", "")
            source_file = response_file.name
            
            # Create document entry
            doc_entry = {
                "doc_id": doc_id,
                "source_file": source_file,
                "chunks": chunks
            }
            
            documents.append(doc_entry)
            logger.info(f"Loaded {len(chunks)} chunks from {source_file}")
            
        except Exception as e:
            logger.error(f"Error loading {response_file}: {str(e)}")
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
        logger.error("No documents found. Please run the agentic RAG system first.")
        return
    
    # Initialize hierarchical RAG system
    hierarchical_rag = HierarchicalRAG()
    
    if build_index:
        # Build hierarchical indices
        logger.info("🏗️ Building hierarchical indices...")
        hierarchical_rag.build_level1_index(documents)
        hierarchical_rag.build_level2_index(documents)
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
            "Show me the imagery and diagrams"
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
  # Build indices and run default queries
  python src/hierarchical_rag_working.py
  
  # Search with a specific query (will build indices first)
  python src/hierarchical_rag_working.py --query "What is the roof area?"
  
  # Search without rebuilding indices (faster if indices already exist)
  python src/hierarchical_rag_working.py --query "What are the obstructions?" --no-build
  
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
    
    args = parser.parse_args()
    
    # Handle build-only mode
    if args.build_only:
        main(query=None, build_index=True)
        print("\n✅ Indices built successfully! You can now run queries with --no-build flag.")
        sys.exit(0)
    
    # Run main function with parsed arguments
    build_index = not args.no_build
    main(query=args.query, build_index=build_index, show_raw=args.show_raw, raw_only=args.raw_only, json_only=args.json_only)
