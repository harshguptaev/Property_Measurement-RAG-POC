#!/usr/bin/env python3
"""
Script to load processed chunks from Final_Chunks directory into Milvus vector store.
This will populate the vector store with the existing processed documents.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.config import config
from src.vector_store import VectorStoreManager
from src.bedrock_client import create_bedrock_embeddings
from langchain.schema import Document

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def load_chunks_from_directory(chunks_dir: str = "Final_Chunks") -> List[Document]:
    """
    Load all processed chunks from Final_Chunks directory.
    
    Args:
        chunks_dir: Directory containing the processed JSON chunks
        
    Returns:
        List of Document objects ready for vector store
    """
    chunks_path = Path(chunks_dir)
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks directory not found: {chunks_path}")
    
    all_documents = []
    
    # Get all JSON files
    json_files = list(chunks_path.glob("*.json"))
    logging.info(f"Found {len(json_files)} chunk files to load")
    
    for json_file in json_files:
        try:
            logging.info(f"Loading chunks from {json_file.name}")
            
            with open(json_file, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            
            # Handle different file formats
            if isinstance(chunks_data, dict) and 'text' in chunks_data:
                # Format: {"text": [...]}
                chunks_data = chunks_data['text']
            elif not isinstance(chunks_data, list):
                logging.warning(f"Unexpected format in {json_file.name}, skipping")
                continue
            
            file_documents = []
            
            for chunk_dict in chunks_data:
                if not isinstance(chunk_dict, dict):
                    continue
                
                # Build page content based on chunk type
                chunk_type = chunk_dict.get('type', 'text')
                
                if chunk_type == 'text':
                    # For text chunks, use the actual content
                    chunk_data = chunk_dict.get('data', {})
                    if 'content' in chunk_data:
                        page_content = chunk_data['content']
                    else:
                        # Fallback to JSON representation
                        page_content = json.dumps({
                            "section": chunk_dict.get('section', ''),
                            **chunk_data
                        }, ensure_ascii=False)
                elif chunk_type == 'table':
                    # For table chunks, create readable table content
                    chunk_data = chunk_dict.get('data', {})
                    page_content = f"Table from section: {chunk_dict.get('section', '')}\n"
                    if 'content' in chunk_data:
                        page_content += chunk_data['content']
                    elif 'table_data' in chunk_data:
                        page_content += str(chunk_data['table_data'])
                    else:
                        page_content += json.dumps(chunk_data, ensure_ascii=False)
                else:  # image
                    # For image chunks, use description
                    chunk_data = chunk_dict.get('data', {})
                    page_content = f"Image from section: {chunk_dict.get('section', '')}"
                    if 'description' in chunk_data:
                        page_content += f"\nDescription: {chunk_data['description']}"
                
                # Create metadata
                metadata = {
                    'type': chunk_type,
                    'extraction_method': 'docling_processed',
                    'chunk_id': chunk_dict.get('chunk_id', ''),
                    'section': chunk_dict.get('section', ''),
                    'source_file': json_file.stem,
                    'file_name': json_file.stem
                }
                
                # Add chunk-specific metadata
                chunk_data = chunk_dict.get('data', {})
                metadata.update(chunk_data)
                
                # Create document
                doc = Document(page_content=page_content, metadata=metadata)
                file_documents.append(doc)
            
            all_documents.extend(file_documents)
            logging.info(f"Loaded {len(file_documents)} chunks from {json_file.name}")
            
        except Exception as e:
            logging.error(f"Error loading chunks from {json_file.name}: {e}")
            continue
    
    logging.info(f"Total loaded documents: {len(all_documents)}")
    return all_documents

def main():
    """Main function to load chunks into vector store."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🔄 Loading processed chunks into vector store...")
    
    try:
        # Setup configuration
        bedrock_config = config.get_bedrock_config()
        vector_config = config.get_vector_store_config()
        
        # Create embeddings
        logger.info("Creating embeddings client...")
        embeddings = create_bedrock_embeddings(bedrock_config)
        
        # Create vector store manager
        logger.info(f"Setting up vector store: {vector_config['store_type']}")
        vector_store_manager = VectorStoreManager(
            store_type=vector_config["store_type"],
            collection_name=vector_config["collection_name"],
            embeddings=embeddings
        )
        
        # Load documents from Final_Chunks
        logger.info("Loading documents from Final_Chunks directory...")
        documents = load_chunks_from_directory()
        
        if not documents:
            logger.warning("No documents found to load!")
            return
        
        # Add documents to vector store
        logger.info(f"Adding {len(documents)} documents to vector store...")
        vector_store_manager.add_documents(documents)
        
        # Get final count
        final_count = vector_store_manager.get_count()
        logger.info(f"✅ Successfully loaded {final_count} documents into vector store!")
        
        # Test a simple query
        logger.info("Testing vector store with a sample query...")
        try:
            results = vector_store_manager.similarity_search("roof", k=3)
            logger.info(f"Sample query returned {len(results)} results")
            if results:
                logger.info(f"First result: {results[0].metadata.get('source_file', 'Unknown')}")
        except Exception as e:
            logger.warning(f"Sample query failed: {e}")
        
        logger.info("🎉 Vector store is now ready for queries!")
        
    except Exception as e:
        logger.error(f"Error loading chunks: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()